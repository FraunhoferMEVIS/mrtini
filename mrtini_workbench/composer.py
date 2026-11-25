"""Editable held-gradient research acquisitions; all moments cycles/m, time s.
Not a hardware safety checker. Nominal trajectory follows the Sequence convention:
EXCIT resets primary coherence, REFOC conjugates it. All states are propagated by MRTini.
"""
import json, math
import numpy as np
from native_sequences import Design, RASTER, LOBE


def number(block, key, default, lo, hi):
    try: v=float(block.get(key,default))
    except (ValueError,TypeError): raise ValueError(f'{key} must be numeric')
    if not math.isfinite(v) or not lo<=v<=hi: raise ValueError(f'{key} must be in [{lo}, {hi}]')
    return v


def compose(p):
    try: recipe=json.loads(p.custom_recipe)
    except (ValueError,TypeError): raise ValueError('Invalid sequence recipe JSON')
    if isinstance(recipe,dict) and recipe.get('mode')=='timed':
        from timed_design import compile_timed
        return compile_timed(p,recipe)
    if not isinstance(recipe,dict) or not isinstance(recipe.get('blocks'),list): raise ValueError('Recipe needs a blocks array')
    blocks=recipe['blocks']; n=p.resolution; th=p.slice_mm/1000; dk=1000/p.fov_mm; tr=p.tr_ms/1000
    if not 1<=len(blocks)<=128 or not all(isinstance(b,dict) for b in blocks): raise ValueError('Use 1–128 blocks')
    if blocks[0].get('type') not in ('excitation','inversion'): raise ValueError('First block must be excitation or inversion preparation')
    types={'excitation','inversion','refocusing','delay','gradient','parallel_gradient','readout'}
    if any(b.get('type') not in types for b in blocks): raise ValueError('Unknown block type')
    read_blocks=[b for b in blocks if b['type']=='readout']
    if not read_blocks: raise ValueError('Add at least one readout')
    lines=0
    for b in read_blocks:
        v=number(b,'lines',1,1,256)
        if v!=int(v): raise ValueError('Readout lines must be an integer')
        lines+=int(v)
    auto=math.ceil(n/lines)
    q=number(recipe,'shots',0,0,512)
    if q!=int(q): raise ValueError('Shot count must be an integer')
    shots=int(q) or auto
    if shots*len(blocks)>8192 or shots*lines*n>2_000_000: raise ValueError('Sequence exceeds the workbench design budget')
    step=number(recipe,'phase_step_deg',0,-360,360);spoiling=recipe.get('quadratic_phase',False)
    if type(recipe.get('interleaved',False)) is not bool:raise ValueError('Interleaved ordering must be boolean')
    if type(spoiling) is not bool: raise ValueError('Quadratic RF phase must be boolean')
    pulses=[]; grads=[]; expected=[]; adc=[]; t=0.; k=np.zeros(3); durations=[]; entry=None; line_id=0
    def lobe(dur,area):
        nonlocal t,k
        area=np.asarray(area,float)
        if dur<=0: raise ValueError('Nonpositive gradient interval')
        if np.any(area):grads.append((t,t+dur,area/dur))
        k=k+area;t+=dur
    for shot in range(shots):
        shot_start=shot*tr;t=shot_start; phaseoff=np.deg2rad((step*(shot*(shot+1)/2 if spoiling else shot))%360)
        for block in blocks:
            typ=block['type']
            if typ in ('excitation','inversion','refocusing'):
                exc=typ!='refocusing'; dur=round(number(block,'duration_ms',p.rf_ms,.1,20)*1e-3/RASTER)*RASTER;cells=round(dur/RASTER);half=dur/2
                flip=number(block,'flip_deg',180 if typ!='excitation' else 90,0,360);phase=(np.deg2rad(number(block,'phase_deg',0 if exc else 90,-360,360))+phaseoff)%(2*np.pi)
                tbw=number(block,'tbw',p.tbw,1,16);gz=tbw/(dur*th);sel=gz*half;crusher=number(block,'crusher',1.5,0,20)/th
                balanced=block.get('balanced',False)
                if type(balanced) is not bool:raise ValueError('Balanced slice prephaser must be boolean')
                if not exc:lobe(LOBE,[0,0,crusher-sel])
                elif balanced:lobe(LOBE,[0,0,-sel])
                a=t;c=a+half;b=a+dur;u=(np.arange(cells)+.5)/cells-.5;env=np.sinc(tbw*u)*(.54+.46*np.cos(2*np.pi*u));amp=env*np.deg2rad(flip)/(2*np.pi*RASTER*env.sum())
                span=dict(kind='bloch',t=a+np.arange(cells+1)*RASTER,t0=a,t1=b,rf_center=c,rf_amp=np.r_[amp,0.],rf_phase=np.full(cells+1,phase),rf_freq=np.zeros(cells+1),gx=np.zeros(cells+1),gy=np.zeros(cells+1),gz=np.r_[np.full(cells,gz),0.],adc_active=np.zeros(cells+1),adc_phase=np.zeros(cells+1),adc_freq=np.zeros(cells+1),flip_deg=flip)
                grads.append((a,b,np.array([0.,0.,gz])));k+=np.array([0.,0.,sel]);k=np.zeros(3) if exc else -k;k+=np.array([0.,0.,sel]);t=b
                entry=dict(centre=c,span=span,exc=exc,phase=phase,reads=[]);pulses.append(entry);durations.append(dur*1000)
                lobe(LOBE,[0,0,-sel if exc else crusher-sel])
            elif typ=='delay': t+=number(block,'duration_ms',2,0,10000)*1e-3
            elif typ=='gradient': lobe(number(block,'duration_ms',1,.01,1000)*1e-3,[number(block,a,0,-100000,100000) for a in ('kx','ky','kz')])
            elif typ=='parallel_gradient':
                # Three independently timed gradient axes run within one block. RF/ADC overlap is not implied.
                base=t;end=0.
                for axis,name in enumerate(('kx','ky','kz')):
                    dur=number(block,name+'_duration_ms',1,.01,1000)*1e-3;offset=number(block,name+'_offset_ms',0,0,1000)*1e-3;area=number(block,name,0,-100000,100000)
                    if area:
                        g=np.zeros(3);g[axis]=area/dur;grads.append((base+offset,base+offset+dur,g));k[axis]+=area;end=max(end,offset+dur)
                if end==0:raise ValueError('Parallel gradients need at least one nonzero moment')
                t=base+end
            else:
                count=int(number(block,'lines',1,1,256));style=block.get('style','cartesian');dwell=number(block,'dwell_us',p.dwell_us,1,100)*1e-6
                if style not in ('cartesian','epi','radial','spiral','spiral_in'):raise ValueError('Unknown readout geometry')
                alternating=block.get('alternating',False) or style=='epi';rewind=block.get('rewind',True)
                if type(alternating) is not bool or type(rewind) is not bool:raise ValueError('Readout switches must be boolean')
                gap=number(block,'gap_ms',1,0,1000)*1e-3;phase=np.deg2rad(number(block,'receiver_deg',0,-360,360));track=block.get('track_rf_phase',False)
                if type(track) is not bool:raise ValueError('Receiver tracking must be boolean')
                if track:phase-=entry['phase']
                for line in range(count):
                    if not q and line_id>=n:break
                    reverse=block.get('reverse',False)
                    if type(reverse) is not bool:raise ValueError('Reverse readout must be boolean')
                    pol=-1 if bool(reverse) ^ bool(alternating and line%2) else 1
                    if style in ('cartesian','epi'):
                        x=(np.arange(n)-n//2)*dk
                        if pol<0:x=x[::-1]
                        ky_index=(shot+(line_id-shot*lines)*shots) % n if recipe.get('interleaved',False) else line_id%n; ky=(ky_index-n//2)*dk;targets=np.c_[x,np.full(n,ky),np.zeros(n)]
                    elif style=='radial':
                        theta=np.pi*line_id/(shots*lines if q else n)+np.deg2rad(number(block,'angle_deg',0,-360,360));x=(np.arange(n)-n//2)*dk*pol;targets=np.c_[x*np.cos(theta),x*np.sin(theta),np.zeros(n)]
                    else:
                        # Discrete Archimedean interleaves; generic adjoint, no density-compensation claim.
                        u=np.linspace(0,1,n);theta=2*np.pi*(number(block,'turns',2,.25,20)*u+line_id/(shots*lines if q else n))+np.deg2rad(number(block,'angle_deg',0,-360,360));r=u*n*dk/2;targets=np.c_[r*np.cos(theta),r*np.sin(theta),np.zeros(n)]
                        if (style=='spiral_in') ^ (pol<0):targets=targets[::-1]
                    # Position at first ADC; each subsequent held gradient connects samples exactly.
                    first_step=targets[1]-targets[0];lobe(LOBE,targets[0]-.5*first_step-k);lobe(dwell/2,.5*first_step)
                    entry['reads'].append((t,phase));adc.append(t);expected.append(targets[0,:2]);
                    for j in range(1,n):
                        lobe(dwell,targets[j]-targets[j-1]);entry['reads'].append((t,phase));adc.append(t);expected.append(targets[j,:2])
                    lobe(dwell/2,.5*(targets[-1]-targets[-2]));line_id+=1
                    if rewind:lobe(LOBE,-k)
                    if line<count-1:t+=gap
        if t>shot_start+tr+1e-9:raise ValueError(f'Shot {shot+1} needs {(t-shot_start)*1000:.2f} ms but TR is {p.tr_ms:.2f} ms')
    if not adc:raise ValueError('No ADC samples in design')
    final=shots*tr
    # native rep boundaries are RF centres. No ADC may coincide with a boundary.
    if final<=pulses[-1]['centre']:raise ValueError('Final RF pulse does not fit in TR')
    return Design(pulses,grads,final,np.asarray(expected),np.asarray(adc),max(durations))
