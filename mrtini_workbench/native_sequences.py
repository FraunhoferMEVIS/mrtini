# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native finite-RF research sequences, using the recovered thesis mini_sequences.py
centre-to-centre event convention. Held gradient lobes: not scanner-ready waveforms.
All gradient amplitudes Hz/m, RF Hz, moments cycles/m, times seconds.
"""
from dataclasses import dataclass
import numpy as np
from physics import Protocol, line_plan

RASTER=10e-6
LOBE=1e-3
@dataclass
class Design:
    pulses: list
    gradients: list
    final: float
    expected_k: np.ndarray
    adc_times: np.ndarray
    actual_rf_ms: float
    def area(self,a,b):
        return sum((max(0.,min(b,hi)-max(a,lo))*g for lo,hi,g in self.gradients),np.zeros(3))


def design(p:Protocol):
    p.check()
    if p.sequence=='custom':
        from composer import compose
        return compose(p)
    n=p.resolution; dk=1000/p.fov_mm; th=p.slice_mm*1e-3
    tr=p.tr_ms*1e-3; te=p.te_ms*1e-3; esp=p.esp_ms*1e-3; dwell=p.dwell_us*1e-6
    cells=round(p.rf_ms*1e-3/RASTER); dur=cells*RASTER; half=dur/2
    gz=p.tbw/(dur*th); sel=gz*half; crusher=1.5/th
    u=(np.arange(cells)+.5)/cells-.5
    envelope=np.sinc(p.tbw*u)*(.54+.46*np.cos(2*np.pi*u))
    pulses=[]; grads=[]; expected=[]; adc=[]
    def lobe(a,b,area):
        if b<=a:raise ValueError('Nonpositive gradient interval')
        area=np.asarray(area,float)
        if np.any(area):grads.append((float(a),float(b),area/(b-a)))
    def pulse(c,flip,phase,exc):
        a=c-half;b=c+half; t=a+np.arange(cells+1)*RASTER
        amp=envelope*np.deg2rad(flip)/(2*np.pi*RASTER*envelope.sum())
        span=dict(kind='bloch',t=t,t0=float(a),t1=float(b),rf_center=float(c),rf_amp=np.r_[amp,0.],rf_phase=np.full(cells+1,phase),rf_freq=np.zeros(cells+1),gx=np.zeros(cells+1),gy=np.zeros(cells+1),gz=np.r_[np.full(cells,gz),0.],adc_active=np.zeros(cells+1),adc_phase=np.zeros(cells+1),adc_freq=np.zeros(cells+1),flip_deg=float(flip))
        entry=dict(centre=c,span=span,exc=exc,phase=phase,reads=[])
        pulses.append(entry);lobe(a,b,[0,0,2*sel]);
        if exc:lobe(b,b+LOBE,[0,0,-sel])
        else:
            lobe(a-LOBE,a,[0,0,crusher-sel]);lobe(b,b+LOBE,[0,0,crusher-sel])
        return entry
    def require(a,b,label):
        if a>b+1e-10:raise ValueError(f'{label} does not fit by {(a-b)*1000:.3f} ms. Increase TE, TR or echo spacing, or reduce RF duration/ADC dwell.')
    def read(entry,t,ky,pol,previous=None):
        # ADC at event ends; integer k-grid with the kx=0 sample at t.
        kx=np.arange(n)-n//2
        if pol<0:kx=kx[::-1]
        idx=int(np.flatnonzero(kx==0)[0]); times=t+(np.arange(n)-idx)*dwell
        start=float(times[0]-dwell/2);end=float(times[-1]+dwell/2)
        left=np.array([(kx[0]-.5*pol)*dk,ky*dk,0]);right=np.array([(kx[-1]+.5*pol)*dk,ky*dk,0]);
        if previous is None:
            require(entry['centre']+half+LOBE,start-LOBE,'RF to readout prephaser');lobe(start-LOBE,start,left)
        else:
            endprev,rightprev=previous;require(endprev,start-LOBE,'EPI phase-encoding blip');lobe(start-LOBE,start,left-rightprev)
        lobe(start,end,[pol*n*dk,0,0]);entry['reads'].extend((float(x),-entry['phase'] if p.sequence=='flash' else 0.) for x in times)
        expected.extend(np.c_[kx*dk,np.full(n,ky*dk)]);adc.extend(times)
        return end,right
    def rewind(end,right,spoil=False):
        lobe(end,end+LOBE,-right+np.array([0,0,4/th if spoil else 0]));return end+LOBE
    rows=line_plan(p); shots=sorted(set(x['shot'] for x in rows));origin=half+LOBE
    for shot in shots:
        lines=[x for x in rows if x['shot']==shot];start=origin+shot*tr
        phase=np.deg2rad((117*shot*(shot+1)/2)%360) if p.sequence=='flash' else 0.
        exc=pulse(start,p.flip_deg,phase,True);next_start=start+tr-half
        if p.sequence=='flash':
            end,right=read(exc,start+te,lines[0]['ky'],1);require(rewind(end,right,True),next_start,'FLASH TR')
        elif p.sequence=='rare':
            last_end=start+half+LOBE
            for row in lines:
                echo=start+row['time_ms']*1e-3;c=echo-esp/2
                require(last_end,c-half-LOBE,'RARE refocusing crusher')
                ref=pulse(c,p.refocus_deg,np.pi/2,False);end,right=read(ref,echo,row['ky'],1);last_end=rewind(end,right)
            lobe(last_end,last_end+LOBE,[0,0,4/th]);require(last_end+LOBE,next_start,'RARE TR')
        else:
            c=start+te/2;require(start+half+LOBE,c-half-LOBE,'EPI refocusing crusher');ref=pulse(c,p.refocus_deg,np.pi/2,False)
            prev=None
            for row in lines:prev=read(ref,start+row['time_ms']*1e-3,row['ky'],row['polarity'],prev)
            require(rewind(*prev,True),next_start,'EPI TR')
    final=origin+len(shots)*tr
    # Explicit chronological non-overlap validation. All combined x/y/z lobes
    # are represented by a single vector, so overlaps indicate a design error.
    grads.sort(key=lambda x:x[0]);
    for (a,b,_),(c,d,_) in zip(grads,grads[1:]):require(b,c,'Gradient blocks overlap')
    return Design(pulses,grads,final,np.asarray(expected),np.asarray(adc),dur*1000)


def build_native_sequence(p, *, native=None, return_design=False):
    """Return the installed MRTini Sequence, not an alternative simulator."""
    if isinstance(p,dict):p=Protocol(**p)
    if native is None:
        import torch
        from src.sequence.sequence import Sequence,PulseUsage
    else:torch,Sequence,PulseUsage=native
    d=design(p); centres=[x['centre'] for x in d.pulses];bounds=list(zip(centres,centres[1:]+[d.final]))
    seq=Sequence(normalized_grads=False,mrtini_hybrid=[],mrtini_rep_bounds=bounds,mrtini_spans_by_rep=[])
    for r,(entry,(a,b)) in enumerate(zip(d.pulses,bounds)):
        span=dict(entry['span']);usage=PulseUsage.EXCIT if entry['exc'] else PulseUsage.REFOC;span.update(rep_index=r,pulse_type=usage)
        # Keep raw waveforms, gradient halves and physical event times exactly as in the thesis builder.
        samples={round(t,12):phase for t,phase in entry['reads']}
        local=[(lo,hi,g) for lo,hi,g in d.gradients if lo<b and hi>a]
        times=[a,b,*samples];times.extend(x for lo,hi,_ in local for x in (lo,hi) if a<x<b)
        times=np.unique(np.round(times,12));dt=np.diff(times)
        if any(x<=round(a,12) or x>=round(b,12) for x in samples):raise ValueError('ADC outside its RF-centre repetition')
        rep=seq.new_rep(len(dt));rep.event_time[:]=torch.as_tensor(dt,dtype=torch.float32)
        moments=np.zeros((len(dt),3))
        for lo,hi,g in local:
            moments+=np.maximum(0.,np.minimum(times[1:],hi)-np.maximum(times[:-1],lo))[:,None]*g[None,:]
        rep.gradm[:]=torch.as_tensor(moments,dtype=torch.float32)
        rep.gradm_z_eff=rep.gradm[:,2].clone();rep.adc_usage.zero_();rep.adc_phase.zero_()
        rep.pulse.angle=torch.tensor(np.deg2rad(span['flip_deg']),dtype=torch.float32);rep.pulse.phase=torch.tensor(entry['phase'],dtype=torch.float32);rep.pulse.usage=usage;rep.pulse.selective=True
        for i,t in enumerate(times[1:]):
            if t in samples:rep.adc_usage[i]=1;rep.adc_phase[i]=samples[t]
        rep.raw_waveform=span;seq.mrtini_spans_by_rep.append([len(seq.mrtini_hybrid)]);seq.mrtini_hybrid.append(span)
    return (seq,d) if return_design else seq
