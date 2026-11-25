"""Read-only analysis of the native packed PDG and RF-centre sequence.
Graph visualisation limits affect display only, never simulation/pruning.
"""
import copy
from types import SimpleNamespace as NS
import numpy as np

COLORS={1:'water',2:'fat',3:'shared'}
REL=('++','+z','z+','zz','-+','-z')
TISSUES={'gm':('Grey matter',.8,1.56,.083,.32,.83,0), 'wm':('White matter',.7,.83,.075,.18,.65,0), 'csf':('CSF',1.,4.16,1.65,.059,3.19,0), 'fat':('Fat',1.,.37,.125,.012,.10,1)}

def physical_masks(mask,labels):
    mask=np.asarray(mask); out=np.zeros(mask.shape,dtype=np.uint8)
    for i,label in enumerate(labels):
        if label not in (0,1):raise ValueError('Unknown physical species')
        out|=(((mask>>i)&1).astype(np.uint8)<<label)
    return out

def counts(m):return {v:int(np.count_nonzero(m==k)) for k,v in COLORS.items()}

def graph_window(graph,start=0,window=6,mode='union',kind='all',limit=85,edge_limit=4500,activity='all'):
    if not all(hasattr(graph,k) for k in ('offsets','node_mask','case_species','kt_end','rows','src','mask')):raise ValueError('Packed graph unavailable')
    if mode not in ('union','water','fat','shared','water_only','fat_only') or kind not in ('all','F','Z'):raise ValueError('Invalid graph filter')
    if activity not in ('all','emitting','latent'):raise ValueError('Invalid activity filter')
    n=len(graph.offsets)-1;start=max(0,min(int(start),n-1));end=min(n,start+max(2,min(12,int(window))))
    limit=max(10,min(200,int(limit))); edge_limit=max(1,min(12000,int(edge_limit)))
    def filt(m):
        if mode=='union':return m!=0
        if mode=='water':return (m&1)!=0
        if mode=='fat':return (m&2)!=0
        return m=={'shared':3,'water_only':1,'fat_only':2}[mode]
    labels=graph.case_species; nodes=[];edges=[];layers=[];selected={};candidate_edges=0
    for layer in range(start,end):
        a,b=map(int,graph.offsets[layer:layer+2]);m=physical_masks(graph.node_mask[a:b],labels);k=graph.kind[a:b];ix=np.flatnonzero(filt(m)&((k==0) if kind=='F' else (k!=0) if kind=='Z' else True))
        # Prioritize emitting then latent amplitudes, stable by index. Include all when small.
        emitted=np.max(graph.emitted[a:b],axis=1);latent=np.max(graph.latent[a:b],axis=1)
        if activity!='all':ix=ix[(emitted if activity=='emitting' else latent)[ix]>0]
        eligible=len(ix)
        if len(ix)>limit:
            score=np.maximum(emitted[ix],latent[ix]);ix=ix[np.argsort(-score,kind='stable')[:limit]]
        sel=ix+a;selected[layer]=set(map(int,sel));layers.append(dict(layer=layer,total=b-a,counts=counts(m),eligible=eligible,shown=len(ix)))
        for j in sel:
            node=dict(id=int(j),layer=layer,local=int(j-a),kind=('F+','Z','Z0')[int(graph.kind[j])],mask=int(m[j-a]),kt=graph.kt_end[j].tolist(),start_kt=graph.kt_start[j].tolist(),emitted={('water','fat')[s]:float(graph.emitted[j,i]) for i,s in enumerate(labels)},latent={('water','fat')[s]:float(graph.latent[j,i]) for i,s in enumerate(labels)})
            nodes.append(node)
        if layer==start:continue
        previous=int(graph.offsets[layer-1]);lo,hi=int(graph.rows[a]),int(graph.rows[b]);all_masks=physical_masks(graph.mask[lo:hi],labels);layers[-1]['edge_counts']=counts(all_masks);layers[-1]['edge_total']=hi-lo
        # Each edge is displayed only when both endpoints are in the visual subset.
        for j in sel:
            l,h=int(graph.rows[j]),int(graph.rows[j+1]);parents=previous+graph.src[l:h].astype(np.int64);em=all_masks[l-lo:h-lo];valid=np.flatnonzero(filt(em)&np.isin(parents,list(selected[layer-1])))
            candidate_edges+=len(valid)
            for z in valid[:max(0,edge_limit-len(edges))]:
                e=l+int(z);edges.append(dict(source=int(parents[z]),target=int(j),mask=int(em[z]),relation=REL[int(graph.rel[e])],tap=int(graph.tap[e])))
    all_masks=physical_masks(graph.node_mask,labels)
    ranges={label:np.max(np.abs(graph.kt_end[int(graph.offsets[start]):int(graph.offsets[end])][sel]),axis=0).tolist() if np.any(sel) else [1.,1.,1.,1.] for label,sel in [('F',graph.kind[int(graph.offsets[start]):int(graph.offsets[end])]==0),('Z',graph.kind[int(graph.offsets[start]):int(graph.offsets[end])]!=0)]}
    return dict(axis_ranges=ranges,start=start,end=end,layers_total=n,case_species=list(labels),nodes=nodes,edges=edges,layers=layers,totals=counts(all_masks),total_nodes=len(all_masks),total_edges=len(graph.mask),shown_edges=len(edges),candidate_edges=int(candidate_edges),display_limited=any(x['shown']<x['eligible'] for x in layers) or candidate_edges>len(edges),mode=mode,kind=kind)

def cpu(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

def scalar(x):return float(cpu(x).reshape(-1)[0])

def sequence_window(seq=None,design=None,start=0,window=4,preview=False,shot_end=None):
    if design is not None:
        n=len(design.pulses);start=max(0,min(start,n-1));end=min(n,start+max(1,min(window,12)))
        if shot_end is not None:
            start=0;end=sum(r['centre']<shot_end-1e-10 for r in design.pulses);lo=0.;hi=float(shot_end)
        else:
            chosen=design.pulses[start:end];lo=float(chosen[0]['span']['t0']);hi=float(design.pulses[end]['span']['t0']) if end<n else float(design.final)
        chosen=design.pulses[start:end]
        # Merge simultaneous held lobes at event boundaries; no false axis resets.
        events={}
        for a,b,g in design.gradients:
            if a>=hi or b<=lo:continue
            a,b=round(max(lo,a),12),round(min(hi,b),12)
            if b<=a:continue
            g=np.asarray(g,dtype=float)/42577.47892
            events[a]=events.get(a,np.zeros(3))+g;events[b]=events.get(b,np.zeros(3))-g
        grads=[];level=np.zeros(3);prev=None
        for t,delta in sorted(events.items()):
            if prev is not None and t>prev and np.any(abs(level)>1e-10):grads.append([prev,t,*level.tolist()])
            level+=delta;prev=t
        pulses=[];adc=[]
        for i,r in enumerate(chosen,start):
            sp=r['span'];t=np.asarray(sp['t']);amp=np.asarray(sp['rf_amp']);phase=np.asarray(sp['rf_phase']);pulses.append(dict(rep=i,t=t.tolist(),real=(amp*np.cos(phase)).tolist(),imag=(amp*np.sin(phase)).tolist(),centre=r['centre'],flip=float(sp['flip_deg']),phase_deg=float(np.rad2deg(r['phase'])%360)))
            adc.extend(float(t) for t,_ in r['reads'])
        return dict(start=start,end=end,total=n,lo=lo,hi=hi,pulses=pulses,gradients=grads,adc=adc,source='Generated finite RF / held physical gradients'+(' · design only (preview)' if preview else ''),gradient_unit='mT/m')
    # Imported sequence: expose the actual integrated event moments; no inferred RF envelopes.
    n=len(seq);start=max(0,min(start,n-1));end=min(n,start+max(1,min(window,12)))
    bounds=getattr(seq,'mrtini_rep_bounds',None)
    lo=float(bounds[start][0]) if bounds is not None and len(bounds)==n else sum(scalar(r.event_time.sum()) for r in seq[:start]);t=lo;grads=[];pulses=[];adc=[]
    for i in range(start,end):
        r=seq[i];dt=cpu(r.event_time);g=cpu(r.gradm);a=cpu(r.adc_usage);times=t+np.r_[0,np.cumsum(dt,dtype=float)]
        for j,d in enumerate(dt):
            if d>0:grads.append([float(times[j]),float(times[j+1]),*(g[j]/d/(1 if seq.normalized_grads else 42577.47892)).tolist()])
        adc.extend(times[1:][a>0].tolist());pulses.append(dict(rep=i,t=[],real=[],imag=[],centre=t,flip=float(np.rad2deg(scalar(r.pulse.angle))),phase_deg=float(np.rad2deg(scalar(r.pulse.phase))%360)));t=float(times[-1])
    return dict(start=start,end=end,total=n,lo=lo,hi=t,pulses=pulses,gradients=grads,adc=adc,source='Imported RF-centre events · RF markers only; gradient event averages',gradient_unit='normalized/s' if seq.normalized_grads else 'mT/m')

def rf_context(design, rep):
    """Pulse-centred view of the actual compiled gradient events, including
    nearby lobes before/after the RF. Full interval option retains the context
    between neighbouring finite RF spans. No invented rephasers or signs.
    """
    if not 0<=rep<len(design.pulses):raise ValueError('RF repetition out of range')
    span=design.pulses[rep]['span'];a,b=float(span['t0']),float(span['t1']);centre=float(design.pulses[rep]['centre'])
    lower=float(design.pulses[rep-1]['span']['t1']) if rep else 0.
    upper=float(design.pulses[rep+1]['span']['t0']) if rep+1<len(design.pulses) else float(design.final)
    margin=max(.002,1.5*(b-a));lo=max(lower,a-margin);hi=min(upper,b+margin)
    # Include a whole Gz lobe touching the local window, rather than clipping
    # the associated rephaser/crusher midway through the held event.
    for x,y,g in design.gradients:
        if y>lo and x<hi and abs(g[2])>0:lo=max(lower,min(lo,x));hi=min(upper,max(hi,y))
    strip=sequence_window(design=design,start=max(0,rep-1),window=min(3,len(design.pulses)-max(0,rep-1)))
    pulse=next(x for x in strip['pulses'] if x['rep']==rep)
    def view(x,y):
        grads=[[max(x,u),min(y,v),gx,gy,gz] for u,v,gx,gy,gz in strip['gradients'] if v>x and u<y]
        def area(q,r):return sum(max(0.,min(r,v)-max(q,u))*gz*42577.47892 for u,v,gx,gy,gz in grads)
        return dict(lo=x,hi=y,gradients=grads,area_before=area(x,a),area_rf=area(a,b),area_after=area(b,y),adc=[t for t in strip['adc'] if x<=t<=y])
    return dict(pulse=pulse,role='excitation/preparation' if design.pulses[rep]['exc'] else 'refocusing',t0=a,t1=b,centre=centre,local=view(lo,hi),full=view(lower,upper),gradient_unit='mT/m',source='Compiled finite RF and signed held gradient events')

def probe_on_graph(engine, tissue, thickness, workspace_mb=128):
    """Replay the *same* packed graph with a one-voxel tissue probe.
    RF coefficients are a column of the actual baked bank, not an invented RF model.
    Copy only that column, use the original geometry/dephasing and zero spatial position.
    """
    import torch
    from src.mrtini.optimized import prepare_gpu_execution
    if tissue not in TISSUES:raise ValueError('Unknown tissue')
    name,pd,t1,t2,t2d,d,species=TISSUES[tissue];data=engine.data;graph=engine.graph
    if species not in graph.case_species:raise ValueError('Selected species is absent from this acquisition graph.')
    ids=engine.rf.physical;ix=torch.nonzero((ids==species)&(data.PD.reshape(-1)!=0),as_tuple=True)[0]
    if not ix.numel():raise ValueError('No contributing voxel of this species in the phantom.')
    b0=data.B0.reshape(-1);b1=data.B1.sum(0).abs().reshape(-1);target=b0[ix].median() if species else torch.zeros((),device=b0.device)
    chosen=int(ix[((b0[ix]-target).abs()/10+(b1[ix]-1).abs()*10).argmin()].item())
    def f(x):return torch.tensor(x,device=data.PD.device,dtype=torch.float32)
    obj=copy.copy(data)
    for k,v in dict(PD=f([pd]),T1=f([t1]),T2=f([t2]),T2dash=f([t2d]),D=f([d]),B0=b0[chosen:chosen+1].clone(),B1=data.B1[:,chosen:chosen+1].clone(),species_id=ids[chosen:chosen+1].clone(),voxel_pos=f([[0.,0.,0.]]),coil_sens=torch.ones((1,1),device=data.PD.device,dtype=torch.complex64),voxel_motion=None,phantom_motion=None).items():object.__setattr__(obj,k,v)
    if getattr(data,'chemical_shift',None) is not None:
        cs=torch.as_tensor(data.chemical_shift,device=data.PD.device).reshape(-1)
        object.__setattr__(obj,'chemical_shift',(cs[chosen:chosen+1] if cs.numel()==data.PD.numel() else cs.clone()))
    art=copy.copy(engine.art);pulses={}
    for key,p in engine.art.pulses.items():
        q=copy.copy(p)
        if species==1 and getattr(p,'fat',None) is None:raise ValueError('Selected acquisition has no separate fat RF bank.')
        for label in ('water','fat'):
            bank=getattr(p,label,None)
            if bank is None:continue
            new=copy.copy(bank);vox=NS(**{k:getattr(bank.vox6,k)[chosen:chosen+1].contiguous() for k in ('pp','pz','mp','mz','zp','zz')});object.__setattr__(new,'vox6',vox);object.__setattr__(q,label,new)
        pulses[key]=q
    object.__setattr__(art,'pulses',pulses)
    # Stored sequence keys are recovered from exactly the same RF spans as prepass.
    from src.mrtini.optimized.prepass import build_inputs
    keys=getattr(engine,'workbench_rep_keys',None)
    if keys is None:keys,_,_=build_inputs(engine.seq,art)
    probe=prepare_gpu_execution(graph,engine.seq,obj,keys,art,slice_thickness=thickness,signal_workspace_mb=workspace_mb,kernel_mode=engine.kernel_mode,kernel_backend=engine.backend)
    signal=cpu(probe.run(publish=False,print_progress=False)).reshape(-1)
    if not np.isfinite(signal).all():raise ValueError('Non-finite voxel signal')
    # Preserve extrema and readout structure via a contiguous sample index selection.
    ix=np.linspace(0,len(signal)-1,min(len(signal),8192),dtype=int) if len(signal) else np.array([],int)
    times=[];t=0.
    for r in engine.seq:
        tt=np.cumsum(cpu(r.event_time),dtype=float);times.extend((tt[cpu(r.adc_usage)>0]+t).tolist());t+=float(tt[-1])
    if len(times)!=len(signal):raise ValueError('Probe ADC timing mismatch')
    return dict(tissue=name,field='3 T tissue constants',PD=pd,T1=t1,T2=t2,T2dash=t2d,D=d,B0=scalar(obj.B0),B1=float(obj.B1.sum(0).abs().item()),species=species,source_voxel=chosen,chemical_shift=scalar(obj.chemical_shift) if getattr(obj,'chemical_shift',None) is not None else None,total_samples=len(signal),index=ix.tolist(),time_ms=(np.asarray(times)[ix]*1000).tolist(),real=signal[ix].real.tolist(),imag=signal[ix].imag.tolist(),source='Same retained graph; pure tissue relaxation/PD; inherited RF/B0/B1 sample; position (0,0,0), one unit coil')
