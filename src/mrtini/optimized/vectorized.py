# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded batches of states/ADCs/voxels for the static-object signal path.

Same finite-voxel sinc, off-resonance, diffusion, relaxation and first-ancestor
physical-k convention as the supplied executor. No graph-edge approximation.
"""
import time
from collections import Counter
import numpy as np
import torch
from .fast_graph import FastRFContext,PackedGraph
from .signal_fast import SignalPlan


def execute_vectorized(graph,seq,data,rep_keys,art,*,slice_thickness=.005,
                       min_emitted_signal=1e-4,backend='auto',context=None,
                       state_batch=64,adc_batch=None,voxel_batch=None,
                       signal_workspace_mb=256,max_decay_groups=16,
                       performance=None,print_progress=False):
    if state_batch<1 or any(x is not None and x<1 for x in (adc_batch,voxel_batch)) or signal_workspace_mb<1 or max_decay_groups<0:raise ValueError('Invalid batch/workspace settings')
    wall_start=time.perf_counter()
    workspace_bytes=int(signal_workspace_mb*1024**2)
    device=data.PD.device;V=data.PD.numel();dtype=torch.complex64
    rf=context or FastRFContext(graph,art,data,backend)
    if rf.graph is not graph or rf.art is not art or rf.data is not data:raise ValueError('RF context belongs to different inputs')
    D=-1e-9*data.D;T1=data.T1.abs();T2=data.T2.abs();T2d=data.T2dash.abs()
    signal_plan=SignalPlan(data,D,T2d,max_decay_groups)
    if performance is not None:
        performance.update(voxels=V,signal_active_voxels=signal_plan.active_count,
            signal_decay_groups=None if signal_plan.groups is None else len(signal_plan.groups),
            signal_mode="generic" if signal_plan.groups is None else "factorized",signal_workspace_mb=signal_workspace_mb,
            emitted_states_per_rep=[])
    coil=data.coil_sens.T.to(dtype)*data.PD.abs()[:,None];C=coil.shape[1]
    scale=1/data.size if seq.normalized_grads else torch.ones_like(data.size)
    ez=torch.tensor([0.,0.,1.,0.],device=device)
    parent=torch.ones(1,V,device=device,dtype=dtype);previous_kt=torch.zeros(1,4,device=device)
    live=np.ones(1,dtype=bool);signal=[];host={};rf_seconds=0.;gpu=[];backend_counts=Counter()
    for i,(dists,rep) in enumerate(zip(graph[1:],seq)):
        if print_progress:print(f'\rCalculating repetition {i+1} / {len(seq)}',end='')
        key=rep_keys[i]
        if key not in host:
            t=art.pulses[key].water.nominal_taps
            host[key]=(float(t.dk),np.asarray(t.deltas.detach().cpu(),dtype=np.int64))
        dk,deltas=host[key]
        shim=torch.as_tensor(rep.pulse.shim_array,device=device)
        B1=data.B1.sum(0) if len(shim)==1 else (data.B1*(shim[:,0]*torch.exp(-1j*shim[:,1]))[:,None]).sum(0)
        phase=torch.as_tensor(rep.pulse.phase,device=device)+B1.angle()
        if performance is not None and parent.is_cuda:
            rf_begin,rf_end,free_end=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
            rf_begin.record(torch.cuda.current_stream(device))
        start=time.perf_counter();X,plan,_=rf.layer(i,key,phase,parent,live)
        rf_seconds+=time.perf_counter()-start
        backend_counts[getattr(plan,"last_backend",backend)]+=1
        if performance is not None and parent.is_cuda:rf_end.record(torch.cuda.current_stream(device))
        # First live edge per child, without GPU scalar reads or per-state kernels.
        if live.all():
            first=plan.rows[:-1].copy();first[np.diff(plan.rows)==0]=-1
        else:
            first=np.full(plan.children,-1,dtype=np.int64)
            es=np.flatnonzero(live[plan.src]);ds=plan.dst[es]
            keep=np.r_[True,np.diff(ds)!=0] if len(ds) else np.zeros(0,dtype=bool)
            first[ds[keep]]=es[keep]
        valid=first>=0;safe=np.maximum(first,0)
        src=plan.src[safe] if len(plan.src) else np.zeros(plan.children,dtype=np.int64)
        sign=np.where(plan.rel[safe]>=4,-1.,1.) if len(plan.src) else np.ones(plan.children)
        offset=deltas[plan.tap[safe]]*dk if len(plan.src) else np.zeros(plan.children)
        def tensor(a,dt=None):return torch.as_tensor(a,device=device,dtype=dt)
        kt=previous_kt.index_select(0,tensor(src,torch.long))*tensor(sign,previous_kt.dtype)[:,None]+tensor(offset,previous_kt.dtype)[:,None]*ez if len(previous_kt) else torch.zeros(plan.children,4,device=device)
        kind=np.asarray([0 if d.dist_type=='+' else (2 if d.dist_type=='z0' else 1) for d in dists])
        z0=tensor(kind==2,torch.bool);kt=torch.where(z0[:,None],torch.zeros_like(kt),kt)
        live=valid|(kind==2);X=X*tensor(live,X.real.dtype)[:,None]
        trajectory=torch.cumsum(torch.cat([rep.gradm*scale[None,:],rep.event_time[:,None]],1),0)
        adc_index=(rep.adc_usage>0).nonzero(as_tuple=True)[0];adc=trajectory.index_select(0,adc_index)
        A=len(adc_index);rep_sig=torch.zeros(A,C,device=device,dtype=dtype)
        total=rep.event_time.sum();r1=torch.exp(-total/T1);r2=torch.exp(-total/T2)
        k2z=kt[:,:3].square().sum(-1)
        end_mag=X*r1*torch.exp(D[None,:]*total*k2z[:,None])
        recovery=z0[:,None]*(1-r1)
        if isinstance(graph,PackedGraph):
            n0,n1=map(int,graph.offsets[i+1:i+3])
            bits=tensor(graph.node_mask[n0:n1].astype(np.int64),torch.long)
            recovery=recovery*((bits[:,None]>>rf.ids[None,:])&1)
        end_mag=end_mag+recovery
        p_indices=np.flatnonzero((kind==0)&live)
        emissions=[]
        for lo in range(0,len(p_indices),state_batch):
            ix=p_indices[lo:lo+state_batch];ind=tensor(ix,torch.long);k0=kt.index_select(0,ind);M=X.index_select(0,ind)
            traj=k0[:,None,:]+trajectory[None,:,:]
            k2=traj[:,:,:3];k1=torch.cat([k0[:,None,:3],k2[:,:-1,:]],1)
            b=((1./3.)*(2*torch.pi)**2)*rep.event_time[None,:]*(k1*k1+k1*k2+k2*k2).sum(-1)
            cb=b.cumsum(1)
            end_mag.index_copy_(0,ind,M*r2*torch.exp(D[None,:]*cb[:,-1:]))
            emit=np.asarray([dists[j].emitted_signal>=min_emitted_signal for j in ix])
            if not A or not emit.any():continue
            emit_indices=np.flatnonzero(emit);mi=tensor(emit_indices,torch.long)
            adc_traj=traj.index_select(0,mi).index_select(1,adc_index)
            emit_k0=k0.index_select(0,mi)
            cb_adc=cb.index_select(0,mi).index_select(1,adc_index)
            deph=data.dephasing_func(adc_traj[:,:,:2].reshape(-1,2),data.nyquist[:2]).reshape(len(emit_indices),A)
            if not data.multi_slice:deph=deph*torch.sinc(adc_traj[:,:,2]*slice_thickness)
            allowed=None
            if isinstance(graph,PackedGraph):
                absolute=int(graph.offsets[i+1])+ix[emit_indices]
                allowed=tensor((graph.emitted[absolute]>=min_emitted_signal).copy(),torch.bool)
            emissions.append((ind.index_select(0,mi),emit_k0,adc_traj[:,:,3],cb_adc,deph,allowed))
        if emissions:
            emit_rows,k0,t,cb,deph=(torch.cat([e[j] for e in emissions],dim=0) if len(emissions)>1 else emissions[0][j] for j in range(5))
            allowed=torch.cat([e[5] for e in emissions],dim=0) if isinstance(graph,PackedGraph) else None
            rep_sig=signal_plan.contract(X.index_select(0,emit_rows),k0,adc,t,cb,deph,allowed,rf.ids,data,D,T2d,coil,
                state_batch=state_batch,adc_batch=adc_batch,voxel_batch=voxel_batch,workspace_bytes=workspace_bytes)
        if performance is not None:performance['emitted_states_per_rep'].append(sum(e[0].shape[0] for e in emissions))
        end_kt=torch.where(tensor(kind==0,torch.bool)[:,None],kt+trajectory[-1],kt)
        rep_sig*=torch.exp(1j*rep.adc_phase.index_select(0,adc_index))[:,None]
        signal.append(rep_sig)
        # Keep the public graph inspection convention; views are cheap and no
        # physics/tensor arithmetic is dispatched per edge or state here.
        for p in graph[i]:p.mag=None
        for j,d in enumerate(dists):
            d.kt_vec=end_kt[j];d.mag=end_mag[j] if live[j] else None
        parent=end_mag;previous_kt=end_kt
        if performance is not None and parent.is_cuda:
            free_end.record(torch.cuda.current_stream(device))
            gpu.append((rf_begin,rf_end,free_end))
    if print_progress:print(' - done')
    measured=torch.cat(signal)
    if performance is not None:
        performance['rf_cpu_dispatch_seconds']=rf_seconds;performance['vectorized_free_evolution']=True
        performance['rf_backend']=backend;performance['rf_banks_prepared']=len(rf.banks)
        performance['rf_backend_counts']=dict(backend_counts)
        performance['rf_bank_table_bytes']=sum(b.table.numel()*b.table.element_size() for b in rf.banks.values())
        if parent.is_cuda:
            torch.cuda.synchronize(device)
            performance['rf_cuda_seconds']=sum(start.elapsed_time(end) for start,end,_ in gpu)/1000
            performance['free_evolution_and_signal_cuda_seconds']=sum(end.elapsed_time(finish) for _,end,finish in gpu)/1000
        performance['execution_wall_seconds']=time.perf_counter()-wall_start
    return measured
