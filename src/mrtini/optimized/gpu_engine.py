# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepared static-object inference executor. Fixed graph, maps, RF and sequence.

CUDA Graph capture covers numerical execution, not preparation or Python graph
inspection. A prepared instance must be rebuilt after any input change. Not
thread safe. The torch backend is an independently executable schedule oracle.
"""
from dataclasses import dataclass
import time
from contextlib import nullcontext
import numpy as np
import torch
from .fast_graph import FastRFContext, PackedGraph, RFPlan, RFBank, rf_reference

@dataclass
class Step:
    plan: object; bank: object; phase: torch.Tensor; ep: torch.Tensor
    kind: torch.Tensor; live: torch.Tensor; live_host: np.ndarray
    bits: torch.Tensor; end_kt: torch.Tensor; b: torch.Tensor
    r1: torch.Tensor; r2: torch.Tensor; rows: torch.Tensor; k0: torch.Tensor
    t: torch.Tensor; cb: torch.Tensor; deph: torch.Tensor; allowed: object
    adc: torch.Tensor; rot: torch.Tensor; offset: int

class PreparedGPUExecution:
    """No-grad, static sequence replay. Output and final graph states are snapshots.

    For changing tissue maps, pulse amplitudes/phases, graph thresholds, etc.,
    create a fresh instance. Captured kernels are not a differentiable operator.
    """
    @torch.no_grad()
    def __init__(self, graph, seq, data, rep_keys, art, *, slice_thickness=.005,
                 min_emitted_signal=1e-4, signal_workspace_mb=256, adc_batch=None,
                 voxel_batch=None, kernel_backend='triton', context=None, kernel_mode='dynamic'):
        start=time.perf_counter()
        if kernel_backend not in ('triton','torch'):raise ValueError('Unknown GPU engine backend')
        if kernel_mode not in ('dynamic','specialized'):raise ValueError('kernel_mode must be dynamic or specialized')
        self.kernel_mode=kernel_mode
        if data.voxel_motion is not None or data.phantom_motion is not None:raise ValueError('Prepared engine is static-object only')
        if len(seq)==0 or len(graph)!=len(seq)+1 or len(rep_keys)!=len(seq):raise ValueError('Graph/sequence length mismatch or empty sequence')
        if len(graph[0])!=1:raise ValueError('Prepared engine expects a single initial equilibrium state')
        if signal_workspace_mb<1 or any(x is not None and x<1 for x in (adc_batch,voxel_batch)):raise ValueError('Invalid workspace settings')
        self.graph,self.seq,self.data,self.art=graph,seq,data,art
        self.device=data.PD.device; self.V=data.PD.numel(); self.backend=kernel_backend
        if self.V<1:raise ValueError('Empty phantom')
        if kernel_backend=='triton':
            if not data.PD.is_cuda:raise ValueError('Triton engine requires NVIDIA CUDA')
            if kernel_mode=='dynamic':
                from . import triton_evolution_dynamic as kernels
                from .triton_rf_dynamic import rf_triton
            else:
                from . import triton_evolution as kernels
                from .triton_rf import rf_triton
            self.kernels,self.rf_triton=kernels,rf_triton
        self.rf=context or FastRFContext(graph,art,data,'triton' if kernel_backend=='triton' else 'reference')
        if self.rf.graph is not graph or self.rf.art is not art or self.rf.data is not data:raise ValueError('RF context belongs to different inputs')
        def tensor(x,dtype=None):return torch.as_tensor(x,device=self.device,dtype=dtype).contiguous()
        def vec(x):return x.reshape(-1).contiguous()
        self.D=vec(-1e-9*data.D);self.T2=vec(data.T2.abs());self.T2d=vec(data.T2dash.abs());self.B0=vec(data.B0);self.pos=data.voxel_pos.contiguous()
        self.ids=self.rf.ids.contiguous();T1=vec(data.T1.abs())
        self.coil=(data.coil_sens.T.to(torch.complex64)*data.PD.abs()[:,None]).contiguous();self.C=self.coil.shape[1]
        if self.C<1:raise ValueError('Expected at least one receiver coil')
        for name,x in [('PD',data.PD),('T1',T1),('T2',self.T2),('T2dash',self.T2d),('D',self.D),('B0',self.B0),('voxel_pos',self.pos)]:
            if x.dtype!=torch.float32 or x.device!=self.device:raise ValueError(f'Prepared engine expects FP32 {name} on {self.device}')
        if self.pos.shape!=(self.V,3) or any(x.numel()!=self.V for x in (T1,self.T2,self.T2d,self.D,self.B0)):raise ValueError('Unexpected phantom map shape')
        scale=1/data.size if seq.normalized_grads else torch.ones_like(data.size)
        ez=tensor([0.,0.,1.,0.],torch.float32);previous_kt=torch.zeros(1,4,device=self.device)
        live=np.ones(1,dtype=bool);self.steps=[];offset=0;host={}
        for i,(dists,rep) in enumerate(zip(graph[1:],seq)):
            key=rep_keys[i]
            if key not in host:
                taps=art.pulses[key].water.nominal_taps
                host[key]=(float(taps.dk),np.asarray(taps.deltas.detach().cpu(),dtype=np.int64))
            dk,deltas=host[key]
            if i not in self.rf.plans:
                if isinstance(graph,PackedGraph):
                    p=graph.rf_plan(i);self.rf.plans[i]=RFPlan(p.rows,p.src,p.rel,p.tap,p.mask,p.parents)
                else:self.rf.plans[i]=RFPlan.from_graph(graph[i],dists)
            if key not in self.rf.banks:self.rf.banks[key]=RFBank(art.pulses[key],self.rf.physical,self.rf.ids)
            plan=self.rf.plans[i];bank=self.rf.banks[key]
            if bank.table.dtype!=torch.complex64:raise ValueError('Prepared engine requires complex64 RF coefficients')
            if not plan.children:raise ValueError('Prepared engine requires nonempty graph layers')
            plan.tensors(bank) # all H2D uploads occur before graph capture
            shim=tensor(rep.pulse.shim_array)
            B1=data.B1.sum(0) if len(shim)==1 else (data.B1*(shim[:,0]*torch.exp(-1j*shim[:,1]))[:,None]).sum(0)
            phase=(tensor(rep.pulse.phase)+B1.angle()).expand(self.V).contiguous()
            if phase.dtype!=torch.float32:raise ValueError('Prepared engine requires FP32 RF phase')
            ep=torch.polar(torch.ones_like(phase),phase).contiguous()
            first=np.full(plan.children,-1,dtype=np.int64)
            es=np.flatnonzero(live[plan.src]);ds=plan.dst[es]
            keep=np.r_[True,np.diff(ds)!=0] if len(ds) else np.zeros(0,dtype=bool)
            first[ds[keep]]=es[keep];valid=first>=0;safe=np.maximum(first,0)
            src=plan.src[safe] if len(plan.src) else np.zeros(plan.children,dtype=np.int64)
            sign=np.where(plan.rel[safe]>=4,-1.,1.) if len(plan.src) else np.ones(plan.children)
            off=deltas[plan.tap[safe]]*dk if len(plan.src) else np.zeros(plan.children)
            kt=previous_kt.index_select(0,tensor(src,torch.long))*tensor(sign,torch.float32)[:,None]+tensor(off,torch.float32)[:,None]*ez
            kind=np.asarray([0 if d.dist_type=='+' else (2 if d.dist_type=='z0' else 1) for d in dists]);k=tensor(kind,torch.int32)
            kt=torch.where((k==2)[:,None],torch.zeros_like(kt),kt);live=valid|(kind==2)
            trajectory=torch.cumsum(torch.cat([rep.gradm*scale[None,:],rep.event_time[:,None]],1),0)
            if trajectory.dtype!=torch.float32 or trajectory.shape[0]==0:raise ValueError('Prepared engine requires nonempty FP32 event trajectory')
            ai=(rep.adc_usage>0).nonzero(as_tuple=True)[0];adc=trajectory.index_select(0,ai).contiguous();A=len(ai)
            total=rep.event_time.sum();r1=torch.exp(-total/T1).contiguous();r2=torch.exp(-total/self.T2).contiguous()
            b=(total*kt[:,:3].square().sum(-1)).contiguous()
            pp=np.flatnonzero((kind==0)&live);rows=[];ks=[];ts=[];cbs=[];dsig=[];masks=[]
            # Limit temporary trajectory workspace even for very long repetitions.
            for lo in range(0,len(pp),64):
                ix=pp[lo:lo+64];ind=tensor(ix,torch.long);k0=kt.index_select(0,ind)
                traj=k0[:,None,:]+trajectory[None,:,:];k2=traj[:,:,:3];k1=torch.cat([k0[:,None,:3],k2[:,:-1,:]],1)
                cb=(((1./3.)*(2*torch.pi)**2)*rep.event_time[None,:]*(k1*k1+k1*k2+k2*k2).sum(-1)).cumsum(1)
                b.index_copy_(0,ind,cb[:,-1]);emit=np.flatnonzero([dists[j].emitted_signal>=min_emitted_signal for j in ix])
                if not A or not len(emit):continue
                mi=tensor(emit,torch.long);at=traj.index_select(0,mi).index_select(1,ai)
                deph=data.dephasing_func(at[:,:,:2].reshape(-1,2),data.nyquist[:2]).reshape(len(emit),A)
                if not data.multi_slice:deph=deph*torch.sinc(at[:,:,2]*slice_thickness)
                if deph.dtype!=torch.float32:raise ValueError('Prepared engine requires real FP32 dephasing weights')
                rows.append(ind.index_select(0,mi));ks.append(k0.index_select(0,mi));ts.append(at[:,:,3]);cbs.append(cb.index_select(0,mi).index_select(1,ai));dsig.append(deph)
                if isinstance(graph,PackedGraph):masks.append(tensor((graph.emitted[int(graph.offsets[i+1])+ix[emit]]>=min_emitted_signal).copy(),torch.bool))
            def cat(xs,shape,dtype=torch.float32):return torch.cat(xs).contiguous() if xs else torch.empty(shape,device=self.device,dtype=dtype)
            bits=tensor(graph.node_mask[int(graph.offsets[i+1]):int(graph.offsets[i+2])].astype(np.int64),torch.long) if isinstance(graph,PackedGraph) else torch.full((plan.children,),-1,device=self.device,dtype=torch.long)
            end_kt=torch.where((k==0)[:,None],kt+trajectory[-1],kt).contiguous()
            self.steps.append(Step(plan,bank,phase,ep,k,tensor(live.astype(np.int32)),live.copy(),bits,end_kt,b,r1,r2,cat(rows,(0,),torch.long),cat(ks,(0,4)),cat(ts,(0,A)),cat(cbs,(0,A)),cat(dsig,(0,A)),cat(masks,(0,graph.emitted.shape[1]),torch.bool) if isinstance(graph,PackedGraph) else None,adc,torch.exp(1j*rep.adc_phase.index_select(0,ai))[:,None].contiguous(),offset))
            previous_kt=end_kt;offset+=A
        max_n=max(s.plan.children for s in self.steps);max_s=max(s.rows.numel() for s in self.steps);max_a=max(s.adc.shape[0] for s in self.steps)
        def empty(*shape):return torch.empty(shape,device=self.device,dtype=torch.complex64)
        self.parent=empty(max(1,max_n),self.V);self.rf_out=empty(max_n,self.V);self.q=empty(max_s,self.V)
        # Use flat buffers so even tail tiles are contiguous for cuBLAS.
        self.ab=max(1,min(max_a,adc_batch or 128));self.vb=max(1,min(self.V,voxel_batch or self.V,int(signal_workspace_mb*1024**2)//(8*self.ab)))
        self.tile=empty(self.ab*self.vb);self.partial=empty(self.ab*self.C);self.output=empty(offset,self.C)
        # Metadata is retained for the entire acquisition; report its storage
        # separately from workspaces and RF banks (not a total VRAM estimate).
        metadata=[self.D,self.T2,self.T2d,self.B0,self.pos,self.ids,self.coil]
        for step in self.steps:
            metadata.extend(v for v in vars(step).values() if isinstance(v,torch.Tensor))
            metadata.extend(step.plan.tensors(step.bank).values())
        storages={ (str(x.device),x.untyped_storage().data_ptr()):x.untyped_storage().nbytes() for x in metadata if x.numel() }
        self.prepared_metadata_bytes=sum(storages.values())
        self.capture_graph=None;self.capture_seconds=0.;self.prepare_seconds=time.perf_counter()-start
        self.last_n=max_n;self.calls=0

    def _run(self, *, print_progress=False):
        # No printing or synchronization on the normal/captured path.
        if print_progress:
            progress_start=last_update=time.perf_counter()
            total=len(self.steps)
            print(f"[MRtini] Calculating repetition 0 / {total} (first run may compile kernels)...", flush=True)
        self.parent[:1].fill_(1);self.output.zero_();n=1
        for i,s in enumerate(self.steps,1):
            m=s.plan.children;X=self.rf_out[:m];P=self.parent[:m];Q=self.q[:s.rows.numel()];A=s.adc.shape[0]
            if self.backend=='triton':self.rf_triton(s.plan,s.bank,self.parent[:n],s.phase,out=X,phase_complex=s.ep)
            else:X.copy_(rf_reference(s.plan,s.bank,self.parent[:n],s.phase))
            sig=self.output[s.offset:s.offset+A]
            if Q.shape[0] and A:
                if self.backend=='triton':self.kernels.state_phase(X,Q,s.rows,s.k0,s.allowed,self.ids,self.pos,self.B0)
                else:
                    q=X.index_select(0,s.rows)
                    if s.allowed is not None:q=q*s.allowed[:,self.ids]
                    Q.copy_(1.41421356237*q*torch.exp(2j*torch.pi*(s.k0[:,3:]*self.B0+s.k0[:,:3]@self.pos.T)))
                for al in range(0,A,self.ab):
                    ah=min(A,al+self.ab);na=ah-al
                    for vl in range(0,self.V,self.vb):
                        vh=min(self.V,vl+self.vb);nv=vh-vl;tile=self.tile[:na*nv].view(na,nv)
                        if self.backend=='triton':self.kernels.adc_tile(Q,s.t,s.cb,s.deph,self.D,self.T2d,self.T2,self.B0,self.pos,s.adc,tile,al,ah,vl,vh)
                        else:
                            damp=torch.exp(-s.t[:,al:ah,None].abs()/self.T2d[vl:vh]+s.cb[:,al:ah,None]*self.D[vl:vh]);a=s.adc[al:ah]
                            event=torch.exp(2j*torch.pi*(a[:,3:]*self.B0[vl:vh]+a[:,:3]@self.pos[vl:vh].T))*torch.exp(-a[:,3:]/self.T2[vl:vh])
                            tile.copy_((Q[:,None,vl:vh]*damp*s.deph[:,al:ah,None]).sum(0)*event)
                        partial=self.partial[:na*self.C].view(na,self.C)
                        torch.mm(tile,self.coil[vl:vh],out=partial);sig[al:ah].add_(partial)
            sig.mul_(s.rot)
            # RF output is consumed before parent storage is overwritten.
            if self.backend=='triton':self.kernels.evolve(X,P,self.D,s.r1,s.r2,s.b,s.kind,s.live,s.bits,self.ids)
            else:
                rec=(s.kind==2)[:,None]*(1-s.r1)*((s.bits[:,None]>>self.ids)&1)
                # Match the reference grouping of the two multiplications.
                P.copy_(X*s.live[:,None]*torch.where((s.kind==0)[:,None],s.r2,s.r1)*torch.exp(self.D*s.b[:,None])+rec)
            n=m
            if print_progress and (i==total or time.perf_counter()-last_update>=0.25):
                # Synchronize only for displayed updates, so the count reflects
                # completed GPU work, not merely queued launches.
                if self.device.type=='cuda':torch.cuda.current_stream(self.device).synchronize()
                last_update=time.perf_counter();elapsed=last_update-progress_start
                print(f"[MRtini] Calculating repetition {i} / {total} ({100*i/total:.1f}%) | {elapsed:.3f}s elapsed", flush=True)
        self.last_n=n

    @torch.no_grad()
    def capture(self):
        """Warm up on a side stream, then capture once. No silent fallback."""
        if self.backend!='triton':raise ValueError('CUDA Graph requires the Triton CUDA engine')
        if self.capture_graph is not None:return self
        start=time.perf_counter()
        with torch.cuda.device(self.device):
            current=torch.cuda.current_stream(self.device);side=torch.cuda.Stream(device=self.device);side.wait_stream(current)
            with torch.cuda.stream(side):self._run();self._run()
            current.wait_stream(side);current.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=side):self._run()
            self.capture_graph=g;torch.cuda.synchronize(self.device)
        self.capture_seconds=time.perf_counter()-start
        return self

    @torch.no_grad()
    def run(self, *, cuda_graph=False, publish=True, performance=None, print_progress=False):
        """Return independent signal snapshot; publish final magnetization by default."""
        if cuda_graph:
            if print_progress and self.capture_graph is None:
                print("[MRtini] Warming up and capturing CUDA Graph (first run may compile kernels)...", flush=True)
            self.capture()
            if print_progress:print(f"[MRtini] Replaying CUDA Graph ({len(self.steps)} repetitions)...", flush=True)
        start=time.perf_counter()
        if self.device.type=='cuda' and performance is not None:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record(torch.cuda.current_stream(self.device))
        with torch.cuda.device(self.device) if self.device.type=='cuda' else nullcontext():
            if cuda_graph:self.capture_graph.replay()
            else:self._run(print_progress=True) if print_progress else self._run()
        if self.device.type=='cuda' and performance is not None:b.record(torch.cuda.current_stream(self.device))
        out=self.output.clone();self.calls+=1
        if publish:
            for layer in self.graph[:-1]:
                for d in layer:d.mag=None
            last=self.steps[-1];state=self.parent[:self.last_n].clone();kt=last.end_kt.clone()
            for j,d in enumerate(self.graph[-1]):d.mag=state[j] if last.live_host[j] else None;d.kt_vec=kt[j]
            # Preserve physical k-vectors of earlier layers, as the ordinary executor does.
            for layer,step in zip(self.graph[1:-1],self.steps[:-1]):
                for j,d in enumerate(layer):d.kt_vec=step.end_kt[j]
        if performance is not None:
            performance.update(gpu_engine=self.backend,kernel_mode=self.kernel_mode,signal_mode='fused_adc' if self.backend=='triton' else 'prepared_torch_oracle',cuda_graph=bool(cuda_graph),prepared_calls=self.calls,prepared_metadata_bytes=self.prepared_metadata_bytes,prepare_seconds=self.prepare_seconds,capture_seconds=self.capture_seconds,voxels=self.V,emitted_states_per_rep=[s.rows.numel() for s in self.steps],persistent_buffer_bytes=sum(x.numel()*x.element_size() for x in (self.parent,self.rf_out,self.q,self.tile,self.partial,self.output)),rf_bank_table_bytes=sum(b.table.numel()*b.table.element_size() for b in self.rf.banks.values()))
            if self.device.type=='cuda':b.synchronize();performance['gpu_execution_seconds']=a.elapsed_time(b)/1000
            performance['execution_wall_seconds']=time.perf_counter()-start
        if print_progress:
            if self.device.type=='cuda':torch.cuda.current_stream(self.device).synchronize()
            print(f"[MRtini] Done: {len(self.steps)} / {len(self.steps)} repetitions | execution {time.perf_counter()-start:.3f}s", flush=True)
        return out

def prepare_gpu_execution(graph,seq,data,rep_keys,art,**kwargs):
    return PreparedGPUExecution(graph,seq,data,rep_keys,art,**kwargs)
