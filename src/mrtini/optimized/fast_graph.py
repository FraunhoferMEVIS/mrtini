# SPDX-License-Identifier: AGPL-3.0-or-later
"""Packed phase graphs and RF application without Python tensor operations per edge.

The rest of the Bloch/PDG signal physics is intentionally left in the executor.
Uniform RF coefficients use dense GEMM; general coefficients use grouped torch
operations or the optional fused Triton CSR kernel on CUDA.
"""
from dataclasses import dataclass
import numpy as np
import torch
from collections import OrderedDict

REL=('++','+z','z+','zz','-+','-z')
COEF=('pp','pz','zp','zz','mp','mz')
PHASE=(0,-1,1,0,2,1)

class PackedDist:
    __slots__=("graph","index","layer","local","dist_type","mag","kt_vec","emitted_signal","latent_signal","prepass_kt_vec")
    def __init__(self,graph,index,layer,local):
        self.graph,self.index,self.layer,self.local=graph,index,layer,local
        self.dist_type=('+','z','z0')[int(graph.kind[index])]
        self.mag=None;self.kt_vec=None
        self.emitted_signal=float(graph._emitted_max[index])
        self.latent_signal=float(graph._latent_max[index])
        self.prepass_kt_vec=graph.kt_end[index]
    @property
    def ancestors(self):
        if self.layer==0:return []
        plan=self.graph.rf_plan(self.layer-1)
        return [(REL[int(plan.rel[e])],self.graph[self.layer-1][int(plan.src[e])],0,int(plan.tap[e]))
                for e in range(int(plan.rows[self.local]),int(plan.rows[self.local+1]))]
    @property
    def ancestor_species_masks(self):
        if self.layer==0:return []
        p=self.graph.rf_plan(self.layer-1);return p.mask[int(p.rows[self.local]):int(p.rows[self.local+1])].tolist()

class PackedGraph:
    def __init__(self,payload,case_species=None):
        self.payload=payload;N=payload['nodes'];S=payload['cases']
        self.case_species=list(range(S)) if case_species is None else list(case_species)
        def a(name,dtype):return np.frombuffer(payload[name],dtype)
        self.offsets=a('layers','<u8');self.kind=a('kind','u1');self.node_mask=a('node_mask','u1')
        self.kt_start=a('kt_start','<f8').reshape(N,4);self.kt_end=a('kt_end','<f8').reshape(N,4)
        self.emitted=a('emitted','<f4').reshape(N,S);self.latent=a('latent','<f4').reshape(N,S)
        self.rows=a('row_ptr','<u8');self.src=a('parent','<u4');self.rel=a('relation','u1');self.tap=a('tap','<i4');self.mask=a('edge_mask','u1')
        self._emitted_max=self.emitted.max(axis=1);self._latent_max=self.latent.max(axis=1)
        self._layers=[[PackedDist(self,j,i,j-int(self.offsets[i])) for j in range(int(self.offsets[i]),int(self.offsets[i+1]))] for i in range(len(self.offsets)-1)]
        self._plans={}
    def __len__(self):return len(self._layers)
    def __iter__(self):return iter(self._layers)
    def __getitem__(self,i):return self._layers[i]
    def rf_plan(self,i):
        if i not in self._plans:
            a,b=map(int,self.offsets[i+1:i+3]);lo=int(self.rows[a]);hi=int(self.rows[b])
            self._plans[i]=RFPlan(self.rows[a:b+1].astype(np.int64)-lo,self.src[lo:hi],self.rel[lo:hi],self.tap[lo:hi],self.mask[lo:hi],len(self[i]))
        return self._plans[i]

class RFPlan:
    def __init__(self,rows,src,rel,tap,mask,parents):
        self.rows=np.asarray(rows,dtype=np.int64);self.src=np.asarray(src,dtype=np.int64)
        self.rel=np.asarray(rel,dtype=np.int64);self.tap=np.asarray(tap,dtype=np.int64);self.mask=np.asarray(mask,dtype=np.int64)
        self.children=len(self.rows)-1;self.parents=parents
        self.dst=np.repeat(np.arange(self.children,dtype=np.int64),np.diff(self.rows))
        self.cache={};self.dense_cache={};self.group_cache={}
    @classmethod
    def from_graph(cls,previous,current):
        ids={id(p):i for i,p in enumerate(previous)};rows=[0];src=[];rel=[];tap=[];mask=[]
        for d in current:
            m=getattr(d,'ancestor_species_masks',None)
            for i,(r,p,_,j) in enumerate(d.ancestors):
                src.append(ids[id(p)]);rel.append(REL.index(r));tap.append(int(j));mask.append(3 if m is None else int(m[i]))
            rows.append(len(src))
        return cls(rows,src,rel,tap,mask,len(previous))
    def tensors(self,bank):
        key=(str(bank.table.device),id(bank))
        if key not in self.cache:
            if self.tap.size and (self.tap.min()<0 or self.tap.max()>=len(bank.tap_lookup)):raise ValueError('Graph tap outside pulse grid')
            mapped=bank.tap_lookup[self.tap]
            if np.any(mapped<0):raise ValueError('Graph tap missing from both species supports')
            slots=self.rel*bank.K+mapped
            self.cache[key]={n:torch.tensor(v,device=bank.table.device,dtype=torch.int64) for n,v in
                [('rows',self.rows),('src',self.src),('dst',self.dst),('rel',self.rel),('mask',self.mask),('slot',slots)]}
        return self.cache[key]
    def first_ancestor(self,i,previous,live):
        for e in range(int(self.rows[i]),int(self.rows[i+1])):
            p=int(self.src[e])
            if live[p]:return (REL[int(self.rel[e])],previous[p],0,int(self.tap[e]))
        return None

class RFBank:
    def __init__(self,pulse,physical_species,case_ids):
        w=pulse.water;f=pulse.fat if pulse.fat is not None else w
        self.tap_map={t:i for i,t in enumerate(sorted(set(w.supp_map)|set(f.supp_map)))}
        self.K=len(self.tap_map)
        self.tap_lookup=np.full(max(self.tap_map,default=-1)+1,-1,dtype=np.int64)
        for tap,slot in self.tap_map.items():self.tap_lookup[tap]=slot
        self.ids=case_ids;self.V=case_ids.numel()
        device=w.vox6.pp.device;self.physical_species=physical_species.to(device=device).reshape(-1)
        def full(p):
            a=torch.stack([getattr(p.vox6,n).T for n in COEF]).contiguous()
            if a.shape[-1]!=self.V:raise ValueError('RF bank row count does not match phantom')
            zero=torch.zeros(6,1,self.V,device=device,dtype=a.dtype);a=torch.cat([a,zero],dim=1)
            index=torch.tensor([p.supp_map.get(t,a.shape[1]-1) for t in self.tap_map],device=device,dtype=torch.long)
            return a.index_select(1,index)
        wa=full(w)
        if pulse.fat is None:self.table=wa
        else:self.table=torch.where(self.physical_species[None,None,:]==1,full(f),wa).contiguous()
        self.table=self.table.reshape(6*self.K,self.V).contiguous()
        self.case_indices=[];self.first=[];templates=[];uniform=True
        S=int(case_ids.max().item())+1 if self.V else 0
        for s in range(S):
            ix=(case_ids==s).nonzero(as_tuple=True)[0];self.case_indices.append(ix)
            if not ix.numel():templates.append(torch.zeros(6*self.K,device=device,dtype=self.table.dtype));self.first.append(0);continue
            first=int(ix[0]);self.first.append(first);template=self.table[:,first]
            # Exact check; no approximation of B0/B1-dependent coefficients.
            for chunk in ix.split(1024):
                if not bool((self.table.index_select(1,chunk)==template[:,None]).all()):uniform=False;break
            templates.append(template)
        self.uniform=uniform;self.templates=torch.stack(templates) if templates else None

def _phase_factor(phase,r):
    if r==0 or r==3:return 1
    return torch.exp((2j if r==4 else (-1j if r==1 else 1j))*phase)

def rf_reference(plan,bank,parent,phase):
    out=torch.zeros(plan.children,bank.V,device=parent.device,dtype=parent.dtype)
    for d in range(plan.children):
        for e in range(int(plan.rows[d]),int(plan.rows[d+1])):
            r=int(plan.rel[e]);coef=bank.table[r*bank.K+bank.tap_map[int(plan.tap[e])]]
            active=((int(plan.mask[e])>>bank.ids)&1).to(torch.bool)
            m=parent[int(plan.src[e])];m=m.conj() if r>=4 else m
            out[d].add_(m*coef*_phase_factor(phase,r)*active)
    return out

def rf_dense(plan,bank,parent,phase):
    if not bank.uniform:raise ValueError('Dense path requires exactly uniform coefficients within each case')
    t=plan.tensors(bank);out=torch.zeros(plan.children,bank.V,device=parent.device,dtype=parent.dtype)
    for s,ix in enumerate(bank.case_indices):
        if not ix.numel():continue
        key=(id(bank),s,str(parent.device));matrices=plan.dense_cache.get(key)
        if matrices is None:
            # Five constant matrices, factored by relation/conjugation. RF
            # phase is applied below, never baked incorrectly across voxels.
            matrices=[]
            for relations in [(0,3),(1,),(2,),(4,),(5,)]:
                choose=np.isin(plan.rel,relations)&((plan.mask&(1<<s))!=0)
                ei=torch.tensor(np.flatnonzero(choose),device=parent.device,dtype=torch.long)
                if not ei.numel():matrices.append(None);continue
                A=torch.zeros(plan.children,plan.parents,device=parent.device,dtype=parent.dtype)
                A.index_put_((t['dst'][ei],t['src'][ei]),bank.templates[s,t['slot'][ei]],accumulate=True)
                matrices.append(A)
            plan.dense_cache[key]=matrices
        X=parent if ix.numel()==bank.V else parent.index_select(1,ix)
        p=phase if phase.ndim==0 else (phase if ix.numel()==bank.V else phase.index_select(0,ix))
        # A relation has one shared voxel phase factor, even when phase varies.
        Y=matrices[0]@X if matrices[0] is not None else torch.zeros(plan.children,ix.numel(),device=parent.device,dtype=parent.dtype)
        for A,r in zip(matrices[1:],(1,2,4,5)):
            if A is not None:Y=Y+(A@(X.conj() if r>=4 else X))*_phase_factor(p,r)
        if ix.numel()==bank.V:out=Y
        else:out.index_copy_(1,ix,Y)
    return out

def rf_grouped(plan,bank,parent,phase,chunk_voxels=1024,chunk_edges=2048):
    t=plan.tensors(bank);key=(id(bank),str(parent.device));groups=plan.group_cache.get(key)
    if groups is None:
        codes=(plan.rel*bank.K+bank.tap_lookup[plan.tap])*4+plan.mask
        order=np.argsort(codes,kind='stable');sorted_codes=codes[order]
        starts=np.r_[0,np.flatnonzero(np.diff(sorted_codes))+1,len(order)] if len(order) else np.array([0])
        groups=[]
        for lo,hi in zip(starts[:-1],starts[1:]):
            code=int(sorted_codes[lo]);r=(code//4)//bank.K;j=(code//4)%bank.K;m=code%4
            groups.append((r,j,m,torch.tensor(order[lo:hi].copy(),device=parent.device,dtype=torch.long)))
        plan.group_cache[key]=groups
    out=torch.zeros(plan.children,bank.V,device=parent.device,dtype=parent.dtype)
    for r,j,m,es in groups:
        for edges in es.split(chunk_edges):
            src=t['src'][edges];dst=t['dst'][edges]
            for lo in range(0,bank.V,chunk_voxels):
                hi=min(lo+chunk_voxels,bank.V);p=phase if phase.ndim==0 else phase[lo:hi]
                coef=bank.table[r*bank.K+j,lo:hi]*_phase_factor(p,r)*((m>>bank.ids[lo:hi])&1)
                X=parent[:,lo:hi].index_select(0,src);X=X.conj() if r>=4 else X
                out[:,lo:hi].index_add_(0,dst,X*coef)
    return out

def rf_sparse(plan,bank,parent,phase):
    if not bank.uniform:raise ValueError('Sparse matrix path requires exactly uniform coefficients within each case')
    t=plan.tensors(bank);out=torch.zeros(plan.children,bank.V,device=parent.device,dtype=parent.dtype)
    for s,ix in enumerate(bank.case_indices):
        if not ix.numel():continue
        key=('sparse',id(bank),s,str(parent.device));matrices=plan.dense_cache.get(key)
        if matrices is None:
            matrices=[]
            for relations in [(0,3),(1,),(2,),(4,),(5,)]:
                choose=np.isin(plan.rel,relations)&((plan.mask&(1<<s))!=0)
                ei=torch.tensor(np.flatnonzero(choose),device=parent.device,dtype=torch.long)
                if not ei.numel():matrices.append(None);continue
                A=torch.sparse_coo_tensor(torch.stack([t['dst'][ei],t['src'][ei]]),bank.templates[s,t['slot'][ei]],
                    (plan.children,plan.parents),device=parent.device,check_invariants=True).coalesce().to_sparse_csr()
                matrices.append(A)
            plan.dense_cache[key]=matrices
        X=parent if ix.numel()==bank.V else parent.index_select(1,ix)
        p=phase if phase.ndim==0 else (phase if ix.numel()==bank.V else phase.index_select(0,ix))
        Y=torch.sparse.mm(matrices[0],X) if matrices[0] is not None else torch.zeros(plan.children,ix.numel(),device=parent.device,dtype=parent.dtype)
        for A,r in zip(matrices[1:],(1,2,4,5)):
            if A is not None:Y=Y+torch.sparse.mm(A,X.conj() if r>=4 else X)*_phase_factor(p,r)
        if ix.numel()==bank.V:out=Y
        else:out.index_copy_(1,ix,Y)
    return out

def apply_rf(plan,bank,parent,phase,backend='auto'):
    if phase.ndim:phase=phase.reshape(-1)
    automatic=backend=='auto'
    if backend=='auto':
        matrix_bytes=5*plan.children*plan.parents*len(bank.case_indices)*parent.element_size()
        dense_enough=len(plan.src)>=0.20*plan.children*plan.parents
        needs_grad=torch.is_grad_enabled() and (parent.requires_grad or bank.table.requires_grad or phase.requires_grad)
        if not parent.is_cuda and not needs_grad and parent.dtype==torch.complex64 and bank.table.dtype==torch.complex64:
            backend='dense' if bank.uniform and dense_enough and matrix_bytes<=256*1024**2 else 'native'
        elif bank.uniform and not needs_grad:
            backend='dense' if dense_enough and matrix_bytes<=256*1024**2 else 'sparse'
        else:backend='triton' if parent.is_cuda and not needs_grad else 'grouped'
    if backend=='dense':return rf_dense(plan,bank,parent,phase)
    if backend=='grouped':return rf_grouped(plan,bank,parent,phase)
    if backend=='sparse':return rf_sparse(plan,bank,parent,phase)
    if backend=='native':
        try:
            from mrtini_prepass_rs import apply_rf_cpu
        except ImportError:
            if automatic:return rf_grouped(plan,bank,parent,phase)
            raise RuntimeError('Rebuild the Rust extension with apply_rf_cpu first')
        if parent.requires_grad or bank.table.requires_grad or phase.requires_grad:
            raise ValueError('Native RF is an inference kernel; use grouped/dense for autograd')
        t=plan.tensors(bank)
        ep=torch.exp(1j*phase.expand(bank.V)).to(dtype=torch.complex64).contiguous()
        return apply_rf_cpu(parent.contiguous(),bank.table,ep,bank.ids,t['rows'],t['src'],t['slot'],t['rel'],t['mask'])
    if backend=='reference':return rf_reference(plan,bank,parent,phase)
    if backend=='triton':
        try:
            from .triton_rf import rf_triton
        except (ImportError,ModuleNotFoundError) as e:
            raise RuntimeError(
                'Fused CUDA RF backend unavailable. Install MRtini[cuda] in the '
                'release torch 2.8 environment (triton-windows on Windows), or '
                'explicitly choose rf_backend="grouped" for the slower torch path. '
                'Automatic execution will not silently fall back on this graph.'
            ) from e
        return rf_triton(plan,bank,parent,phase)
    raise ValueError(f'Unknown RF backend {backend}')

class FastRFContext:
    def __init__(self,graph,art,data,backend='auto',max_cached_matrix_bytes=256*1024**2):
        self.graph,self.art,self.data,self.backend=graph,art,data,backend;self.plans={};self.banks={}
        self.max_cached_matrix_bytes=max_cached_matrix_bytes;self._dense_lru=OrderedDict();self._dense_bytes=0
        device=data.PD.device;physical=getattr(data,'species_id',torch.zeros_like(data.PD,dtype=torch.long)).reshape(-1).to(device=device)
        self.physical=physical
        labels=graph.case_species if isinstance(graph,PackedGraph) else [0,1]
        ids=torch.zeros_like(physical,dtype=torch.long)
        for i,label in enumerate(labels):ids=torch.where(physical==label,torch.full_like(ids,i),ids)
        if any(bool(((physical==s)&(data.PD.reshape(-1)!=0)).any()) for s in [0,1] if s not in labels):
            raise ValueError('Graph omits a species with nonzero proton density')
        self.ids=ids
    def layer(self,i,key,phase,parent=None,live=None):
        if i not in self.plans:self.plans[i]=self.graph.rf_plan(i) if isinstance(self.graph,PackedGraph) else RFPlan.from_graph(self.graph[i],self.graph[i+1])
        if key not in self.banks:self.banks[key]=RFBank(self.art.pulses[key],self.physical,self.ids)
        plan=self.plans[i];bank=self.banks[key];previous=self.graph[i]
        if parent is None:
            zero=torch.zeros_like(self.data.PD,dtype=torch.cfloat).reshape(-1)
            live=[p.mag is not None for p in previous]
            parent=torch.stack([p.mag.reshape(-1) if p.mag is not None else zero for p in previous])
        result=apply_rf(plan,bank,parent,phase,self.backend)
        if i not in self._dense_lru:
            def nbytes(a):
                if a.layout==torch.sparse_csr:return a.values().numel()*a.element_size()+(a.crow_indices().numel()+a.col_indices().numel())*8
                return a.numel()*a.element_size()
            size=sum(nbytes(a) for matrices in plan.dense_cache.values() for a in matrices if a is not None)
            if size:self._dense_lru[i]=size;self._dense_bytes+=size
        elif plan.dense_cache:self._dense_lru.move_to_end(i)
        while self._dense_bytes>self.max_cached_matrix_bytes and self._dense_lru:
            old,size=self._dense_lru.popitem(last=False)
            self.plans[old].dense_cache.clear();self._dense_bytes-=size
        return result,plan,live
