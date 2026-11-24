# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optional CUDA RF kernel: CSR reduction, no E-by-V temporary or atomics.

This file requires Triton and a CUDA GPU. CPU tests exercise the same algebra
through the torch backends; this CUDA kernel needs target-device validation.
"""
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["V"])
def _rf_csr(PARENT,COEFF,PHASE,CASE,ROW,SRC,SLOT,REL,MASK,OUT,
            V,BLOCK:tl.constexpr,CASES:tl.constexpr):
    child=tl.program_id(0);v=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK);valid=v<V
    er=tl.load(PHASE+2*v,valid,other=1.);ei=tl.load(PHASE+2*v+1,valid,other=0.)
    case=tl.load(CASE+v,valid,other=0) if CASES>1 else tl.full((BLOCK,),0,tl.int32)
    ar=tl.full((BLOCK,),0.,tl.float32);ai=tl.full((BLOCK,),0.,tl.float32)
    start=tl.load(ROW+child);end=tl.load(ROW+child+1)
    for e in tl.range(start,end):
        parent=tl.load(SRC+e);slot=tl.load(SLOT+e);relation=tl.load(REL+e)
        bits=tl.load(MASK+e);active=valid&(((bits>>case)&1)!=0)
        pr=tl.load(PARENT+2*(parent*V+v),active,other=0.)
        pi=tl.load(PARENT+2*(parent*V+v)+1,active,other=0.)
        cr=tl.load(COEFF+2*(slot*V+v),active,other=0.)
        ci=tl.load(COEFF+2*(slot*V+v)+1,active,other=0.)
        if relation>=4:pi=-pi
        if relation==1:
            nr=cr*er+ci*ei;ci=ci*er-cr*ei;cr=nr
        elif relation==2 or relation==5:
            nr=cr*er-ci*ei;ci=cr*ei+ci*er;cr=nr
        elif relation==4:
            e2r=er*er-ei*ei;e2i=2.*er*ei
            nr=cr*e2r-ci*e2i;ci=cr*e2i+ci*e2r;cr=nr
        ar+=pr*cr-pi*ci;ai+=pr*ci+pi*cr
    tl.store(OUT+2*(child*V+v),ar,valid);tl.store(OUT+2*(child*V+v)+1,ai,valid)

def rf_triton(plan,bank,parent,phase,block=128,*,out=None,phase_complex=None):
    if not parent.is_cuda or parent.dtype!=torch.complex64:
        raise ValueError('Triton backend requires CUDA complex64')
    if not parent.is_contiguous():parent=parent.contiguous()
    t=plan.tensors(bank)
    p=phase.expand(bank.V) if phase.ndim==0 else phase
    ep=torch.polar(torch.ones_like(p),p).contiguous() if phase_complex is None else phase_complex
    if out is None:out=torch.empty((plan.children,bank.V),device=parent.device,dtype=parent.dtype)
    if not plan.children or not bank.V:return out
    grid=(plan.children,triton.cdiv(bank.V,block))
    _rf_csr[grid](torch.view_as_real(parent),torch.view_as_real(bank.table),torch.view_as_real(ep),bank.ids,
                  t['rows'],t['src'],t['slot'],t['rel'],t['mask'],torch.view_as_real(out),
                  V=bank.V,BLOCK=block,CASES=2 if len(bank.case_indices)>1 else 1,num_warps=4)
    return out
