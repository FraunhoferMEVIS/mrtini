# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shape-dynamic FP32 / complex64 kernels; arithmetic matches triton_evolution."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit(do_not_specialize=["V", "N"])
def _evolve(X, OUT, D, R1, R2, B, KIND, LIVE, MASK, IDS, V, N, BLOCK:tl.constexpr):
    j=tl.program_id(0); v=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK); ok=v<V
    kind=tl.load(KIND+j); live=tl.load(LIVE+j)!=0
    x=tl.load(X+2*(j*V+v),ok,other=0.); y=tl.load(X+2*(j*V+v)+1,ok,other=0.)
    d=tl.load(D+v,ok,other=0.); r1=tl.load(R1+v,ok,other=0.); r2=tl.load(R2+v,ok,other=0.)
    b=tl.load(B+j); e=libdevice.exp(d*b); relax=tl.where(kind==0,r2,r1)
    bits=tl.load(MASK+j); ids=tl.load(IDS+v,ok,other=0)
    rec=tl.where((kind==2)&(((bits>>ids)&1)!=0),1.-r1,0.)
    tl.store(OUT+2*(j*V+v),(tl.where(live,x,0.)*relax)*e+rec,ok)
    tl.store(OUT+2*(j*V+v)+1,(tl.where(live,y,0.)*relax)*e,ok)

@triton.jit(do_not_specialize=["V", "S", "CASES"])
def _state_phase(X,Q,ROWS,K,ALLOW,IDS,POS,B0,V,S,CASES,PACKED:tl.constexpr,BLOCK:tl.constexpr):
    s=tl.program_id(0); v=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK); ok=v<V
    row=tl.load(ROWS+s)
    if PACKED:
        cid=tl.load(IDS+v,ok,other=0); allowed=tl.load(ALLOW+s*CASES+cid,ok,other=0)!=0
    else: allowed=ok
    kr=tl.load(K+4*s); ky=tl.load(K+4*s+1); kz=tl.load(K+4*s+2); kt=tl.load(K+4*s+3)
    p=6.283185307179586*(kt*tl.load(B0+v,ok,other=0.)+kr*tl.load(POS+3*v,ok,other=0.)+ky*tl.load(POS+3*v+1,ok,other=0.)+kz*tl.load(POS+3*v+2,ok,other=0.))
    # Match the split phase representation in SignalPlan; not approximate trig.
    r=1.41421356237*tl.load(X+2*(row*V+v),ok&allowed,other=0.); i=1.41421356237*tl.load(X+2*(row*V+v)+1,ok&allowed,other=0.)
    c=libdevice.cos(p); z=libdevice.sin(p)
    tl.store(Q+2*(s*V+v),r*c-i*z,ok); tl.store(Q+2*(s*V+v)+1,r*z+i*c,ok)

@triton.jit(do_not_specialize=["V", "S", "A", "VL", "NV", "AL", "NA"])
def _adc(Q,T,CB,DEPH,D,T2D,T2,B0,POS,ADC,OUT,V,S,A, VL, NV, AL, NA, BS:tl.constexpr,BV:tl.constexpr):
    a=AL+tl.program_id(0); v0=tl.program_id(1)*BV+tl.arange(0,BV); v=VL+v0; ok=v0<NV
    d=tl.load(D+v,ok,other=0.); t2d=tl.load(T2D+v,ok,other=1.); t2=tl.load(T2+v,ok,other=1.)
    ss=tl.arange(0,BS); rr=tl.full((BS,BV),0.,tl.float32); ii=tl.full((BS,BV),0.,tl.float32)
    for start in tl.range(0,tl.cdiv(S,BS)):
        s=start*BS+ss; valid=s<S
        t=tl.load(T+s*A+a,valid,other=0.); cb=tl.load(CB+s*A+a,valid,other=0.); deph=tl.load(DEPH+s*A+a,valid,other=0.)
        w=libdevice.exp(-tl.abs(t[:,None])/t2d[None,:]+cb[:,None]*d[None,:])*deph[:,None]
        mask=valid[:,None]&ok[None,:]; ix=2*(s[:,None]*V+v[None,:])
        r=tl.load(Q+ix,mask,other=0.); i=tl.load(Q+ix+1,mask,other=0.)
        rr+=tl.where(mask,r*w,0.); ii+=tl.where(mask,i*w,0.)
    r=tl.sum(rr,axis=0); i=tl.sum(ii,axis=0)
    ax=tl.load(ADC+4*a); ay=tl.load(ADC+4*a+1); az=tl.load(ADC+4*a+2); at=tl.load(ADC+4*a+3)
    p=6.283185307179586*(at*tl.load(B0+v,ok,other=0.)+ax*tl.load(POS+3*v,ok,other=0.)+ay*tl.load(POS+3*v+1,ok,other=0.)+az*tl.load(POS+3*v+2,ok,other=0.))
    e=libdevice.exp(-at/t2); c=libdevice.cos(p)*e; z=libdevice.sin(p)*e
    j=2*((a-AL)*NV+v0)
    tl.store(OUT+j,r*c-i*z,ok); tl.store(OUT+j+1,r*z+i*c,ok)

def evolve(X,out,D,r1,r2,b,kind,live,mask,ids):
    n,v=X.shape
    if n and v: _evolve[(n,triton.cdiv(v,128))](torch.view_as_real(X),torch.view_as_real(out),D,r1,r2,b,kind,live,mask,ids,v,n,128,enable_fp_fusion=False,num_warps=4)

def state_phase(X,Q,rows,k,allow,ids,pos,b0):
    s,v=Q.shape
    if s and v: _state_phase[(s,triton.cdiv(v,128))](torch.view_as_real(X),torch.view_as_real(Q),rows,k,allow if allow is not None else ids,ids,pos,b0,v,s,allow.shape[1] if allow is not None else 1,allow is not None,128,enable_fp_fusion=False,num_warps=4)

def adc_tile(Q,t,cb,deph,D,T2d,T2,B0,pos,adc,out,al,ah,vl,vh):
    s,v=Q.shape; a=adc.shape[0]; nv=vh-vl; na=ah-al
    if s and nv and na: _adc[(na,triton.cdiv(nv,64))](torch.view_as_real(Q),t,cb,deph,D,T2d,T2,B0,pos,adc,torch.view_as_real(out),v,s,a,vl,nv,al,na,8,64,enable_fp_fusion=False,num_warps=4)
