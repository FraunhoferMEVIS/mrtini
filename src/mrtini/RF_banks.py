'''
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 <Fraunhofer MEVIS/ Snawar Hussain>.
#
# This file is part of MRtini Simulator.
# It contains original code written for MRtini Simulator.
'''
# rf_kernel_bank_so3.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Literal
import math, numpy as np, torch
import sys
sys.path.append("./")
from src.mrtini.config import DTYPE
CMPLX_DTYPE = torch.complex64 if DTYPE==torch.float32 else torch.complex128





# ------------------------- small utils -------------------------

def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x=np.asarray(x,np.float64).reshape(-1); w=np.asarray(w,np.float64).reshape(-1)
    m=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not np.any(m):
        xf=x[np.isfinite(x)]
        return float(np.nanmedian(xf)) if xf.size else 1.0
    x=x[m]; w=w[m]
    idx=np.argsort(x); x=x[idx]; w=w[idx]
    cw=np.cumsum(w)
    return float(x[np.searchsorted(cw,0.5*cw[-1],side="left")])

def pd_weighted_b1_ref(B1: np.ndarray, PD: np.ndarray, mode: Literal["wmedian","wmean"]="wmedian") -> float:
    B1=np.asarray(B1,np.float64).reshape(-1)
    PD=np.clip(np.asarray(PD,np.float64).reshape(-1),0.0,None)
    if mode=="wmean":
        m=np.isfinite(B1)&np.isfinite(PD)&(PD>0)
        if not np.any(m):
            bf=B1[np.isfinite(B1)]
            return float(np.nanmedian(bf)) if bf.size else 1.0
        return float((B1[m]*PD[m]).sum()/(PD[m].sum()+1e-12))
    return weighted_median(B1,PD)

# ------------------------- axes + mapping -------------------------

@dataclass(frozen=True)
class BankAxes:
    b0_centers_hz: torch.Tensor  # (Nb0,)
    b1_centers:    torch.Tensor  # (Nb1,)
    Nb0: int
    Nb1: int

def make_axes(b0_centers_hz, b1_centers, device="cuda") -> BankAxes:
    b0=torch.as_tensor(b0_centers_hz,device=device).reshape(-1)
    b1=torch.as_tensor(b1_centers,   device=device).reshape(-1)
    b0,_=torch.sort(b0); b1,_=torch.sort(b1)
    return BankAxes(b0_centers_hz=b0, b1_centers=b1, Nb0=int(b0.numel()), Nb1=int(b1.numel()))

@dataclass(frozen=True)
class IMap:
    i0: torch.Tensor; i1: torch.Tensor
    j0: torch.Tensor; j1: torch.Tensor
    tx: torch.Tensor; ty: torch.Tensor
    valid: torch.Tensor


@torch.no_grad()
def build_imap(B0_hz_vox, B1_vox, ax: BankAxes, device="cuda", clamp=True) -> IMap:
    B0 = torch.as_tensor(B0_hz_vox, device=device, dtype=DTYPE).reshape(-1)
    B1 = torch.as_tensor(B1_vox, device=device, dtype=DTYPE).reshape(-1)
    valid = torch.isfinite(B0) & torch.isfinite(B1)
    if clamp:
        B0 = B0.clamp(ax.b0_centers_hz.min(), ax.b0_centers_hz.max())
        B1 = B1.clamp(ax.b1_centers.min(), ax.b1_centers.max())
    i0 = (torch.searchsorted(ax.b0_centers_hz, B0,
          right=True) - 1).clamp(0, ax.Nb0 - 2)
    i1 = i0 + 1
    j0 = (torch.searchsorted(ax.b1_centers, B1,
          right=True) - 1).clamp(0, ax.Nb1 - 2)
    j1 = j0 + 1
    tx = (B0 - ax.b0_centers_hz[i0]) / \
        (ax.b0_centers_hz[i1] - ax.b0_centers_hz[i0] + 1e-12)
    ty = (B1 - ax.b1_centers[j0]) / \
        (ax.b1_centers[j1] - ax.b1_centers[j0] + 1e-12)
    return IMap(i0=i0.long(), i1=i1.long(), j0=j0.long(), j1=j1.long(), tx=tx, ty=ty, valid=valid)


@torch.no_grad()
def pair_weights_from_imap_pd(im: IMap, ax: BankAxes, PD_vox, device) -> torch.Tensor:
    PD=torch.as_tensor(PD_vox,device=device,dtype=DTYPE).reshape(-1).clamp_min(0)
    W=torch.zeros((ax.Nb0,ax.Nb1),device=device,dtype=DTYPE)
    v=im.valid&torch.isfinite(PD)&(PD>0)
    if not bool(v.any()): return W
    tx,ty=im.tx[v],im.ty[v]
    w00=(1-tx)*(1-ty); w10=tx*(1-ty); w01=(1-tx)*ty; w11=tx*ty
    pd=PD[v]
    i0,i1,j0,j1=im.i0[v],im.i1[v],im.j0[v],im.j1[v]
    W.index_put_((i0,j0),pd*w00,accumulate=True)
    W.index_put_((i1,j0),pd*w10,accumulate=True)
    W.index_put_((i0,j1),pd*w01,accumulate=True)
    W.index_put_((i1,j1),pd*w11,accumulate=True)
    return W

# ------------------------- 6-coeff container -------------------------

@dataclass
class Coeff6:
    # each (..., Nk) complex64
    pp: torch.Tensor; pz: torch.Tensor; mp: torch.Tensor
    mz: torch.Tensor; zp: torch.Tensor; zz: torch.Tensor
    
    def select_k(self, k_idx: torch.Tensor) -> "Coeff6":
        k_idx=k_idx.long()
        return Coeff6(pp=self.pp[...,k_idx], pz=self.pz[...,k_idx], mp=self.mp[...,k_idx],
                      mz=self.mz[...,k_idx], zp=self.zp[...,k_idx], zz=self.zz[...,k_idx])
    def with_rf_phase(self, phi: torch.Tensor|float) -> "Coeff6":
        phi=torch.as_tensor(phi,device=self.pp.device,dtype=DTYPE)
        em1 = torch.exp( -1j*phi.to(CMPLX_DTYPE))
        ep1= torch.conj(em1)
        ep2= torch.exp(+2j*phi.to(CMPLX_DTYPE))
        return Coeff6(pp=self.pp, pz=self.pz*em1, mp=self.mp*ep2, mz=self.mz*ep1, zp=self.zp*ep1, zz=self.zz)

def energy_from_coeff6(c6: Coeff6) -> torch.Tensor:
    E=(c6.pp.abs()**2 + c6.pz.abs()**2 + c6.mp.abs()**2 + c6.mz.abs()**2 + c6.zp.abs()**2 + c6.zz.abs()**2)
    return E.to(DTYPE)


@dataclass
class RotTaps1D:
    # Nine tap sequences (fftshifted so delta=0 is centered). Shape: (K,)
    pp: torch.Tensor
    pz: torch.Tensor
    mp: torch.Tensor
    mz: torch.Tensor
    zp: torch.Tensor
    zz: torch.Tensor
    # convenience: a dict view
    as_dict: Dict[str, torch.Tensor]
    # signed delta indices and physical deltak (cycles/m)
    deltas: torch.Tensor  # ints, length K, e.g. [-K//2,...,+K//2-1]
    dk: torch.Tensor  # cycles/m per delta
    dtype: torch.dtype = DTYPE
# ------------------------- SO(3) helpers (Option A) -------------------------

def _qnorm(q, eps=1e-12): return q/(q.norm(dim=-1,keepdim=True).clamp_min(eps))

def q_slerp(q0, q1, t, eps=1e-8):
    q0=_qnorm(q0); q1=_qnorm(q1)
    dot=(q0*q1).sum(-1,keepdim=True)
    q1=torch.where(dot<0,-q1,q1); dot=dot.abs().clamp(-1,1)
    t=t.unsqueeze(-1) if t.ndim==dot.ndim-1 else t
    close=(dot>1-1e-6)
    qlin=_qnorm(q0+t*(q1-q0))
    th0=torch.acos(dot.clamp(-1+eps,1-eps)); s0=torch.sin(th0).clamp_min(eps)
    th=th0*t
    a=torch.sin(th0-th)/s0; b=torch.sin(th)/s0
    qs=_qnorm(a*q0+b*q1)
    return torch.where(close.expand_as(qs), qlin, qs)

def R_to_q(R):
    # R: (...,3,DTYPE -> q: (...,4) [w,x,y,z]
    R=R.to(DTYPE)
    r00,r01,r02 = R[...,0,0],R[...,0,1],R[...,0,2]
    r10,r11,r12 = R[...,1,0],R[...,1,1],R[...,1,2]
    r20,r21,r22 = R[...,2,0],R[...,2,1],R[...,2,2]
    tr = r00+r11+r22
    q=torch.empty(R.shape[:-2]+(4,),device=R.device,dtype=DTYPE)

    m0 = tr>0
    s = torch.sqrt(tr[m0]+1.0)*2.0
    q[m0,0]=0.25*s
    q[m0,1]=(r21[m0]-r12[m0])/s
    q[m0,2]=(r02[m0]-r20[m0])/s
    q[m0,3]=(r10[m0]-r01[m0])/s

    m1 = (~m0) & (r00>=r11) & (r00>=r22)
    s = torch.sqrt(1.0+r00[m1]-r11[m1]-r22[m1])*2.0
    q[m1,0]=(r21[m1]-r12[m1])/s
    q[m1,1]=0.25*s
    q[m1,2]=(r01[m1]+r10[m1])/s
    q[m1,3]=(r02[m1]+r20[m1])/s

    m2 = (~m0) & (~m1) & (r11>=r22)
    s = torch.sqrt(1.0+r11[m2]-r00[m2]-r22[m2])*2.0
    q[m2,0]=(r02[m2]-r20[m2])/s
    q[m2,1]=(r01[m2]+r10[m2])/s
    q[m2,2]=0.25*s
    q[m2,3]=(r12[m2]+r21[m2])/s

    m3 = (~m0) & (~m1) & (~m2)
    s = torch.sqrt(1.0+r22[m3]-r00[m3]-r11[m3])*2.0
    q[m3,0]=(r10[m3]-r01[m3])/s
    q[m3,1]=(r02[m3]+r20[m3])/s
    q[m3,2]=(r12[m3]+r21[m3])/s
    q[m3,3]=0.25*s

    return _qnorm(q)

@torch.no_grad()
def qbank_from_Rxyz_bank(Rxyz_bank: torch.Tensor) -> torch.Tensor:
    # Rxyz_bank: (Nb0,Nb1,Nz,3,DTYPE
    Nb0,Nb1,Nz = Rxyz_bank.shape[:3]
    q = R_to_q(Rxyz_bank.reshape(-1,3,3)).reshape(Nb0,Nb1,Nz,4).contiguous()
    return q


## ========TESTED and Confirmed with sign convention in mr0========
@torch.no_grad()
def rch6_from_q_old(q):
    # Convention: '+' state stores M_- = (Mx - i My)/sqrt(2)
    q=_qnorm(q)
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    rxx=1-2*(y*y+z*z); rxy=2*(x*y-w*z); rxz=2*(x*z+w*y)
    ryx=2*(x*y+w*z);   ryy=1-2*(x*x+z*z); ryz=2*(y*z-w*x)
    rzx=2*(x*z-w*y);   rzy=2*(y*z+w*x);   rzz=1-2*(x*x+y*y)

    I=torch.tensor(1j,device=q.device,dtype=CMPLX_DTYPE)
    s2=math.sqrt(2.0)

    # '+' == M_- basis => entries are in [M_-, M_+, Z]
    pp = 0.5*(rxx+ryy).to(CMPLX_DTYPE) + 0.5*I*(rxy-ryx).to(CMPLX_DTYPE)   # +<-+
    mp = 0.5*(rxx-ryy).to(CMPLX_DTYPE) - 0.5*I*(rxy+ryx).to(CMPLX_DTYPE)   # +<-(-)  (i.e. +<-M_+)
    zp = (rxz.to(CMPLX_DTYPE) - I*ryz.to(CMPLX_DTYPE))/s2                 # +<-Z   (Z->+)
    pz = (rzx.to(CMPLX_DTYPE) + I*rzy.to(CMPLX_DTYPE))/s2                 # Z<-+   (+->Z)
    mz = (rzx.to(CMPLX_DTYPE) - I*rzy.to(CMPLX_DTYPE))/s2                 # Z<-(-) (i.e. Z<-M_+)
    zz = rzz.to(CMPLX_DTYPE)                                              # Z<-Z
    return pp, pz, mp, mz, zp, zz

@torch.no_grad()
def rch6_from_q(q):
    # Convention: '+' state is m+ = (Mx + i My)/sqrt(2), '-' is m- = (Mx - i My)/sqrt(2)
    q=_qnorm(q)
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    rxx=1-2*(y*y+z*z); rxy=2*(x*y-w*z); rxz=2*(x*z+w*y)
    ryx=2*(x*y+w*z);   ryy=1-2*(x*x+z*z); ryz=2*(y*z-w*x)
    rzx=2*(x*z-w*y);   rzy=2*(y*z+w*x);   rzz=1-2*(x*x+y*y)

    I=torch.tensor(1j,device=q.device,dtype=CMPLX_DTYPE)
    s2=math.sqrt(2.0)

    pp = 0.5*(rxx+ryy).to(CMPLX_DTYPE) + 0.5*I*(ryx-rxy).to(CMPLX_DTYPE)   # ++ : + -> +
    mp = 0.5*(rxx-ryy).to(CMPLX_DTYPE) + 0.5*I*(ryx+rxy).to(CMPLX_DTYPE)   # -+ : - -> +
    zp = (rxz.to(CMPLX_DTYPE) + I*ryz.to(CMPLX_DTYPE))/s2                  # z+ : z -> +
    pz = (rzx.to(CMPLX_DTYPE) - I*rzy.to(CMPLX_DTYPE))/s2                  # +z : + -> z
    mz = (rzx.to(CMPLX_DTYPE) + I*rzy.to(CMPLX_DTYPE))/s2                  # -z : - -> z
    zz = rzz.to(CMPLX_DTYPE)                                               # zz : z -> z
    return pp, pz, mp, mz, zp, zz
## ========TESTED and Confirmed with sign convention in mr0========

# optional: build full R_ch later (debug/legacy)
def _basis_mats(device):
    S=(1/math.sqrt(2))*torch.tensor([[1,1j,0],[1,-1j,0],[0,0,math.sqrt(2)]], device=device, dtype=CMPLX_DTYPE)
    Sinv=S.conj().T
    P=torch.diag(torch.tensor([1.,1.,-1.], device=device, dtype=CMPLX_DTYPE))
    return S,Sinv,P

@torch.no_grad()
def Rch_from_Rxyz(Rxyz: torch.Tensor) -> torch.Tensor:
    # Rxyz: (...,3,3) real -> Rch: (...,3,3) complex
    dev=Rxyz.device
    S,Sinv,P=_basis_mats(dev)
    R = Rxyz.to(CMPLX_DTYPE)
    # Rch = P S Rxyz S^{-1} P
    tmp = torch.einsum("ab,...b c->...a c", S, R)          # S*R
    tmp = torch.einsum("...a b,bc->...a c", tmp, Sinv)     # S*R*Sinv
    tmp = torch.einsum("ab,...b c->...a c", P, tmp)
    tmp = torch.einsum("...a b,bc->...a c", tmp, P)
    return tmp

# ------------------------- k-grid + projector -------------------------

@dataclass(frozen=True)
class KGrid:
    kz: torch.Tensor     # (Nk,) cycles/m
    deltas: torch.Tensor # (Nk,) int64 delta orders
    dk_tap: float        # cycles/m
    M: int               # dk_tap = M * (1/inv_kt_z)

@torch.no_grad()
def build_kgrid_projection( kmax: float=None, z_m: torch.Tensor=None, device="cuda") -> KGrid:
    # inv_kt_z MUST be inv_kt_grid[2] (the key scaler used later in PDG), not 1/inv_kt_z.
    z = z_m.to(DTYPE).reshape(-1)
    K = z.numel()
    dz = float((z[1]-z[0]).item())
    Lz = K * dz
    dk = 1.0 / Lz  # cycles/m
    M = K//2  # full FFT range
    deltas = torch.arange(-M, M+1, device=device, dtype=torch.int64)
    kz = deltas.to(DTYPE) * dk
    return KGrid(kz=kz, deltas=deltas, dk_tap=float(dk), M=K//2)


@torch.no_grad()
def make_projector_z_to_k(z_m: torch.Tensor, kz: torch.Tensor, *, norm: str="K", bake_ifftshift: bool=True) -> torch.Tensor:
    """
    Returns P so that: H = x @ P
    - If bake_ifftshift=False: x is assumed aligned with z_m (your normal centered storage).
    - If bake_ifftshift=True: x is still centered storage, but P is built so that x @ P == ifftshift(x) @ P_base
    where P_base is built on the ifftshifted coordinate order.
    """
    z  = z_m.to(DTYPE).reshape(-1)
    kz = kz.to(z.device).to(DTYPE).reshape(-1)
    K  = z.numel()
    dz = float((z[1]-z[0]).item())
    Lz = K * dz

    if norm == "dz":      scale = dz
    elif norm == "Lz":    scale = dz / Lz   # == 1/K (up to rounding)
    elif norm == "K":     scale = 1.0 / K
    else: raise ValueError("norm must be one of {'dz','Lz','K'}")

    if not bake_ifftshift:
        # “physical / aligned” projector: samples x[n] correspond to z[n]
        E = torch.exp(-1j * 2*math.pi * (z[:,None].to(CMPLX_DTYPE) * kz[None,:].to(CMPLX_DTYPE)))
        return (scale * E).contiguous()

    # FFT-like path:
    # 1) build base projector on FFT-ordered coordinates z_u
    z_u = torch.fft.ifftshift(z)  # same permutation you'd apply to x before torch.fft.fft
    E_base = torch.exp(-1j * 2*math.pi * (z_u[:,None].to(CMPLX_DTYPE) * kz[None,:].to(CMPLX_DTYPE)))
    P_base = (scale * E_base)

    # 2) bake the sample ifftshift into P so you can keep x centered:
    #    (ifftshift(x) @ P_base) == (x @ ifftshift(P_base, dim=0))
    P_baked = torch.fft.ifftshift(P_base, dim=0)
    return P_baked.contiguous()
# ------------------------- energy support -------------------------

@dataclass(frozen=True)
class EnergySupport:
    k_idx: torch.Tensor   # (M,) indices into kgrid.kz
    deltas: torch.Tensor  # (M,) delta orders
    p: torch.Tensor       # (Nk,) normalized energy
    E: torch.Tensor       # (Nk,) raw energy

@torch.no_grad()
def select_support_by_energy(nom6: Coeff6, kgrid: KGrid, eps=1e-8) -> EnergySupport:
    E = energy_from_coeff6(nom6)     # (Nk,)
    Et = E.sum()
    if float(Et) <= 0:
        k_idx=torch.empty((0,),dtype=torch.int32,device=E.device)
        p=torch.zeros_like(E)
        return EnergySupport(k_idx=k_idx, deltas=kgrid.deltas[k_idx], p=p, E=E)
    p=E/Et
    order=torch.argsort(p,descending=True)
    c=torch.cumsum(p[order],dim=0)
    if eps <= 0:
        keep_n = order.numel()
    else:
        target = 1.0 - float(eps)
        keep_n = int(torch.searchsorted(c, torch.tensor(target, device=c.device, dtype=c.dtype), right=False).item()) + 1
        keep_n = min(keep_n, order.numel())
    k_idx=torch.sort(order[:keep_n]).values.to(torch.int32)
    return EnergySupport(k_idx=k_idx, deltas=kgrid.deltas[k_idx], p=p, E=E)

# ------------------------- main bank (Option A) -------------------------

@dataclass
class RFKernel6BankSO3:
    ax: BankAxes
    z_m: torch.Tensor                   # (Nz,)
    Rxyz_bank: torch.Tensor             # (Nb0,Nb1,Nz,3,DTYPE

    kgrid: Optional[KGrid] = None
    P_z2k: Optional[torch.Tensor] = None          # (Nz,Nk) complex64

    # bin taps (no interpolation)
    H6_bank: Optional[Coeff6] = None              # each (Nb0,Nb1,Nk) complex64

    # SO(3) cache for voxel interpolation
    q_bank: Optional[torch.Tensor] = None         # (Nb0,Nb1,Nz,DTYPE

    # nominal taps for PDG prepass
    nominal: Optional[Coeff6] = None              # (Nk,) complex64

    @staticmethod
    @torch.no_grad()
    def from_Rxyz_flat(Rxyz_flat: torch.Tensor, ax: BankAxes, z_m: torch.Tensor,
                       *, device="cuda",
                       flat_order: Literal["b0_major_b1_minor","b1_major_b0_minor"]="b0_major_b1_minor",
                       in_layout: Literal["BNz33","B33Nz"]="BNz33") -> "RFKernel6BankSO3":
        """
        Rxyz_flat:
          - if in_layout=="SBNz33": (B, Nz, 3, 3) float
          - if in_layout=="SB33Nz": (B, 3, 3, Nz) float
        """
        R=torch.as_tensor(Rxyz_flat, device=device, dtype=DTYPE)
        z=torch.as_tensor(z_m, device=device, dtype=DTYPE).reshape(-1)
        B = R.shape[0]
        assert B==ax.Nb0*ax.Nb1, f"B={B} must equal Nb0*Nb1={ax.Nb0*ax.Nb1}"
        if in_layout=="B33Nz":
            R = R.permute(0,3,1,2).contiguous()  # -> (B,Nz,3,3)
        Nz = R.shape[1]
        assert z.numel()==Nz, "z_m must match Nz"
        if flat_order=="b0_major_b1_minor":
            Rg = R.reshape(ax.Nb0, ax.Nb1, Nz, 3, 3)
        else:
            Rg = R.reshape(ax.Nb1, ax.Nb0, Nz, 3, 3).permute(1,0,2,3,4).contiguous()
        return RFKernel6BankSO3(ax=ax, z_m=z, Rxyz_bank=Rg)

    @torch.no_grad()
    def attach_kgrid(self, kgrid: KGrid) -> None:
        self.kgrid = kgrid
        self.P_z2k = make_projector_z_to_k(self.z_m, kgrid.kz).to(self.Rxyz_bank.device)
        self.H6_bank = None
        self.nominal = None

    @torch.no_grad()
    def ensure_qbank(self) -> None:
        if self.q_bank is None:
            self.q_bank = qbank_from_Rxyz_bank(self.Rxyz_bank)

    @torch.no_grad()
    def ensure_H6_bins(self) -> None:
        # build bin taps (no interpolation): q_bank -> rch6(z) -> project
        if self.H6_bank is not None: return
        if self.P_z2k is None: raise ValueError("Call attach_kgrid(kgrid) first.")
        self.ensure_qbank()
        P = self.P_z2k  # (Nz,Nk)
        qB = self.q_bank  # (Nb0,Nb1,Nz,4)
        pp_z,pz_z,mp_z,mz_z,zp_z,zz_z = rch6_from_q(qB)   # each (Nb0,Nb1,Nz)
        # project: (Nb0,Nb1,Nz) @ (Nz,Nk) -> (Nb0,Nb1,Nk)
        K = P.shape[-1]
        pp = torch.einsum("ijn,nk->ijk", pp_z, P)
        pz = torch.einsum("ijn,nk->ijk", pz_z, P)
        mp = torch.einsum("ijn,nk->ijk", mp_z, P)
        mz = torch.einsum("ijn,nk->ijk", mz_z, P)
        zp = torch.einsum("ijn,nk->ijk", zp_z, P)
        zz = torch.einsum("ijn,nk->ijk", zz_z, P)
        self.H6_bank = Coeff6(pp=pp.contiguous(), pz=pz.contiguous(), mp=mp.contiguous(),
                              mz=mz.contiguous(), zp=zp.contiguous(), zz=zz.contiguous())

    @torch.no_grad()
    def compute_nominal_PD_weighted(self, im: IMap, PD_vox) -> Coeff6:
        """
        Nominal taps for PDG prepass:
        - NO SO(3) interpolation
        - PD-weighted linear average over the bin taps (correct physics for mixture)
        """
        self.ensure_H6_bins()
        hb = self.H6_bank
        W = pair_weights_from_imap_pd(im, self.ax, PD_vox, device=self.Rxyz_bank.device)
        s = W.sum().clamp_min(1e-12)
        Wn = (W/s).view(self.ax.Nb0, self.ax.Nb1, 1).to(CMPLX_DTYPE)
        nom = Coeff6(pp=(hb.pp*Wn).sum((0,1)),
                     pz=(hb.pz*Wn).sum((0,1)),
                     mp=(hb.mp*Wn).sum((0,1)),
                     mz=(hb.mz*Wn).sum((0,1)),
                     zp=(hb.zp*Wn).sum((0,1)),
                     zz=(hb.zz*Wn).sum((0,1)))
        self.nominal = nom
        return nom

    @torch.no_grad()
    def eval_voxels_so3(self, im: IMap, k_idx: torch.Tensor, *, chunk_vox: int=4096) -> Tuple[Coeff6, Dict[int,int]]:
        """
        Voxel taps:
          - SLERP in SO(3) in z-domain (q_bank)
          - convert to rch6(z)
          - project to selected k bins
        """
        self.ensure_qbank()
        if self.P_z2k is None: raise ValueError("Call attach_kgrid(kgrid) first.")
        dev = self.Rxyz_bank.device
        k_idx = k_idx.to(dev).long().reshape(-1)
        Psel = self.P_z2k[:, k_idx]  # (Nz,Nk_sel)

        qB = self.q_bank
        Nvox = im.i0.numel(); Nk = k_idx.numel()
        out={k: torch.empty((Nvox,Nk),device=dev,dtype=CMPLX_DTYPE) for k in ["pp","pz","mp","mz","zp","zz"]}

        for s in range(0,Nvox,chunk_vox):
            e=min(Nvox,s+chunk_vox); sl=slice(s,e)
            i0,i1,j0,j1=im.i0[sl],im.i1[sl],im.j0[sl],im.j1[sl]
            tx=im.tx[sl].view(-1,1,1); ty=im.ty[sl].view(-1,1,1)

            q00=qB[i0,j0]; q10=qB[i1,j0]; q01=qB[i0,j1]; q11=qB[i1,j1]  # (C,Nz,4)
            qx0=q_slerp(q00,q10,tx); qx1=q_slerp(q01,q11,tx); qv=q_slerp(qx0,qx1,ty)

            pp_z,pz_z,mp_z,mz_z,zp_z,zz_z = rch6_from_q(qv)  # each (C,Nz)
            out["pp"][sl]=pp_z @ Psel; out["pz"][sl]=pz_z @ Psel; out["mp"][sl]=mp_z @ Psel
            out["mz"][sl]=mz_z @ Psel; out["zp"][sl]=zp_z @ Psel; out["zz"][sl]=zz_z @ Psel

        for k in out:
            out[k]=torch.where(im.valid[:,None], out[k], torch.zeros_like(out[k]))
        
        k_supp_map = {int(full): int(j) for j, full in enumerate(k_idx)}
        return Coeff6(**out), k_supp_map

    # ---- debug helpers ----
    @torch.no_grad()
    def eval_bins_rch(self) -> torch.Tensor:
        """Return full R_ch_bank on demand (Nb0,Nb1,Nz,3,3) complex64."""
        return Rch_from_Rxyz(self.Rxyz_bank)

# ------------------------- tests you should actually run -------------------------

@torch.no_grad()
def check_rotations(Rxyz: torch.Tensor, tol_ortho=3e-3, tol_det=3e-3, name="Rxyz"):
    # Rxyz: (...,3,3)
    I = torch.eye(3, device=Rxyz.device, dtype=Rxyz.dtype)
    RtR = Rxyz.transpose(-1,-2) @ Rxyz
    ortho_err = (RtR - I).abs().amax().item()
    det = torch.det(Rxyz)
    det_err = (det - 1.0).abs().amax().item()
    print(f"[{name}] ortho_err(max)={ortho_err:.3e} det_err(max)={det_err:.3e}")
    assert ortho_err < tol_ortho, f"{name}: not orthonormal enough (err {ortho_err})"
    assert det_err   < tol_det,   f"{name}: det not +1 enough (err {det_err})"

@torch.no_grad()
def test_bin_identity(bank: RFKernel6BankSO3, n_test=8):
    """
    If we query voxels exactly at bin centers (tx=ty=0) the voxel taps should match bin taps.
    This catches broken indexing, broken q conversion, broken rch6, broken projector.
    """
    bank.ensure_H6_bins()
    k_idx = torch.arange(0, bank.kgrid.kz.numel(), device=bank.Rxyz_bank.device)  # all bins
    # craft a fake imap that hits exact bin corners:
    Nb0,Nb1 = bank.ax.Nb0, bank.ax.Nb1
    ii = torch.randint(0, Nb0-1, (n_test,), device=bank.Rxyz_bank.device)
    jj = torch.randint(0, Nb1-1, (n_test,), device=bank.Rxyz_bank.device)
    im = IMap(i0=ii, i1=ii+1, j0=jj, j1=jj+1,
              tx=torch.zeros_like(ii,dtype=DTYPE),
              ty=torch.zeros_like(jj,dtype=DTYPE),
              valid=torch.ones_like(ii,dtype=torch.bool))
    vox = bank.eval_voxels_so3(im, k_idx, chunk_vox=n_test)
    hb = bank.H6_bank
    # expected is exactly bin (i0,j0)
    exp = Coeff6(pp=hb.pp[ii,jj], pz=hb.pz[ii,jj], mp=hb.mp[ii,jj], mz=hb.mz[ii,jj], zp=hb.zp[ii,jj], zz=hb.zz[ii,jj])
    err = (vox.pp-exp.pp).abs().amax().item()
    print(f"[test_bin_identity] max|Δpp|={err:.3e}")
    assert err < 2e-3, "bin identity failed (likely convention / projector / indexing bug)"

def R_axis_angle(axis, ang, device="cuda", dtype=DTYPE):
    a=torch.as_tensor(axis,device=device,dtype=dtype); a=a/torch.linalg.norm(a)
    x,y,z=a
    c=math.cos(ang); s=math.sin(ang); C=1-c
    return torch.tensor([
      [c+x*x*C, x*y*C - z*s, x*z*C + y*s],
      [y*x*C + z*s, c+y*y*C, y*z*C - x*s],
      [z*x*C - y*s, z*y*C + x*s, c+z*z*C]
    ], device=device, dtype=dtype)
def S_minus(device, cdtype):
    s=1/math.sqrt(2)
    S=torch.tensor([[s, -1j*s, 0],
                    [s, +1j*s, 0],
                    [0, 0, 1]], device=device, dtype=cdtype)
    return S, S.conj().T

def S_fplus(device, cdtype=torch.complex128):
    s=1/math.sqrt(2)
    S=torch.tensor([[s, 1j*s, 0],
                    [s,-1j*s, 0],
                    [0, 0, 1]], device=device, dtype=cdtype)
    return S, S.conj().T

@torch.no_grad()
def gold6_from_Rxyz(Rxyz):
    cdtype = torch.complex128 if Rxyz.dtype==DTYPE else CMPLX_DTYPE
    S,Sinv = S_minus(Rxyz.device, cdtype)
    Rpmz = S @ Rxyz.to(cdtype) @ Sinv  # [M_-, M_+, Z], row=out col=in
    pp = Rpmz[0,0]; mp = Rpmz[0,1]; zp = Rpmz[0,2]
    pz = Rpmz[2,0]; mz = Rpmz[2,1]; zz = Rpmz[2,2]
    return pp,pz,mp,mz,zp,zz

@torch.no_grad()
def test_rch6_matches_gold():
    for _ in range(50):
        axis = torch.randn(3, device="cuda", dtype=DTYPE); axis/=axis.norm()
        ang  = float(torch.rand(1, device="cuda")*2*math.pi)
        Rxyz = R_axis_angle(axis, ang, device="cuda", dtype=DTYPE)
        q = R_to_q(Rxyz)                      # your function
        pp,pz,mp,mz,zp,zz = rch6_from_q(q)    # YOUR function under test
        gpp,gpz,gmp,gmz,gzp,gzz = gold6_from_Rxyz(Rxyz)
        err = max((pp-gpp).abs(), (pz-gpz).abs(), (mp-gmp).abs(),
                  (mz-gmz).abs(), (zp-gzp).abs(), (zz-gzz).abs()).item()
        assert err < 1e-10, err
        
        
def build_prepass_inputs_from_hybrid(seq, art):
    # 1) map hybrid bloch spans -> per-repetition key
    bloch_span = [si for si,s in enumerate(seq.mrtini_hybrid) if s["kind"]=="bloch"]
    R = len(list(seq))  # repetitions you pass to Rust
    assert len(bloch_span) >= R, "Not enough bloch spans for repetitions"
    rep_keys = [art.span_key[bloch_span[r]] for r in range(R)]

    # 2) assign pulse_id for each unique key (stable ordering)
    uniq_keys = list(dict.fromkeys(rep_keys))  # preserve first-seen order
    key_to_pid = {k:i for i,k in enumerate(uniq_keys)}
    pulse_id_per_rep = torch.tensor([key_to_pid[k] for k in rep_keys], dtype=torch.int32)

    # 3) build taps/support banks aligned with pid
    taps_bank = [art.pulses[k].combined_taps for k in uniq_keys]
    supp_bank = [art.pulses[k].water.support      for k in uniq_keys]

    return rep_keys, pulse_id_per_rep, taps_bank, supp_bank,

# ------------------------- usage snippet -------------------------
"""
ax = make_axes(b0_centers_hz, b1_centers, device)
bank = RFKernel6BankSO3.from_Rxyz_flat(Rxyz_flat, ax=ax, z_m=z_m, device=device, in_layout="BNz33")

kgrid = build_kgrid_projection(inv_kt_z=inv_kt_grid[2], N=200, kmax=18000.0, device=device)
bank.attach_kgrid(kgrid)

# nominal for PDG prepass (linear sum in H-space, no interpolation):
nom6_full = bank.compute_nominal_PD_weighted(im, PD)
support = select_support_by_energy(nom6_full, kgrid, eps=1e-8)
k_idx = support.k_idx

# voxelwise taps via SO(3):
vox6 = bank.eval_voxels_so3(im, k_idx)

# sanity checks:
check_rotations(bank.Rxyz_bank, name="Rxyz_bank")
test_bin_identity(bank, n_test=8)
"""

