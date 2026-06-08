# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 Fraunhofer MEVIS, Germany
#
# This file is part of MRtini.
import math
from typing import List, Tuple
from dataclasses import dataclass, field
import torch

import sys
sys.path.append("./")
from src.mrtini.config import DTYPE


# ---------- shared, piecewise-linear controls (no batching here) ----------
class InterpSharedLinear(torch.nn.Module):
    """
    t_grid: (T,)
    rf_complex: (T,) complex (same for all bins)
    gz: (T,) real         (same for all bins)
    eval(t_scalar) -> scalars B1x, B1y, Gz at time t_scalar via linear interp.
    """

    def __init__(self, t_grid, rf_complex, gz, *, device="cuda",
                 dtype=torch.float64):
        super().__init__()
        t = torch.as_tensor(t_grid, dtype=dtype, device=device).contiguous()
        assert (t[1:] > t[:-1]).all(), "t_grid must be strictly increasing"
        self.t = t
        rf = torch.as_tensor(rf_complex, device=device)
        self.register_buffer("B1x", rf.real.to(dtype).contiguous())  # (T,)
        self.register_buffer("B1y", rf.imag.to(dtype).contiguous())  # (T,)
        self.register_buffer(
            "Gz", torch.as_tensor(gz, device=device, dtype=dtype).contiguous()
        )  # (T,)

    def forward(self, t_scalar: torch.Tensor):
        ts = t_scalar.to(device=self.t.device, dtype=self.t.dtype)
        # find segment index i with t[i] <= ts < t[i+1]
        i = torch.searchsorted(self.t, ts, right=True) - 1
        i = i.clamp(0, self.t.numel() - 2)
        t0, t1 = self.t[i], self.t[i + 1]
        w = (ts - t0) / (t1 - t0 + 1e-12)  # scalar
        # linear blend
        B1x = (1.0 - w) * self.get_buffer("B1x")[i] + w * self.get_buffer("B1x")[i + 1]
        B1y = (1.0 - w) * self.get_buffer("B1y")[i] + w * self.get_buffer("B1y")[i + 1]
        Gz = (1.0 - w) * self.get_buffer("Gz")[i] + w * self.get_buffer("Gz")[i + 1]
        return B1x, B1y, Gz  # each scalar tensor on device


class InterpSharedZOH(torch.nn.Module):
    def __init__(self, t0, dt, rf_complex,rf_freq, gz, device="cuda"):
        super().__init__()
        self.t0 = torch.tensor(float(t0), device=device)
        self.dt = torch.tensor(float(dt), device=device)
        rf = torch.as_tensor(rf_complex, device=device)
        self.register_buffer("B1x", rf.real.to(torch.float32).contiguous())
        self.register_buffer("B1y", rf.imag.to(torch.float32).contiguous())
        self.register_buffer("B1f", rf_freq.to(torch.float32).contiguous())
        self.register_buffer("Gz",  torch.as_tensor(gz, device=device, dtype=torch.float32).contiguous())
        self.T = self.get_buffer("Gz").numel()

    def forward(self, t_scalar: torch.Tensor):
        # ZOH: pick k = floor((t-t0)/dt)
        k = torch.floor((t_scalar.to(self.t0.dtype) - self.t0) / self.dt).long()
        k = k.clamp(0, self.T-1)
        B1x = self.get_buffer("B1x")[k]
        B1y = self.get_buffer("B1y")[k]
        Gz  = self.get_buffer("Gz")[k]
        B1f = self.get_buffer("B1f")[k]
        return B1x, B1y,B1f, Gz


##------------------------- Rmix---------------- 
def get_Rmix_from_R(k_ref, z_m, R_ch):
    k0 = -float(k_ref)
    z = z_m.to(R_ch.device).reshape(-1)
    phi = (2*math.pi*k0)*z                      # (Nz,)
    d0 = torch.exp(-1j*phi).to(R_ch.dtype)      # column 0 (F+)
    d1 = torch.exp(+1j*phi).to(R_ch.dtype)      # column 1 (F-)
    R2 = R_ch.clone()
    R2[...,0,:] *= d0[None,:]
    R2[...,1,:] *= d1[None,:]
    return R2


@torch.no_grad()
def so3_exp(omega, dt, eps=1e-12):
    # omega: (...,3) rad/s
    wx, wy, wz = omega.unbind(-1)
    O = torch.zeros_like(wx)
    W = torch.stack([
        torch.stack([O, -wz,  wy], dim=-1),
        torch.stack([wz,  O, -wx], dim=-1),
        torch.stack([-wy, wx,  O], dim=-1),
    ], dim=-2)  # (...,3,3)  (this is [omega]x)

    w = torch.linalg.norm(omega, dim=-1)           # (...)
    theta = w * dt                                 # (...)

    # a = sin(theta)/w,  b = (1-cos(theta))/w^2, with small-angle limits
    w_safe = w.clamp_min(eps)
    a = torch.sin(theta) / w_safe                  # (...)
    b = (1 - torch.cos(theta)) / (w_safe*w_safe)   # (...)

    # small-angle override improves fp32 when theta ~ 0
    small = theta.abs() < 1e-4
    if small.any():
        # sin(theta)/w ~ dt - (w^2 dt^3)/6
        a = torch.where(small, dt - (w*w)*(dt**3)/6.0, a)
        # (1-cos(theta))/w^2 ~ dt^2/2 - (w^2 dt^4)/24
        b = torch.where(small, 0.5*(dt**2) - (w*w)*(dt**4)/24.0, b)

    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(W.shape)
    return I + a[...,None,None]*W + b[...,None,None]*(W @ W)

# -------------------- RHS: broadcast over bins, ICs, z --------------------
class BlochRHS(torch.nn.Module):
    """
    State M: (B, C=3, Nz, 3). bins_T in Tesla (per bin).
    RF & Gz are shared across bins and linearly interpolated in time once.
    """

    def __init__(
        self,
        control: InterpSharedLinear,
        z_m,
        b0_T,
        b1_scale=None,
        gamma_rad=2 * torch.pi * 42.577478e6,
        T1=None,
        T2=None,
        *,
        device="cuda",
        dtype=torch.float32,
    ):
        super().__init__()
        self.control = control
        self.register_buffer(
            "z", torch.as_tensor(z_m, dtype=dtype, device=device)
        )  # (Nz,)
        self.register_buffer(
            "b0_T", torch.as_tensor(b0_T, dtype=dtype, device=device)
        )  # (B,)
        self.register_buffer(
            "b1", torch.as_tensor(b1_scale, dtype=dtype, device=device)
        )
        
        self.gamma = torch.as_tensor(gamma_rad, dtype=dtype, device=device)
        self.M0 = torch.tensor(1.0, dtype=dtype, device=device)
        self.T1 = (
            None if T1 is None else torch.as_tensor(T1, dtype=dtype, device=device)
        )
        self.T2 = (
            None if T2 is None else torch.as_tensor(T2, dtype=dtype, device=device)
        )
        

    def forward(self, t, M) -> torch.Tensor:  # M: (B, C=3, Nz, 3)
        B1x, B1y,B1_freq, Gz = self.control(t)  # scalars
        z = self.get_buffer("z")  # (Nz,)
        b0_T = self.get_buffer("b0_T")  # (B,)
        B = b0_T.numel(); Nz = z.numel()
        
        b1 = self.get_buffer("b1") if hasattr(self, "b1") else torch.zeros_like(b0_T)  # (B,)
        # (B,1,1) scaling
        s = b1.view(B, 1, 1)
        Bx = (B1x * s)   # (B,1,1)
        By = (B1y * s)
        # RF detuning as extra Bz term in rotating frame

        dBz = (2.0 * math.pi * B1_freq) / self.gamma  # scalar broadcast
        
        # Bz(t,z) = Gz*z + ΔB0
        Bz = (Gz * z.view(1,1,Nz)) + b0_T.view(B,1,1) + dBz  # (B,1,Nz)

        Mx, My, Mz = M[...,0], M[...,1], M[...,2]  # (B,3,Nz)
        dMx = self.gamma * (My*Bz - Mz*By)
        dMy = self.gamma * (Mz*Bx - Mx*Bz)
        dMz = self.gamma * (Mx*By - My*Bx)

        if self.T2 is not None:
            dMx -= Mx/self.T2; dMy -= My/self.T2
        if self.T1 is not None:
            dMz -= (Mz - self.M0)/self.T1

        dM = torch.empty_like(M)
        dM[...,0], dM[...,1], dM[...,2] = dMx, dMy, dMz
        return dM


def check_unitarity_Rch(R_ch: torch.Tensor, name: str = "R_ch"):
    """
    R_ch: (B, 3_out, 3_in, Nz), complex
    Checks unitarity of each 3x3 slice over (B, Nz).
    """
    B, C_out, C_in, Nz = R_ch.shape
    assert C_out == 3 and C_in == 3, f"{name} must be (B,3,3,Nz), got {R_ch.shape}"

    # (B, 3_out, 3_in, Nz) -> (B, Nz, 3, 3) -> (B * Nz, 3, 3)
    mats = R_ch.permute(0, 3, 1, 2).reshape(-1, 3, 3)

    I = torch.eye(3, dtype=mats.dtype, device=mats.device)

    # Right-unitary: Rᴴ R ≈ I
    Rt = mats.mH                      # (BNz, 3, 3)
    RtR = Rt @ mats                   # (BNz, 3, 3)

    err = RtR - I
    err_frob = torch.linalg.norm(err, dim=(-2, -1))  # (BNz,)

    print(f"{name}: max ||RᴴR - I||_F = {err_frob.max().item():.6e}")
    print(f"{name}: mean||RᴴR - I||_F = {err_frob.mean().item():.6e}")

    # Optionally also check RRᴴ (left-unitary)
    RRt = mats @ Rt
    err2 = RRt - I
    err2_frob = torch.linalg.norm(err2, dim=(-2, -1))
    print(f"{name}: max ||RRᴴ - I||_F = {err2_frob.max().item():.6e}")
    print(f"{name}: mean||RRᴴ - I||_F = {err2_frob.mean().item():.6e}")

    # Determinant magnitude (should be ~1 for pure rotations)
    detR = torch.linalg.det(mats)
    print(f"{name} det|·|: min = {detR.abs().min().item():.6e}, "
          f"max = {detR.abs().max().item():.6e}")

    # If you want to inspect the worst offender:
    worst_idx = torch.argmax(err_frob)
    print(f"{name}: worst index (flattened B×Nz) = {worst_idx.item()}")
    print(f"{name}: RtR at worst index:\n{RtR[worst_idx]}")


@torch.no_grad()
def demod_Rxyz_to_Rmix_center(*, Rz_SB33Nz, z_m, t, gz, gamma_rad, rf_center_time=None):
    dtype, device = Rz_SB33Nz.dtype, Rz_SB33Nz.device
    t=t.to(device=device,dtype=dtype); gz=gz.to(device=device,dtype=dtype); z=z_m.to(device=device,dtype=dtype)

    gamma_hz = gamma_rad / (2*torch.pi)
    t_ref = rf_center_time if rf_center_time is not None else 0.5*(t[0]+t[-1])
    mid = torch.searchsorted(t, t_ref).clamp(1, t.numel()-1).item()

    area_pre  = torch.trapz(gz[:mid+1], t[:mid+1])
    area_post = torch.trapz(gz[mid:],  t[mid:])

    k_pre  = gamma_hz * area_pre
    k_post = gamma_hz * area_post

    phi_pre  = 2*torch.pi * k_pre  * z
    phi_post = 2*torch.pi * k_post * z

    def Qz(phi):
        c=torch.cos(phi); s=torch.sin(phi)
        Q=torch.zeros((phi.numel(),3,3), device=device, dtype=dtype)
        Q[:,0,0]=c;  Q[:,0,1]=-s
        Q[:,1,0]=s;  Q[:,1,1]= c
        Q[:,2,2]=1
        return Q

    # Inverses of physical precession (Q_phys = Qz(-phi)) are Qz(+phi)
    Qpre_inv  = Qz(-phi_pre)
    Qpost_inv = Qz(-phi_post)

    R = Rz_SB33Nz.permute(0,1,4,2,3).contiguous()      # (S, B,Nz,3,3)
    R = Qpost_inv[None, None] @ R @ Qpre_inv[None,None]  # (S,B,Nz,3,3)
    # check_demod(Rfull_BNz33=Rz_B33Nz.permute(0,3,1,2).contiguous(), Rmix_BNz33=R, phi_pre=phi_pre, phi_post=phi_post)
    return R.permute(0,1,3,4,2).contiguous()

import torch

@torch.no_grad()
def demod_Rxyz_center_epg(*, Rz_B33Nz, z_m, t, gz, gamma_rad, rf_center_time=None, to_pmz=False):
    """
    Rz_B33Nz: (B,3,3,Nz) real Cartesian rotation matrices from full finite-pulse Bloch
    z_m:      (Nz,)
    t:        (Nt,)
    gz:       (Nt,)
    gamma_rad: rad/s/T
    rf_center_time: scalar time at which the instantaneous RF event lives
    to_pmz:   if True, also convert demodulated operator to (+,-,z) basis

    Returns:
      if to_pmz=False: Rdemod_B33Nz  (real, Cartesian)
      if to_pmz=True : (Rdemod_B33Nz, Rpmz_B33Nz) with Rpmz complex
    """
    dtype  = Rz_B33Nz.dtype
    device = Rz_B33Nz.device

    t  = t.to(device=device, dtype=dtype).flatten()
    gz = gz.to(device=device, dtype=dtype).flatten()
    z  = z_m.to(device=device, dtype=dtype).flatten()

    t_ref = torch.as_tensor(
        rf_center_time if rf_center_time is not None else 0.5*(t[0] + t[-1]),
        device=device, dtype=dtype
    )

    def split_trapz_exact(t, y, t_ref):
        if t_ref <= t[0]:
            return t.new_zeros(()), torch.trapz(y, t)
        if t_ref >= t[-1]:
            return torch.trapz(y, t), t.new_zeros(())

        i = int(torch.searchsorted(t, t_ref).item())

        # exact grid hit
        if torch.isclose(t[i], t_ref):
            area_pre  = torch.trapz(y[:i+1], t[:i+1])
            area_post = torch.trapz(y[i:],   t[i:])
            return area_pre, area_post

        # interpolate y at t_ref
        t0, t1 = t[i-1], t[i]
        y0, y1 = y[i-1], y[i]
        w = (t_ref - t0) / (t1 - t0)
        y_ref = y0 + w*(y1 - y0)

        t_pre = torch.cat([t[:i], t_ref[None]])
        y_pre = torch.cat([y[:i], y_ref[None]])

        t_post = torch.cat([t_ref[None], t[i:]])
        y_post = torch.cat([y_ref[None], y[i:]])

        area_pre  = torch.trapz(y_pre,  t_pre)
        area_post = torch.trapz(y_post, t_post)
        return area_pre, area_post

    area_pre, area_post = split_trapz_exact(t, gz, t_ref)

    # phi = gamma * area * z
    phi_pre  = gamma_rad * area_pre  * z
    phi_post = gamma_rad * area_post * z

    def Qz(phi):
        c = torch.cos(phi)
        s = torch.sin(phi)
        Q = torch.zeros((phi.numel(), 3, 3), device=device, dtype=dtype)
        Q[:,0,0] = c;  Q[:,0,1] = -s
        Q[:,1,0] = s;  Q[:,1,1] =  c
        Q[:,2,2] = 1
        return Q

    # Physical gradient precession is Qz(-phi), so to REMOVE it we need Qz(+phi)
    Qpre_inv  = Qz(-phi_pre)
    Qpost_inv = Qz(-phi_post)

    R = Rz_B33Nz.permute(0,3,1,2).contiguous()   # (B,Nz,3,3)
    Rdemod = Qpost_inv[None] @ R @ Qpre_inv[None]
    Rdemod_B33Nz = Rdemod.permute(0,2,3,1).contiguous()

    if not to_pmz:
        return Rdemod_B33Nz

    # Standard MRI (+,-,z) basis: M+ = Mx + i My, M- = Mx - i My
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    U = torch.tensor([[1,  1j, 0],
                      [1, -1j, 0],
                      [0,   0, 1]], device=device, dtype=cdtype)

    Uinv = torch.tensor([[0.5,  0.5, 0],
                         [-0.5j, 0.5j, 0],
                         [0,     0,   1]], device=device, dtype=cdtype)

    Rc = Rdemod.to(cdtype)
    Rpmz = U.view(1,1,3,3) @ Rc @ Uinv.view(1,1,3,3)   # (B,Nz,3,3)
    Rpmz_B33Nz = Rpmz.permute(0,2,3,1).contiguous()

    return Rdemod_B33Nz, Rpmz_B33Nz

@torch.no_grad()
def check_demod(Rfull_BNz33, Rmix_BNz33, phi_pre, phi_post):
    def Qz(phi):
        c=torch.cos(phi); s=torch.sin(phi)
        Q=torch.zeros((phi.numel(),3,3), device=phi.device, dtype=phi.dtype)
        Q[:,0,0]=c; Q[:,0,1]=-s
        Q[:,1,0]=s; Q[:,1,1]= c
        Q[:,2,2]=1
        return Q

    Qpre_phys  = Qz(-phi_pre)   # physical (m+ gets e^{-iφ})
    Qpost_phys = Qz(-phi_post)

    Rrec = Qpost_phys[None] @ Rmix_BNz33 @ Qpre_phys[None]
    err = (Rrec - Rfull_BNz33).abs().max().item()
    print(f"Demod check: max|Rrec - Rfull| = {err:.6e}")

import torch, math

@torch.no_grad()
def fit_linear_k_from_mplus(Rmix_B33Nz, z_m):
    
    def unwrap(phi):
        d = phi[1:] - phi[:-1]
        d = (d + math.pi) % (2*math.pi) - math.pi
        return torch.cat([phi[:1], phi[:1] + torch.cumsum(d, dim=0)], dim=0)
    # use first bin if B==1, otherwise adapt as needed
    R = Rmix_B33Nz[0].permute(2,0,1).contiguous()   # (Nz,3,3)
    Mz = torch.tensor([0.,0.,1.], device=R.device, dtype=R.dtype)
    M  = torch.einsum("nij,j->ni", R, Mz)           # (Nz,3)

    mplus = (M[:,0].to(torch.complex64) + 1j*M[:,1].to(torch.complex64))/math.sqrt(2.0)
    amp = torch.abs(mplus)
    ph  = unwrap(torch.angle(mplus))

    # fit only where slice is actually excited
    mask = amp > 0.2 * amp.max()
    z = z_m[mask]
    y = ph[mask]

    A = torch.stack([z, torch.ones_like(z)], dim=1)
    sol = torch.linalg.lstsq(A, y[:,None]).solution[:,0]
    slope = sol[0]                    # rad/m
    k_fit = slope / (2*math.pi)       # cycles/m
    return float(k_fit)

@torch.no_grad()
def remove_bulk_kz_shift(Rmix_SB33Nz, z_m):
    dtype, device = Rmix_SB33Nz.dtype, Rmix_SB33Nz.device
    z = z_m.to(device=device, dtype=dtype)

    k_fit = fit_linear_k_from_mplus(Rmix_SB33Nz, z)
    phi = 2*math.pi * k_fit * z

    c = torch.cos(-phi); s = torch.sin(-phi)
    Q = torch.zeros((z.numel(),3,3), device=device, dtype=dtype)
    Q[:,0,0]=c; Q[:,0,1]=-s
    Q[:,1,0]=s; Q[:,1,1]= c
    Q[:,2,2]=1

    R = Rmix_SB33Nz.permute(0,1,4,2,3).contiguous()   # (S,B,Nz,3,3)
    R = Q[None, None] @ R
    return R.permute(0,1,2,3,4).contiguous(), k_fit

import torch

@torch.no_grad()
def spinor_step_from_omega(omega, dt, eps=1e-12):
    """
    omega: (..., 3) real, rad/s
    dt:    scalar float / tensor
    returns:
      alpha_step, beta_step : (...) complex
    Convention chosen so that spinor->Rxyz matches so3_exp convention.
    """
    wx, wy, wz = omega.unbind(-1)
    w = torch.linalg.norm(omega, dim=-1)              # (...)
    half = 0.5 * w * dt                               # (...)

    rdtype = omega.dtype
    cdtype = torch.complex64 if rdtype == torch.float32 else torch.complex128

    w_safe = w.clamp_min(eps)
    s_over_w = torch.sin(half) / w_safe               # (...)

    # small-angle stabilization:
    # sin(w dt / 2)/w ~ dt/2 - w^2 dt^3 / 48
    small = half.abs() < 1e-4
    if small.any():
        s_over_w = torch.where(
            small,
            0.5 * dt - (w * w) * (dt ** 3) / 48.0,
            s_over_w
        )

    c = torch.cos(half)

    alpha = c.to(cdtype) - 1j * wz.to(cdtype) * s_over_w.to(cdtype)
    beta  = (wy.to(cdtype) - 1j * wx.to(cdtype)) * s_over_w.to(cdtype)

    return alpha, beta

@torch.no_grad()
def spinor_compose(alpha_step, beta_step, alpha, beta):
    """
    Compose U_step @ U_old in Cayley-Klein form.
    All inputs shape-compatible complex tensors.
    """
    alpha_new = alpha_step * alpha - beta_step.conj() * beta
    beta_new  = beta_step * alpha + alpha_step.conj() * beta
    return alpha_new, beta_new


@torch.no_grad()
def spinor_to_Rxyz(alpha, beta, eps=1e-12):
    """
    alpha,beta: (...,) complex
    returns:
      Rxyz: (..., 3, 3) real
    """
    # optional renormalization before conversion
    nrm = torch.sqrt((alpha.abs()**2 + beta.abs()**2).clamp_min(eps))
    alpha = alpha / nrm
    beta  = beta  / nrm

    # Map Cayley-Klein -> quaternion
    # U = [[alpha, -beta*], [beta, alpha*]]
    # alpha = q0 - i q3
    # beta  = q2 - i q1
    q0 = alpha.real
    q1 = -beta.imag
    q2 = beta.real
    q3 = -alpha.imag

    R = torch.empty(alpha.shape + (3, 3), device=alpha.device, dtype=alpha.real.dtype)

    R[..., 0, 0] = 1 - 2 * (q2*q2 + q3*q3)
    R[..., 0, 1] = 2 * (q1*q2 - q3*q0)
    R[..., 0, 2] = 2 * (q1*q3 + q2*q0)

    R[..., 1, 0] = 2 * (q1*q2 + q3*q0)
    R[..., 1, 1] = 1 - 2 * (q1*q1 + q3*q3)
    R[..., 1, 2] = 2 * (q2*q3 - q1*q0)

    R[..., 2, 0] = 2 * (q1*q3 - q2*q0)
    R[..., 2, 1] = 2 * (q2*q3 + q1*q0)
    R[..., 2, 2] = 1 - 2 * (q1*q1 + q2*q2)

    return R

@dataclass
class Rz_specie:
    name: str
    chem_shift_Hz: float
    Rz_B33Nz: torch.Tensor  # (B,3,3,Nz) real rotation matrices
    
    def to_dict(self):
        return {
            "name": self.name,
            "chem_shift_Hz": self.chem_shift_Hz,
            "Rz_B33Nz": self.Rz_B33Nz,  # move to CPU for serialization
        }
@dataclass
class Species_Solve:
    Rz_solve: List[Rz_specie] = field(default_factory=list)  # list of species with their Rz_B33Nz

    def append_solve(self, name: str, chem_shift_Hz: float, Rz_B33Nz: torch.Tensor):
        self.Rz_solve.append(Rz_specie(name, chem_shift_Hz, Rz_B33Nz))  # move to CPU for storage

# Simple mapping for chemical shift names to values
CHEM_SHIFT_HZ = {
     0: "water",
    -412: "fat",
}

def get_sp_name(chem_shift_Hz: float) -> str:
    """Return species name for a given chemical shift in Hz."""
    return CHEM_SHIFT_HZ.get(int(chem_shift_Hz), "unknown")
        
@torch.no_grad()
def solve_slice(
    *,
    rf_complex,
    rf_freq,
    gz,
    t_grid,
    z_m,
    b0_T,
    b1_scale=None,
    gamma_rad=2 * torch.pi * 42.577478e6,
    device="cuda",
    dtype_policy= DTYPE,
    T1=None,
    T2=None,
    plot_slice_profile=False,
    return_debug=False,
    backend:str = "so3", # "spinor" or "so3",
    rf_center_time: float | None = None,  # if not None, also return Rmix centered at this time
    chem_shift_Hz: torch.Tensor = torch.tensor([0.0, 0.0]),  # if nonzero, also apply this off-resonance shift to the RF frequency before solving
    
) -> Species_Solve|Tuple[Species_Solve, dict]:
    """
    Solve one slice-selective RF pulse over:
    - species axis S (e.g. water/fat),
    - B0/B1 bin axis B,
    - z axis Nz.

    Returns
    -------
    Rtot_mix : torch.Tensor
        Shape (S*B, 3, 3, Nz), flattened over species first, then B bins,
        for compatibility with the existing downstream code.

    Notes
    -----
    - `chem_shift_Hz` is a discrete species axis, not an interpolation axis.
    - Output flattening order is:
        flat_index = species_index * B + b0b1_index
    """
    dtype = torch.float64 if dtype_policy == torch.float64 else torch.float32
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    # ------------------------------------------------------------------
    # Core tensors
    # ------------------------------------------------------------------
    t = torch.as_tensor(t_grid, dtype=dtype, device=device).reshape(-1)
    z_m = torch.as_tensor(z_m, dtype=dtype, device=device).reshape(-1)
    b0_T = torch.as_tensor(b0_T, dtype=dtype, device=device).reshape(-1)

    B = b0_T.numel()
    Nz = z_m.numel()

    if b1_scale is None:
        b1_scale = torch.ones(B, device=device, dtype=dtype)
    else:
        b1_scale = torch.as_tensor(b1_scale, device=device, dtype=dtype).reshape(-1)
        if b1_scale.numel() != B:
            raise ValueError(f"b1_scale must have length B={B}, got {b1_scale.numel()}")

    # species axis
    chem_shift_Hz = torch.as_tensor(chem_shift_Hz, device=device, dtype=dtype).reshape(-1)
    S = chem_shift_Hz.numel()

    rf_complex = torch.as_tensor(rf_complex, device=device, dtype=cdtype)
    rf_freq = torch.as_tensor(rf_freq, device=device, dtype=dtype)
    gz = torch.as_tensor(gz, device=device, dtype=dtype)

    if t.numel() < 2:
        raise ValueError("t_grid must contain at least 2 samples")
    dt0 = t[1] - t[0]

    # assumes uniform RF raster as before
    ctrl = InterpSharedZOH(t[0], dt0, rf_complex, rf_freq, gz, device=device)

    # ------------------------------------------------------------------
    # Broadcasted geometry
    # ------------------------------------------------------------------
    # Shapes:
    #   species_Bz : (S,1,1)
    #   z          : (1,1,Nz)
    #   b0         : (1,B,1)
    #   s          : (1,B,1)
    #
    # chem_shift_Hz is converted to Tesla-equivalent detuning.
    # Sign convention here:
    #   Bz_eff = Gz*z + local_B0 - RF_center + species_shift
    # so that for a pulse centered at -412 Hz:
    #   water (0 Hz) sees ~ +412 Hz detuning
    #   fat   (-412 Hz) sees ~ 0 Hz detuning
    species_Bz = ((2.0 * torch.pi * chem_shift_Hz) / gamma_rad).view(S, 1, 1)  # Tesla
    z = z_m.view(1, 1, Nz)
    b0 = b0_T.view(1, B, 1)
    s = b1_scale.view(1, B, 1)

    # ------------------------------------------------------------------
    # Backend: SO(3)
    # ------------------------------------------------------------------
    if backend == "so3":
        # R_all: (S, B, Nz, 3, 3)
        I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
        R_all = I.expand(S, B, Nz, 3, 3).clone()

        for ti in range(t.numel() - 1):
            dt = float((t[ti + 1] - t[ti]).item())
            B1x, B1y, B1_freq, Gz = ctrl(t[ti])  # scalars

            Bx = (B1x * s).expand(S, B, Nz)
            By = (B1y * s).expand(S, B, Nz)
        
            # dBz = (2.0 * torch.pi * B1_freq) / gamma_rad  # Tesla   dBz baked in Bx and By already 
            Bz = (Gz * (z) + b0 + species_Bz).expand(S, B, Nz)

            omega = gamma_rad * torch.stack([Bx, By, Bz], dim=-1)  # (S,B,Nz,3)
            Rstep = so3_exp(omega, dt)                              # (S,B,Nz,3,3)
            R_all = Rstep @ R_all

    # ------------------------------------------------------------------
    # Backend: spinor
    # ------------------------------------------------------------------
    elif backend == "spinor":
        renorm_every = 32
        eps = 1e-12

        alpha = torch.ones((S, B, Nz), device=device, dtype=cdtype)
        beta = torch.zeros((S, B, Nz), device=device, dtype=cdtype)

        for ti in range(t.numel() - 1):
            dt = float((t[ti + 1] - t[ti]).item())
            B1x, B1y, B1_freq, Gz = ctrl(t[ti])  # scalars

            Bx = (B1x * s).expand(S, B, Nz)
            By = (B1y * s).expand(S, B, Nz)

            # dBz = (2.0 * torch.pi * B1_freq) / gamma_rad  # Tesla dBz baked in Bx and By already 
            Bz = (Gz * z + b0  + species_Bz).expand(S, B, Nz)

            omega = gamma_rad * torch.stack([Bx, By, Bz], dim=-1)   # (S,B,Nz,3)

            alpha_step, beta_step = spinor_step_from_omega(omega, dt, eps=eps)
            alpha, beta = spinor_compose(alpha_step, beta_step, alpha, beta)

            if renorm_every and ((ti + 1) % renorm_every == 0):
                nrm = torch.sqrt((alpha.abs() ** 2 + beta.abs() ** 2).clamp_min(eps))
                alpha = alpha / nrm
                beta = beta / nrm

        nrm = torch.sqrt((alpha.abs() ** 2 + beta.abs() ** 2).clamp_min(eps))
        alpha = alpha / nrm
        beta = beta / nrm

        R_all = spinor_to_Rxyz(alpha, beta, eps=eps)   # (S,B,Nz,3,3)

    else:
        raise ValueError(f"Unknown backend {backend}, expected 'so3' or 'spinor'")
    # flatten back to (B,3,3,Nz) for consistency with rest of code
    # [ Note: the internal shape (B,Nz,3,3) is more natural for the computation, but we permute back to (B,3,3,Nz) for compatibility with the rest of the codebase that expects that shape.]
    R_all = R_all.permute(0,1,3,4,2).contiguous()  # (chem_shift,B,3,3,Nz)
    
    Rtot_mix = demod_Rxyz_to_Rmix_center(Rz_SB33Nz=R_all, z_m=z_m, t=t, gz=gz, gamma_rad=gamma_rad,)  # (B,3,3,Nz) in mix frame, still with bulk kz shift
    specieSolve = Species_Solve()
    for i, chem_shift in enumerate(chem_shift_Hz):
        specieSolve.append_solve(
            name= get_sp_name(chem_shift.item()),
            chem_shift_Hz=chem_shift.item(),
            Rz_B33Nz=Rtot_mix[i]
        )

    if return_debug:
        dbg = dict(
        t=t.detach(), z=z_m.detach(),
        rf=rf_complex.detach(), gz=torch.as_tensor(gz, device=t.device).detach(),
        Rz=R_all.detach(),            # (B,3,3,Nz) centered
        Rz_mix =Rtot_mix.detach(),      # (B,3,3,Nz) demodulated to mix frame
        )
        return specieSolve, dbg
    
    return  specieSolve  # (B,3,3,Nz) demodulated to mix frame
