"""
This Bloch simulation is meant as a ground truth to compare PDG against.

Note that this still uses box shaped voxels, which is generally discouraged.

.. code-block:: python

    data = SimData.load(...)
    seq = Sequence()  # Some definition as used by PDG

    spin_count = 123  # R2 - distributed spins, doesn't need to be a square nr.
    signal = simulate(seq, data, spin_count)

    # Signal has same format as returned by execute_graph(...),
    # no further code changes are needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy import pi

# Absolute project root: directory containing this script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SLICE_THICKNESS = 0.005

from src.voxelgridphantom.voxel_grid_phantom import VoxelGridPhantom
from sequence.sequence import Sequence , PulseUsage


def atomic_torch_save(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def make_checkpoint_name(seq_name, nx, ny, nz, slice_idx, spin_count):
    safe_seq = str(seq_name).replace("/", "_")
    return f"{safe_seq}_nx{nx}_ny{ny}_nz{nz}_slice{slice_idx}_spins{spin_count}.pt"

def isochromat_sim(seq: Sequence, data, spin_count: int,
                   perfect_spoiling=False,
                   print_progress: bool = True,
                   spin_dist: Literal["r2", "rand", "z"] = "z",
                   r2_seed = None,
                     resume=False, checkpoint_path=None
                   ) -> torch.Tensor:
    """Simulate ``seq`` on ``data`` with ``spin_count`` spins per voxel.

    The intra-voxel spin distribution is randomized, except if
    `spin_dist = "r2"` and a fixed `r2_seed` are chosen. For a deterministic
    distribution of spins, call `torch.manual_seed()` before this function.

    Parameters
    ----------
    seq: Sequence
        The sequence that will be simulated
    data: SimData
        Simulation data that defines everything else
    spin_count: int
        Number of spins used for simulation
    perfect_spoiling: bool
        If ``True``, the transversal magnetization is set to zero on excitation
    print_progress: bool
        If ``True``, the currently simulated repetition is printed
    spin_dist: "r2" | "rand"
        Use either a golden-ratio pseudo-random blue-noise like or
        a white-noise like intra-voxel distribution of spins
    r2_seed: None | torch.Tensor
        The seed and position of the first spin for the blue-noise like spin
        distribution. If ``None``, a random position is chosen. Expects a
        tensor with 3 floats in the range of ``[0, 1]``
        
    Returns
    -------
    torch.Tensor
        Complex tensor with shape (sample_count, coil_count)
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    start_rep = 0
    R = len(list(seq))
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        start_rep = int(ckpt["next_rep"])
        spins = ckpt["spins"].to(device=data.device)
        spin_pos = ckpt["spin_pos"].to(device=data.device)
        spin_offset_m = ckpt["spin_offset_m"].to(device=data.device)
        omega = ckpt["omega"].to(device=data.device)
        signal = [s.to(device=data.device) for s in ckpt["signal"]]

        print(f"Resuming from repetition {start_rep + 1}/{R}", flush=True)

    else:

    
    
        # RawSimData doesn't store the voxel shape. We could extract it by FFT'ing
        # the dephasing function, if that feature is desired. Until then, we just
        # use the nyquist frequencies, with a special case for ∞ (custom phantoms).
        voxel_size = 0.5 / torch.tensor(data.nyquist, device=data.device)
        if not torch.isfinite(voxel_size).all():
            # Fallback voxel size
            voxel_size = torch.tensor([0.1, 0.1, 0.1], device=data.device)

        if spin_dist == "rand":
            spin_pos = torch.rand(spin_count, 3, device=data.device)
        elif spin_dist == "r2":
            if r2_seed is None:
                r2_seed = torch.rand(3, device=data.device)

            # 3 dimensional R2 sequence for intravoxel spin distribution
            g = 1.22074408460575947536  # 3D
            # g = 1.32471795724474602596  # 2D
            a = 1.0 / torch.tensor([g**1, g**2, g**3], device=data.device)
            indices = torch.arange(spin_count, device=data.device)
            spin_pos = torch.stack([
                (r2_seed[0] + a[0] * indices) % 1,
                (r2_seed[1] + a[1] * indices) % 1,
                (r2_seed[2] + a[2] * indices) % 1
            ]).T
        else:
            # raise ValueError("unexpected spin_dist", spin_dist)
            # deterministic midpoint samples through the voxel only along z
            # x/y remain at voxel center
            indices = torch.arange(spin_count, device=data.device, dtype=torch.float32)

            spin_pos = torch.full((spin_count, 3), 0.5, device=data.device)
            spin_pos[:, 2] = (indices + 0.5) / spin_count

        # spin_pos = torch.rand_like(spin_pos)  # Use white noise
        spin_pos = 2 * pi * (spin_pos - 0.5) * voxel_size.unsqueeze(0)

        # Omega is a cauchy-distributed tensor of spin offset freqencies (for T2')
        off_res = torch.linspace(-0.5, 0.5, spin_count, device=data.device)
        omega = torch.tan(pi * 0.999 * off_res)  # Cut off high frequencies
        omega = omega[torch.randperm(spin_count)]

        # Start off with relaxed magnetisation, stored as (voxels x spins)
        spins = torch.zeros((data.PD.numel(), spin_count, 3),
                            device=data.device)
        spins[:, :, 2] = 1
        spin_offset_m = spin_pos / (2 * torch.pi)
        # Simulation:
        signal = []
        Mag_hist = []
        
    # Combine coil sensitivities and proton density to a voxels x coils tensor
    coil_sensitivity = (
        data.coil_sens.t().to(torch.cfloat)
        * data.PD.unsqueeze(1) / spin_count
    )
    coil_count = data.coil_sens.shape[0]

    
    bloch_span_ids = [si for si, s in enumerate(seq.mrtini_hybrid) if s["kind"] == "bloch"]

    assert len(bloch_span_ids) >= R, f"Not enough bloch spans: {len(bloch_span_ids)} for {R} reps"
    for r, rep in enumerate(seq):
        if r < start_rep:
            continue
        
        print(f"Simulating repetition {r+1}/{R}...", end="\r" if print_progress else "\n", flush=True)
        span_id = bloch_span_ids[r]
        span = seq.mrtini_hybrid[span_id]

        if perfect_spoiling and rep.pulse.usage == PulseUsage.EXCIT:
            spins[:, :, :2] = 0
        spins = apply_bloch_pulse_rep(spins, span, data, rep,
                                      spin_offset_m=spin_offset_m,
                                      include_relaxation=False)
        
        rep_sig = torch.zeros((rep.event_count, coil_count),
                              dtype=torch.cfloat, device=data.device)
        signal.append(rep_sig)

        
        spins_start = spins
        time = rep.event_time.cumsum(0).to(device=data.device)
        gradm = rep.gradm.cumsum(0).to(device=data.device)
       
        for e in range(rep.event_count):
            spins = spins_start.clone()
            spins = relax(spins, data.T1, data.T2, time[e])
            spins = dephase(spins, omega, data.T2dash, time[e])
            spins = intravoxel_precess(spins, gradm[e], spin_pos)

            spins = B0_precess(spins, data.B0, time[e])
            spins = grad_precess(spins, gradm[e], data.voxel_pos)

            if rep.adc_usage[e] > 0:
                adc_rot = torch.exp(1j * rep.adc_phase[e])
                rep_sig[e, :] = measure(spins, coil_sensitivity) * adc_rot

        if checkpoint_path is not None:
            torch.cuda.synchronize()

            ckpt = {
                "next_rep": r + 1,
                "total_reps": R,
                "spin_count": spin_count,
                "spins": spins.detach().cpu(),
                "spin_pos": spin_pos.detach().cpu(),
                "spin_offset_m": spin_offset_m.detach().cpu(),
                "omega": omega.detach().cpu(),
                "signal": [s.detach().cpu() for s in signal],
            }

            print(f"\nSaving checkpoint after rep {r+1}/{R}: {checkpoint_path}", flush=True)
            atomic_torch_save(ckpt, checkpoint_path)
            print("Checkpoint saved.", flush=True)
            
    if print_progress:
        print(" - done", flush=True)
       
    return torch.cat([
        sig[rep.adc_usage > 0, :] for sig, rep in zip(signal, seq)
    ])

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
        phi = phi.squeeze(-1)  # (Nz,1)
        c=torch.cos(phi); s=torch.sin(phi)
        
        Q=torch.zeros((*phi.shape,3,3), device=device, dtype=dtype)
        Q[...,0,0]=c;  Q[...,0,1]=-s
        Q[...,1,0]=s;  Q[...,1,1]= c
        Q[...,2,2]=1
        return Q

    # Inverses of physical precession (Q_phys = Qz(-phi)) are Qz(+phi)
    Qpre_inv  = Qz(-phi_pre)
    Qpost_inv = Qz(-phi_post)

    R = Rz_SB33Nz    # (S, B,Nz,3,3)
    R = Qpost_inv @ R @ Qpre_inv  # (S,B,Nz,3,3)
    # check_demod(Rfull_BNz33=Rz_B33Nz.permute(0,3,1,2).contiguous(), Rmix_BNz33=R, phi_pre=phi_pre, phi_post=phi_post)
    return R


# @torch.no_grad()
# def so3_exp(omega, dt, eps=1e-12):
#     # omega: (...,3) rad/s
#     wx, wy, wz = omega.unbind(-1)
#     O = torch.zeros_like(wx)
#     W = torch.stack([
#         torch.stack([O, -wz,  wy], dim=-1),
#         torch.stack([wz,  O, -wx], dim=-1),
#         torch.stack([-wy, wx,  O], dim=-1),
#     ], dim=-2)  # (...,3,3)  (this is [omega]x)

#     w = torch.linalg.norm(omega, dim=-1)           # (...)
#     theta = w * dt                                 # (...)

#     # a = sin(theta)/w,  b = (1-cos(theta))/w^2, with small-angle limits
#     w_safe = w.clamp_min(eps)
#     a = torch.sin(theta) / w_safe                  # (...)
#     b = (1 - torch.cos(theta)) / (w_safe*w_safe)   # (...)

#     # small-angle override improves fp32 when theta ~ 0
#     small = theta.abs() < 1e-4
#     if small.any():
#         # sin(theta)/w ~ dt - (w^2 dt^3)/6
#         a = torch.where(small, dt - (w*w)*(dt**3)/6.0, a)
#         # (1-cos(theta))/w^2 ~ dt^2/2 - (w^2 dt^4)/24
#         b = torch.where(small, 0.5*(dt**2) - (w*w)*(dt**4)/24.0, b)

#     I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(W.shape)
#     return I + a[...,None,None]*W + b[...,None,None]*(W @ W)


@torch.no_grad()
def so3_exp(omega, dt, eps=1e-12):
    # omega: (..., 3), rad/s
    wx, wy, wz = omega.unbind(-1)

    w2 = wx*wx + wy*wy + wz*wz
    w = torch.sqrt(w2)
    theta = w * dt

    w_safe = w.clamp_min(eps)
    a = torch.sin(theta) / w_safe
    b = (1.0 - torch.cos(theta)) / (w_safe*w_safe)

    small = theta.abs() < 1e-4
    a_small = dt - w2*(dt**3)/6.0
    b_small = 0.5*(dt**2) - w2*(dt**4)/24.0
    a = torch.where(small, a_small, a)
    b = torch.where(small, b_small, b)

    R = omega.new_empty((*omega.shape[:-1], 3, 3))

    R[..., 0, 0] = 1.0 - b*(wy*wy + wz*wz)
    R[..., 0, 1] = b*wx*wy - a*wz
    R[..., 0, 2] = b*wx*wz + a*wy

    R[..., 1, 0] = b*wy*wx + a*wz
    R[..., 1, 1] = 1.0 - b*(wx*wx + wz*wz)
    R[..., 1, 2] = b*wy*wz - a*wx

    R[..., 2, 0] = b*wz*wx - a*wy
    R[..., 2, 1] = b*wz*wy + a*wx
    R[..., 2, 2] = 1.0 - b*(wx*wx + wy*wy)

    return R

def measure(spins: torch.Tensor, coil_sensitivity: torch.Tensor
            ) -> torch.Tensor:
    """Calculate the measured signal per coil.

    The returned tensor is 1D with ``coil_count`` elements.
    """
    voxel_mag = spins[:, :, 0].sum(1) + 1j*spins[:, :, 1].sum(1)
    # (voxels), (voxels x coils)
    return voxel_mag @ coil_sensitivity


def relax(spins: torch.Tensor, T1: torch.Tensor, T2: torch.Tensor, dt: float
          ) -> torch.Tensor:
    """Relax xy magnetisation with T1 towards 0 and z with T2 towards 1."""
    relaxed = torch.empty_like(spins)
    r1 = torch.exp(-dt / T1).view(-1, 1)
    relaxed[:, :, 2] = spins[:, :, 2] * r1 + (1 - r1)
    relaxed[:, :, :2] = spins[:, :, :2] * torch.exp(-dt / T2).view(-1, 1, 1)

    return relaxed


def dephase(spins: torch.Tensor, omega: torch.Tensor,
            T2dash: torch.Tensor, dt: float) -> torch.Tensor:
    """T2' - dephase spins, the per-voxel amount is given by T2dash and dt."""
    # shape: voxels x spins
    angle = ((1 / T2dash).unsqueeze(1) @ omega.unsqueeze(0)) * dt
    rot_mat = torch.zeros((*angle.shape, 3, 3), device=spins.device)
    rot_mat[:, :, 0, 0] = torch.cos(angle)
    rot_mat[:, :, 0, 1] = -torch.sin(angle)
    rot_mat[:, :, 1, 0] = torch.sin(angle)
    rot_mat[:, :, 1, 1] = torch.cos(angle)
    rot_mat[:, :, 2, 2] = 1
    # (voxels x spins x 3 x 3), (voxels x spins x 3)
    return torch.einsum("vsij, vsj -> vsi", rot_mat, spins)


@torch.no_grad()
def apply_bloch_pulse_rep_chunk(
    spins, rw, data, rep,
    spin_offset_m,
    *,
    gamma_hz=42.577478e6,
    gz_units="T_per_m",
    include_relaxation=False,
    spin_chunk=4000,   # try 512, 1024, 2048
):
    if rw is None:
        return flip(spins, rep.pulse.angle, rep.pulse.phase, data.B1)

    dev, dtype = spins.device, spins.dtype
    gamma_rad = 2 * torch.pi * gamma_hz
    GAMMA_HZ = 42.577478e6

    if spin_offset_m is None:
        spin_offset_m = torch.zeros(spins.shape[1], 3, device=dev, dtype=dtype)
    else:
        spin_offset_m = spin_offset_m.to(device=dev, dtype=dtype)

    t = torch.as_tensor(rw["t"], device=dev, dtype=dtype)

    rf = torch.as_tensor(rw["rf_amp"], device=dev, dtype=torch.cfloat) / GAMMA_HZ
    gz = torch.as_tensor(rw["gz"], device=dev, dtype=dtype) / GAMMA_HZ

    rf_phase = rw.get("rf_phase", None)
    if rf_phase is None:
        rf_phase = torch.zeros_like(t)
    else:
        rf_phase = torch.as_tensor(rf_phase, device=dev, dtype=dtype)

    rf_complex = rf.to(torch.cfloat) * torch.exp(1j * rf_phase)

    rf_freq = rw.get("rf_freq", None)
    if rf_freq is None:
        rf_freq = torch.zeros_like(t)
    else:
        rf_freq = torch.as_tensor(rf_freq, device=dev, dtype=dtype)

    V, S = spins.shape[:2]

    # B1 map: (V, 1)
    B1map = data.B1
    if B1map.ndim > 1:
        B1map = B1map.sum(0)
    B1map = B1map.to(device=dev)
    if not torch.is_complex(B1map):
        B1map = B1map.to(dtype).to(torch.cfloat)
    B1map = B1map.reshape(V, 1)

    # B0 map: (V, 1)
    B0_hz = data.B0.to(device=dev, dtype=dtype).reshape(V, 1)

    # voxel z position: (V, 1)
    voxel_z = data.voxel_pos[:, 2].to(device=dev, dtype=dtype).reshape(V, 1)

    eye = torch.eye(3, device=dev, dtype=dtype).view(1, 1, 3, 3)

    for s0 in range(0, S, spin_chunk):
        s1 = min(s0 + spin_chunk, S)
        Sc = s1 - s0

        # chunked absolute z-position: (V, Sc)
        z_abs = voxel_z + spin_offset_m[s0:s1, 2].reshape(1, Sc)

        # chunked accumulated rotation: (V, Sc, 3, 3)
        Rtot = eye.expand(V, Sc, 3, 3).clone()

        for n in range(t.numel() - 1):
            dt = float((t[n + 1] - t[n]).item())
            if dt <= 0:
                continue

            B1eff = B1map * rf_complex[n]

            # Keep wx/wy as (V,1). Let broadcasting handle spin dimension.
            omega_x = gamma_rad * B1eff.real
            omega_y = gamma_rad * B1eff.imag

            if gz_units == "cycles_per_m_per_s":
                omega_z = 2 * torch.pi * (B0_hz + rf_freq[n] + gz[n] * z_abs)

            elif gz_units == "T_per_m":
                omega_z = 2 * torch.pi * (B0_hz + rf_freq[n]) + gamma_rad * gz[n] * z_abs

            else:
                raise ValueError(f"unknown gz_units={gz_units}")

            Rtot = left_compose_so3_components(
                Rtot, omega_x, omega_y, omega_z, dt
            )

        R = demod_Rxyz_to_Rmix_center(
            Rz_SB33Nz=Rtot,
            z_m=z_abs,
            t=t,
            gz=gz,
            gamma_rad=gamma_rad,
        )

        spins[:, s0:s1, :] = torch.einsum(
            "vsij,vsj->vsi",
            R,
            spins[:, s0:s1, :]
        )

        del z_abs, Rtot, R

    return spins

@torch.no_grad()
def apply_bloch_pulse_rep(
    spins, rw, data,rep,
    spin_offset_m,
    *,
    gamma_hz=42.577478e6,
    gz_units="T_per_m",  # or "T_per_m"
    include_relaxation=False,
):
    if rw is None:
        return flip(spins, rep.pulse.angle, rep.pulse.phase, data.B1)

    if spin_offset_m is None:
        spin_offset_m = torch.zeros(3, device=spins.device)
    dev, dtype = spins.device, spins.dtype
    gamma_rad = 2 * torch.pi * gamma_hz
    GAMMA_HZ= 42.577478e6  # Hz/T ~ 267.5e6 rad/s/T
    t = torch.as_tensor(rw["t"], device=dev, dtype=dtype)
    rf = torch.as_tensor(rw["rf_amp"], device=dev, dtype=torch.cfloat)/GAMMA_HZ
    gz = torch.as_tensor(rw["gz"], device=dev, dtype=dtype)/GAMMA_HZ
    rf_phase = rw.get("rf_phase", None)
    if rf_phase is None:
        rf_phase = torch.zeros_like(t)
    else:
        rf_phase = torch.as_tensor(rf_phase, device=dev, dtype=dtype)

    rf_complex = rf.to(torch.cfloat) * torch.exp(1j * rf_phase)
    rf_freq = rw.get("rf_freq", None)
    if rf_freq is None:
        rf_freq = torch.zeros_like(t)
    else:
        rf_freq = torch.as_tensor(rf_freq, device=dev, dtype=dtype)

    V, S = spins.shape[:2]

    # B1 map
    B1map = data.B1
    if B1map.ndim > 1:
        B1map = B1map.sum(0)
    B1map = B1map.to(device=dev)
    if not torch.is_complex(B1map):
        B1map = B1map.to(dtype).to(torch.cfloat)
    B1map = B1map.reshape(V, 1)

    B0_hz = data.B0.to(device=dev, dtype=dtype).reshape(V, 1)
    
    # absolute z-position of every isochromat
    z_abs = data.voxel_pos[:, 2].to(device=dev, dtype=dtype).reshape(V, 1)\
            + spin_offset_m[:, 2].to(device=dev, dtype=dtype).reshape(1, S)
    Rtot = torch.eye(3, device=spins.device, dtype=spins.dtype)
    Rtot = Rtot.view(1, 1, 3, 3).expand(V, S, 3, 3).clone()
    # for n in range(t.numel() - 1):
    #     dt = float((t[n + 1] - t[n]).item())
    #     if dt <= 0:
    #         continue

    #     b1 = rf_complex[n]
    #     B1eff = B1map * b1

    #     Bx = B1eff.real
    #     By = B1eff.imag

    #     omega_x = gamma_rad * Bx.expand(V, S)
    #     omega_y = gamma_rad * By.expand(V, S)

    #     if gz_units == "cycles_per_m_per_s":
    #         omega_z = 2 * torch.pi * (B0_hz + rf_freq[n] + gz[n] * z_abs)
    #         omega_x = gamma_rad * Bx.expand(V, S)
    #         omega_y = gamma_rad * By.expand(V, S)
    #     elif gz_units == "T_per_m":
    #         omega_x = gamma_rad * Bx.expand(V, S)
    #         omega_y = gamma_rad * By.expand(V, S)
    #         omega_z = 2 * torch.pi * (B0_hz.expand(V, S) + rf_freq[n]) + gamma_rad * gz[n] * z_abs
    #     else:
    #         raise ValueError(f"unknown gz_units={gz_units}")

    #     omega = torch.stack([omega_x, omega_y, omega_z], dim=-1)  # (V,S,3)
    #     Rstep = so3_exp(omega, dt)
    #     Rtot = Rstep @ Rtot
    
    for n in range(t.numel() - 1):
        dt = float((t[n + 1] - t[n]).item())
        if dt <= 0:
            continue

        b1 = rf_complex[n]
        B1eff = B1map * b1

        Bx = B1eff.real
        By = B1eff.imag

        omega_x = gamma_rad * Bx.expand(V, S)
        omega_y = gamma_rad * By.expand(V, S)

        if gz_units == "cycles_per_m_per_s":
            omega_z = 2 * torch.pi * (B0_hz + rf_freq[n] + gz[n] * z_abs)

        elif gz_units == "T_per_m":
            omega_z = 2 * torch.pi * (B0_hz.expand(V, S) + rf_freq[n]) + gamma_rad * gz[n] * z_abs

        else:
            raise ValueError(f"unknown gz_units={gz_units}")

        Rtot = left_compose_so3_components(Rtot, omega_x, omega_y, omega_z, dt)

        del omega_x, omega_y, omega_z

        # if include_relaxation:
            # spins = relax(spins, data.T1, data.T2, dt)
    R = demod_Rxyz_to_Rmix_center(Rz_SB33Nz=Rtot, z_m=z_abs, t=t, gz=gz, gamma_rad=gamma_rad)
    spins = torch.einsum("vsij,vsj->vsi", R, spins)
    ##N delete R to save memoty:
    del R
    return spins

@torch.no_grad()
def left_compose_so3_components(R, wx, wy, wz, dt, eps=1e-12):
    """
    Computes R_new = exp([w]x dt) @ R
    without forming the step rotation matrix.

    R:  (..., 3, 3)
    wx, wy, wz: broadcastable to R.shape[:-2]
    """

    w2 = wx*wx + wy*wy + wz*wz
    w = torch.sqrt(w2)
    theta = w * dt

    w_safe = w.clamp_min(eps)
    a = torch.sin(theta) / w_safe
    b = (1.0 - torch.cos(theta)) / (w_safe*w_safe)

    small = theta.abs() < 1e-4
    a = torch.where(small, dt - w2*(dt**3)/6.0, a)
    b = torch.where(small, 0.5*(dt**2) - w2*(dt**4)/24.0, b)

    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # omega x each column of R
    c00 = wy*r20 - wz*r10
    c01 = wy*r21 - wz*r11
    c02 = wy*r22 - wz*r12

    c10 = wz*r00 - wx*r20
    c11 = wz*r01 - wx*r21
    c12 = wz*r02 - wx*r22

    c20 = wx*r10 - wy*r00
    c21 = wx*r11 - wy*r01
    c22 = wx*r12 - wy*r02

    # omega dot each column
    d0 = wx*r00 + wy*r10 + wz*r20
    d1 = wx*r01 + wy*r11 + wz*r21
    d2 = wx*r02 + wy*r12 + wz*r22

    Rout = torch.empty_like(R)

    Rout[..., 0, 0] = r00 + a*c00 + b*(wx*d0 - w2*r00)
    Rout[..., 0, 1] = r01 + a*c01 + b*(wx*d1 - w2*r01)
    Rout[..., 0, 2] = r02 + a*c02 + b*(wx*d2 - w2*r02)

    Rout[..., 1, 0] = r10 + a*c10 + b*(wy*d0 - w2*r10)
    Rout[..., 1, 1] = r11 + a*c11 + b*(wy*d1 - w2*r11)
    Rout[..., 1, 2] = r12 + a*c12 + b*(wy*d2 - w2*r12)

    Rout[..., 2, 0] = r20 + a*c20 + b*(wz*d0 - w2*r20)
    Rout[..., 2, 1] = r21 + a*c21 + b*(wz*d1 - w2*r21)
    Rout[..., 2, 2] = r22 + a*c22 + b*(wz*d2 - w2*r22)

    return Rout

def flip(spins: torch.Tensor, angle: torch.Tensor, phase: torch.Tensor,
         B1: torch.Tensor, device="cuda", dtype=torch.float32) -> torch.Tensor:
    """Rotate the magnetisation to simulate a RF-pulse."""
    a = angle * B1
    p = torch.as_tensor(phase)
    # Rz(phase) * Rx(angle) * Rz(-phase):
    rot_mat = torch.zeros((B1.numel(), 3, 3), device=spins.device)
    rot_mat[:, 0, 0] = torch.sin(p)**2*torch.cos(a) + torch.cos(p)**2
    rot_mat[:, 0, 1] = (1 - torch.cos(a))*torch.sin(p)*torch.cos(p)
    rot_mat[:, 0, 2] = torch.sin(a)*torch.sin(p)
    rot_mat[:, 1, 0] = (1 - torch.cos(a))*torch.sin(p)*torch.cos(p)
    rot_mat[:, 1, 1] = torch.sin(p)**2 + torch.cos(a)*torch.cos(p)**2
    rot_mat[:, 1, 2] = -torch.sin(a)*torch.cos(p)
    rot_mat[:, 2, 0] = -torch.sin(a)*torch.sin(p)
    rot_mat[:, 2, 1] = torch.sin(a)*torch.cos(p)
    rot_mat[:, 2, 2] = torch.cos(a)

    # (voxels x 3 x 3), (voxels x spins x 3)
    return torch.einsum("vij, vsj -> vsi", rot_mat, spins)


def grad_precess(spins: torch.Tensor, gradm: torch.Tensor,
                 voxel_pos: torch.Tensor) -> torch.Tensor:
    """Rotate individual voxels as given by their position and ```gradm``."""
    angle = 2 * pi * voxel_pos @ gradm  # shape: voxels
    rot_mat = torch.zeros((angle.numel(), 3, 3), device=spins.device)
    rot_mat[:, 0, 0] = torch.cos(angle)
    rot_mat[:, 0, 1] = -torch.sin(angle)
    rot_mat[:, 1, 0] = torch.sin(angle)
    rot_mat[:, 1, 1] = torch.cos(angle)
    rot_mat[:, 2, 2] = 1
    # (voxels x 3 x 3), (voxels x spins x 3)
    return torch.einsum("vij, vsj -> vsi", rot_mat, spins)


def B0_precess(spins: torch.Tensor, B0: torch.Tensor,
               dt: float) -> torch.Tensor:
    """Rotate voxels as given by ``B0`` and the elapsed time ``dt``."""
    angle = 2 * pi * B0 * dt  # shape: voxels
    rot_mat = torch.zeros((angle.numel(), 3, 3), device=spins.device)
    rot_mat[:, 0, 0] = torch.cos(angle)
    rot_mat[:, 0, 1] = -torch.sin(angle)
    rot_mat[:, 1, 0] = torch.sin(angle)
    rot_mat[:, 1, 1] = torch.cos(angle)
    rot_mat[:, 2, 2] = 1
    # (voxels x 3 x 3), (voxels x spins x 3)
    return torch.einsum("vij, vsj -> vsi", rot_mat, spins)


def intravoxel_precess(spins: torch.Tensor, gradm: torch.Tensor,
                       spin_pos: torch.Tensor) -> torch.Tensor:
    """Rotate spins inside of each voxel to simulate the dephasing.

    ``grad_precess`` and ``intravoxel_precess`` are both needed to correctly
    simulate the effect gradients have on the magnetisation.
    """
    angle = spin_pos @ gradm  # shape: spins
    rot_mat = torch.zeros((angle.numel(), 3, 3), device=spins.device)
    rot_mat[:, 0, 0] = torch.cos(angle)
    rot_mat[:, 0, 1] = -torch.sin(angle)
    rot_mat[:, 1, 0] = torch.sin(angle)
    rot_mat[:, 1, 1] = torch.cos(angle)
    rot_mat[:, 2, 2] = 1
    # (spins x 3 x 3), (voxels x spins x 3)
    return torch.einsum("sij, vsj -> vsi", rot_mat, spins)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=30)
    p.add_argument("--slice", type=int, default=14)

    p.add_argument(
        "--seqs",
        nargs="+",
        default=["FLASH4mes","2D_spiral" ],
        help="Sequence names without _sequence_raw.json"
    )

    p.add_argument(
        "--spins",
        nargs="+",
        type=int,
        default=[12_000, 15_000, 20_000],
        help="Spin counts per voxel"
    )

    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory containing phantom data"
    )

    p.add_argument(
        "--sequence-dir",
        type=Path,
        default=ROOT / "src" / "sequence" / "gstar_seqs",
        help="Directory containing sequence JSON files"
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "isochromat_sim_results",
        help="Output directory"
    )

    p.add_argument(
        "--phantom",
        type=str,
        default="subject04_3T-highres-fat_custom_dense.npz"
    )

    p.add_argument("--no-b0", action="store_true")
    p.add_argument("--no-b1", action="store_true")
    p.add_argument("--perfect-spoiling", action="store_true", default=True)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--resume", action="store_true", default=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    args.data_dir = args.data_dir.resolve()
    args.sequence_dir = args.sequence_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    phantom_path = args.data_dir / args.phantom

    print("==== Bloch isochromat simulation ====", flush=True)
    print(f"ROOT         : {ROOT}", flush=True)
    print(f"data_dir     : {args.data_dir}", flush=True)
    print(f"sequence_dir : {args.sequence_dir}", flush=True)
    print(f"out_dir      : {args.out_dir}", flush=True)
    print(f"phantom      : {phantom_path}", flush=True)
    print(f"Nx, Ny, Nz   : {args.nx}, {args.ny}, {args.nz}", flush=True)
    print(f"slice        : {args.slice}", flush=True)
    print(f"seqs         : {args.seqs}", flush=True)
    print(f"spins        : {args.spins}", flush=True)
    print(f"cuda         : {torch.cuda.is_available()}", flush=True)

    if torch.cuda.is_available():
        print(f"gpu          : {torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"gpu mem GiB  : {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}",
            flush=True,
        )

    
    
    if not phantom_path.exists():
        raise FileNotFoundError(f"Phantom not found: {phantom_path}")

    for seq_name in args.seqs:
        seq_path = args.sequence_dir / f"{seq_name}_sequence_raw.json"

        if not seq_path.exists():
            raise FileNotFoundError(f"Sequence not found: {seq_path}")

        for spin_count in args.spins:
            print("", flush=True)
            print(f"Running seq={seq_name}, spins={spin_count}", flush=True)
            print(f"sequence path: {seq_path}", flush=True)

            seq = Sequence.import_file_for_mrtini_sim(str(seq_path))

            phantom = VoxelGridPhantom.brainweb(str(phantom_path))
            phantom = phantom.interpolate(args.nx, args.ny, args.nz).slices([args.slice])

            # Avoid phantom.plot() on cluster/headless Docker.
            # phantom.plot()

            phantom.size[-1] = torch.tensor([SLICE_THICKNESS])
            data = phantom.build(voxel_shape="sinc")

            if args.no_b0:
                data.B0 = torch.zeros_like(data.B0)
                print(" - B0 disabled", flush=True)

            if args.no_b1:
                data.B1 = torch.ones_like(data.B1)
                print(" - B1 disabled", flush=True)

            if args.perfect_spoiling:
                print(" - perfect spoiling enabled", flush=True)
            
            data = data.cuda()
            seq = seq.cuda()

            
            checkpoint_path = None

            if args.checkpoint_dir is not None:
                args.checkpoint_dir = args.checkpoint_dir.resolve()
                args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = args.checkpoint_dir / make_checkpoint_name(
                    seq_name, args.nx, args.ny, args.nz, args.slice, spin_count
                )
            
            
            signal = isochromat_sim(
                seq,
                data,
                spin_count=spin_count,
                perfect_spoiling=args.perfect_spoiling,
                print_progress=True,
                checkpoint_path=checkpoint_path if args.resume else None,
                resume=args.resume,
            )

            out_file = args.out_dir / (
                f"bloch_sim_signal_{seq_name}_"
                f"nx{args.nx}_ny{args.ny}_"
                f"spins{spin_count}_"
                f"{'no_b0' if args.no_b0 else 'b0'}"
                f"{'_no_b1' if args.no_b1 else '_b1'}.npz"
            )

            np.savez(out_file, signal=signal.detach().cpu().numpy())

            print(f"Saved: {out_file}", flush=True)

            del signal, data, seq, phantom
            torch.cuda.empty_cache()
            