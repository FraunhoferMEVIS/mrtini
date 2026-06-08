'''
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 Fraunhofer MEVIS, Germany
#
# This file is part of MRtini Simulator.
# It contains original code written for MRtini Simulator.
'''


# src/mrtini/mrtini_main.py
import hashlib
import math
import sys
sys.path.append("./")  # to access mrtini when running this script
from src.mrtini.config import GAMMA_HZ
# logging_utils.py
import logging, sys, time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from src.sequence.sequence import PulseUsage
from src.mrtini.RF_banks import (
    BankAxes,
    Coeff6,
    IMap,
    RFKernel6BankSO3,
    RotTaps1D,
    build_imap,
    build_kgrid_projection,
    make_axes,
    select_support_by_energy
)

from src.sequence.sequence import Sequence
from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict
import torch

from .bloch import solve_slice
from src.sequence.slice_select import (
    build_joint_binning,
)

# ============================ MRtini main class ============================
# ---- gyromagnetic constants (proton) ----
GAMMA_RAD = 2.0 * np.pi * GAMMA_HZ
from src.mrtini.config import DEVICE, PAD, DTYPE

@dataclass
class PulseArtifacts:
    key: str
    z_m: torch.Tensor
    ax: BankAxes
    bank: RFKernel6BankSO3                      # RFKernel6BankSO3
    Rz: Dict[int, torch.Tensor]  # {bin_label -> (3,3,Nz)}
    t_rf: float
    phi0: float
    nominal_taps: RotTaps1D
    support: Dict[str, torch.Tensor]   # rel -> k_idx
    vox6: Coeff6
    supp_map: Dict[int, int]

@dataclass
class PulseArtifactsBySpecies:
    water: PulseArtifacts
    fat: PulseArtifacts | None = None
    combined_taps : Optional[RotTaps1D] = None  # combined fat and water, if applicable
@dataclass
class HybridArtifacts:
    pulses: Dict[str, PulseArtifactsBySpecies]  # key -> artifacts
    span_key: Dict[int, str]           # span index -> key
    phi0_by_span: Dict[int, float]
    z_m: torch.Tensor


def _hash_array(a: np.ndarray, quant: float = 1e-6) -> str:
    q = np.round(np.asarray(a, dtype=np.float64) / quant).astype(np.int64)
    return hashlib.sha1(q.tobytes()).hexdigest()[:16]


def _freq_tag(f):
    f = np.asarray(f)
    if f.ndim == 0 or f.size == 1:
        return f"{float(f):.6e}"
    if np.allclose(f, f.flat[0], atol=1e-9, rtol=0):  # broadcasted-constant vector
        return f"{float(f.flat[0]):.6e}"
    return "sha1:" + _hash_array(f, quant=1e-6)


def is_constant_phase(phi, amp, tol_rad: float = 1e-3) -> tuple[bool, float]:
    """
    Decide if all entries in `phi` are the same angle up to numerical noise.
    Angles are in radians. Returns (is_const, phi0).
    """
    mask = np.where(np.abs(amp) > 0.0)
    phi = np.asarray(phi, dtype=np.float64)
    phi = phi[mask]
    if phi.size == 0:
        return True, 0.0
    if phi.size == 1:
        return True, float(phi[0])

    base = float(phi[0])
    # wrap diffs to [-pi, pi)
    d = ((phi - base + np.pi) % (2 * np.pi)) - np.pi
    is_const = np.max(np.abs(d)) < tol_rad
    return is_const, (
        base if is_const else float(np.arctan2(np.sin(phi).mean(), np.cos(phi).mean()))
    )


def _qhash(x: np.ndarray, q: float = 1e-7) -> str:
    """quantize + hash array for robust cache keys."""
    xq = np.round(x / q).astype(np.int64, copy=False)
    return hashlib.sha1(xq.tobytes()).hexdigest()

def build_complex_rf(span):
    t = np.asarray(span["t"], dtype=np.float64)
    amp = np.asarray(span["rf_amp"], dtype=np.float64) / GAMMA_HZ

    phase = np.asarray(span.get("rf_phase", 0.0), dtype=np.float64)
    phase = np.broadcast_to(phase, amp.shape).copy()

    rf_freq = np.asarray(span.get("rf_freq", 0.0), dtype=np.float64)
    rf_freq = np.broadcast_to(rf_freq, amp.shape)

    rf_fm = span.get("rf_fm", None)
    if rf_fm is None or (hasattr(rf_fm, "__len__") and len(rf_fm) == 0):
        rf_fm = np.zeros_like(amp, dtype=np.float64)
    else:
        rf_fm = np.asarray(rf_fm, dtype=np.float64)
        rf_fm = np.broadcast_to(rf_fm, amp.shape)

    f_inst = rf_freq + rf_fm   # Hz

    phi_f = np.zeros_like(amp, dtype=np.float64)
    if len(t) > 1:
        dt = np.diff(t)
        phi_f[1:] = 2*np.pi * np.cumsum(f_inst[:-1] * dt)   # ZOH-consistent

    phi_total = phase + phi_f
    rf_complex = amp * np.exp(1j * phi_total)
    return rf_complex.astype(np.complex64)

def _bloch_kernel_signature(
    span: dict,
) -> tuple[str, np.ndarray,np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    span MUST be a 'bloch' item from seq.mrtini_hybrid with:
      - 't' (list/np.ndarray, seconds)
      - 'rf_amp' (same len)
      - 'rf_phase' (same len; often constant)
      - 'gz' (same len, Tesla/m)
      - optional 'rf_freq' (Hz) — include if your slicer uses it (default 0)
    Returns: (key, rf_for_solve (complex), gz, t, meta{phi0, const_phase, rf_freq})
    """
    assert span["kind"] == "bloch"
    t = np.asarray(span["t"], dtype=np.float64)
    amp = np.asarray(span["rf_amp"], dtype=np.float64) / GAMMA_HZ
    phi = np.asarray(span["rf_phase"], dtype=np.float64)
    gz = np.asarray(span["gz"], dtype=np.float64) / GAMMA_HZ
    rf_freq = np.asarray(span.get("rf_freq", 0.0) or 0.0)
    rf_freq = np.broadcast_to(rf_freq, amp.shape)

    rf_fm = span.get("rf_fm", None)
    if rf_fm is None or (hasattr(rf_fm, "__len__") and len(rf_fm) == 0):
        rf_fm = np.zeros_like(amp, dtype=np.float64)
    else:
        rf_fm = np.asarray(rf_fm, dtype=np.float64)
        rf_fm = np.broadcast_to(rf_fm, amp.shape)
    
    const_phase, phi0 = is_constant_phase(phi, amp)

    if const_phase:
        # strip the global phase; solve with a real envelope
        rf = amp.astype(np.float64)
        rf_im = np.zeros_like(rf)
    else:
        rf_c = amp * np.exp(1j * phi)  # truly complex envelope
        rf = rf_c.real.astype(np.float64)
        rf_im = rf_c.imag.astype(np.float64)

    f_inst = rf_freq + rf_fm   # Hz
    phi_f = np.zeros_like(amp, dtype=np.float64)
    if len(t) > 1:
        dt = np.diff(t)
        phi_f[1:] = 2*np.pi * np.cumsum(f_inst[:-1] * dt)   # ZOH-consistent


    
    f_tag = _freq_tag(rf_freq)  # handles scalar/broadcast/hash
    key = (
        float(np.mean(np.diff(t))) if len(t) > 1 else 0.0,  # dt (assume uniform)
        _qhash(rf.view(np.float64)),
        _qhash(rf_im.view(np.float64)),
        _qhash(gz.view(np.float64)),
        f_tag,
    )
    key_str = "dt={:.9e}|rfR={}|rfI={}|gz={}|f={}".format(*key)

    meta = dict(phi0=phi0, const_phase=const_phase, rf_freq=rf_freq)
    # rf_for_solve: pack back as complex (imag=0 if const_phase)
    rf_for_solve = rf + 1j * rf_im
    rf_complex = rf_for_solve * np.exp(1j * phi_f)
    return key_str, rf_complex.astype(np.complex64),f_inst, gz.astype(np.float32), t, meta

def k_grid_from_seq(seq):
    # Collect all gradient moments and event times from the sequence
    gm_all = torch.cat([rep.gradm for rep in seq], dim=0) * 2*math.pi  # (N, 3)
    et_all = torch.cat([rep.event_time for rep in seq])    # (N,)

    # Compute absolute values
    abs_grad = torch.abs(gm_all)  # (N, 3)
    abs_dt = torch.abs(et_all)    # (N,)

    # Find the smallest non-trivial values above thresholds
    min_kt_step = torch.tensor(
        [
            torch.min(abs_grad[:, i][abs_grad[:, i] > 1e-3]) 
            if abs_grad[:, i][abs_grad[:, i] > 1e-3].numel() > 0 
            else torch.tensor(float('inf'), device=abs_grad.device)
            for i in range(3)
        ] + [
            torch.min(abs_dt[abs_dt > 1e-6]) 
            if abs_dt[abs_dt > 1e-6].numel() > 0 
            else torch.tensor(float('inf'), device=abs_dt.device)
        ], 
        device=abs_grad.device
    )

    # Compute the inverse grid spacing with clamping
    inv_kt_grid = 1.0 / torch.clamp(0.1 * torch.where(
        min_kt_step == float('inf'), torch.tensor(1.0, device=min_kt_step.device), min_kt_step
    ), min=1e-9, max=1.0)

    return inv_kt_grid


def make_z_grid_slice_select(L_slice, pad, TBW, T_pulse, Gz_max, os=1.0, device="cpu"):
    Z = 0.5 * L_slice * pad                  # half-span [m]
    BW = TBW / T_pulse                       # Hz (your convention)
    dz_edge = BW / (abs(Gz_max) + 1e-30)     # m (your convention)

    dz_target = dz_edge / (16.0 * os)        # target sampling

    # pick K from dz_target, then SNAP dz so that K*dz = 2Z exactly
    K = int(math.ceil((2*Z) / float(dz_target)))
    if K % 2 == 0: K += 1                    # force ODD K (nice fftshift lattice)
    dz = (2*Z) / K                           # exact periodic spacing

    # FFT-friendly centered grid: [-Z, ..., 0, ..., Z-dz]
    n = torch.arange(K, device=device, dtype=DTYPE)
    z = (n - (K//2)) * dz                    # exact 0 at index K//2

    return z

def make_pairs(b0_centers_T, b1_centers):
    # b0_centers_T: (Nb0,) Tesla, b1_centers: (Nb1,) scale
    b0 = torch.as_tensor(b0_centers_T, dtype=torch.float32)
    b1 = torch.as_tensor(b1_centers, dtype=torch.float32)
    Nb0, Nb1 = b0.numel(), b1.numel()
    b0g = b0[:,None].expand(Nb0,Nb1).reshape(-1)  # (B,)
    b1g = b1[None,:].expand(Nb0,Nb1).reshape(-1)  # (B,)
    return b0g, b1g, Nb0, Nb1

def setup_logger(name="bloch", level="INFO", log_dir="logs", filename="run.log"):
    logger = logging.getLogger(name)
    if logger.handlers:  # don't double-add handlers (common in notebooks)
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File (rotating)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(Path(log_dir)/filename, maxBytes=10_000_000, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

@contextmanager
def log_stage(logger, msg, level=logging.INFO):
    t0 = time.perf_counter()
    logger.log(level, f"{msg} ...")
    try:
        yield
        dt = time.perf_counter() - t0
        logger.log(level, f"{msg} done in {dt:.3f}s")
    except Exception:
        dt = time.perf_counter() - t0
        logger.exception(f"{msg} FAILED after {dt:.3f}s")
        raise

class MRtiniSim:
    """
    One-call builder that:
      - creates slice-selective RF/Gz on your RF grid,
      - bins B0 (→ Tesla),
      - computes per-bin Bloch→EPG H-kernels,
      - returns an EPG3D object ('mrtini') + artifacts.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        gamma_hz: float = GAMMA_HZ,
        seq: Sequence | None = None,
    ):
        self.device = device
        self.gamma_hz = gamma_hz
        self.gamma_rad = (2.0 * np.pi * gamma_hz,)
        self.seq = seq
        self.log = setup_logger(name="mrtini.bloch", level="INFO")

    def _build_pulse_artifacts(self, *, key: str, Rz: torch.Tensor,
                            z_m: torch.Tensor, t: np.ndarray, meta: dict,
                            jb, phantom_data,
                            device, tap_threshold: float):

        ax = make_axes(jb.b0.centers, jb.b1.centers, device)
        bank = RFKernel6BankSO3.from_Rxyz_flat(Rz, ax=ax, z_m=z_m, device=device, in_layout="B33Nz")

        kgrid = build_kgrid_projection(z_m=z_m, device=device)
        bank.attach_kgrid(kgrid)
        im = build_imap(phantom_data.B0, phantom_data.B1, ax, device=device, clamp=True)

        nom6_full = bank.compute_nominal_PD_weighted(im, phantom_data.PD)

        nominal_taps = RotTaps1D(
            pp=nom6_full.pp, pz=nom6_full.pz, mp=nom6_full.mp, mz=nom6_full.mz, zp=nom6_full.zp, zz=nom6_full.zz,
            as_dict={"++": nom6_full.pp, "+z": nom6_full.pz, "-+": nom6_full.mp, "-z": nom6_full.mz, "z+": nom6_full.zp, "zz": nom6_full.zz,},
            deltas=bank.kgrid.deltas, dk=bank.kgrid.dk_tap,
        )

        ### at eps =1e-5 the slice profiles are recovered  nicely :D
        supp = select_support_by_energy(nom6_full, kgrid, eps=tap_threshold)
        support = {rel: supp.k_idx for rel in nominal_taps.as_dict.keys()}
        vox6, k_supp_map = bank.eval_voxels_so3(im, supp.k_idx)

        return PulseArtifacts(
            key=key, z_m=z_m, ax=ax, bank=bank,
            Rz=Rz,
            t_rf=float(t[-1] - t[0]),
            phi0=float(meta["phi0"]),
            nominal_taps=nominal_taps,
            support=support,
            vox6=vox6,
            supp_map=k_supp_map,
        )
    # ---- driver -------------------------------------------------------------

    def bake_bloch_kernels(
        self,
        *,
        phantom_data,  # MRzero SimData (for B0)
        b0_bin_hz = 10,
        b1_bin_width: float = 0.05,
        device: str = DEVICE,
        z_span_m: float = 8e-3,
        solver: str = "so3",  # "so3" or "spinor"
        tbw: float = 10.0,
        global_RF=False,
        tap_threshold: float = 1e-6,
    ) ->  HybridArtifacts:
        """
        1) bin B0 from phantom_data
        2) create an EPG3D ready object (no RF kernels applied yet)
        3) for every unique 'bloch' span in self.seq.mrtini_hybrid, solve once and cache Hf
        """

        if self.seq is None:
            raise ValueError("sequence is None — provide a sequence first")

        ## for debugging, you can override phantom B0/B1 here
        # phantom_data.B0 = torch.zeros_like(phantom_data.B0)
        # phantom_data.B1 = torch.ones_like(phantom_data.B1)   #default B1=1 (no scaling)

        # Inputs
        # B0_hz: [Nvox] float
        # B1_map: [Nvox] float (Tesla or relative; you choose)
        # pos: [Nvox,3] float (not used for binning here)
        B0_hz = phantom_data.B0.cpu().numpy()
        B1_map = phantom_data.B1.cpu().numpy() if hasattr(phantom_data, "B1") else np.ones_like(B0_hz)
        # B0_hz = np.zeros_like(B0_hz)  # for testing, ignore B0 inhomogeneity only for rf pulse application 
        # B1_map =np.ones_like(B1_map)  # for testing, ignore B1 inhomogeneity only for rf pulse application 
        PD = phantom_data.PD.cpu().numpy() if hasattr(phantom_data, "PD") else np.zeros((B0_hz.shape[0], 3))
        jb = build_joint_binning(
            B0_hz, B1_map, PD,
            b0_bin_hz=b0_bin_hz,
            b1_mode="scale",          # recommended
            b1_bin_width=b1_bin_width,        # 5% bins
            b1_ref=None,              # uses median(B1) as ref
            b1_clamp=(0.5, 1.5)       # optional, keeps bank bounded
        )
        b0_T, b1_scale, Nb0, Nb1 = make_pairs(jb.b0.centers/GAMMA_HZ, jb.b1.centers)

        inv_k_grid = k_grid_from_seq(self.seq)
        span = None
        for s in self.seq.mrtini_hybrid:
            if s.get("kind") != "bloch": 
                continue
            if np.max(s.get("gz", np.array([0.0]))) == 0.0:
                continue
            span = s
            break
        if span is None:
            raise ValueError("No 'bloch' span found in sequence for kernel baking.")
        rf_env = span["rf_amp"]
        gz = span["gz"]
        t = span["t"]
        dt_s = float(t[1] - t[0]) if len(t) > 1 else 1e-6
        N_rf = len(rf_env)
        gmax_cyc = np.max(np.abs(gz)) if gz is not None else None

        z_m = make_z_grid_slice_select(
            L_slice=z_span_m,
            pad=PAD,
            TBW=tbw,
            T_pulse=dt_s*N_rf,
            Gz_max=gmax_cyc if gmax_cyc is not None else 0.0,
            os=4,  ## over sampling factor for the z grid (vs Nyquist for the bandwidth of the pulse)
            device=device,
        )

        pulses: Dict[str, PulseArtifactsBySpecies] = {}
        span_key: Dict[int, str] = {}
        phi0_by_span: Dict[int, float] = {}

        for si, span in enumerate(self.seq.mrtini_hybrid):
            if span["kind"] != "bloch": 
                continue

            key, rf_env, rf_freq, gz, t, meta = _bloch_kernel_signature(span)  # your existing function (with freq tag fix)
            span_key[si] = key
            phi0_by_span[si] = float(meta["phi0"])

            if key in pulses:
                continue  # already solved + built taps/bank/etc for this pulse
            if global_RF:
                gz = np.zeros_like(gz)
            '''
            refocusing pulse special case: if the span is tagged as refocusing,
            we can skip the Bloch solve and just set Gz=0 (no slice selectivity)
            since we know it's a 180° pulse that perfectly inverts Mz
            regardless of off-resonance or B1 (ideal hard pulse assumption).
            This is a common approximation for refocusing pulses in hybrid sequences,
            and it saves time by avoiding unnecessary Bloch solves for these spans.
            You can remove this condition if you want to solve Bloch for refocusing pulses
            as well (e.g. if they have non-ideal profiles or you want to capture their 
            slice selectivity).
            '''            
            # if span['pulse_type'] == PulseUsage.REFOC:  ## enable this if you don't care about slice selectivity of refocusing pulses and want to skip Bloch solve for them
            #     gz = np.zeros_like(gz)

            chem_shift = (
                torch.unique(phantom_data.chemical_shift) if hasattr(phantom_data, "chemical_shift") else None
            )
            # ---- solve Bloch for this pulse key ----
            with log_stage(self.log, f"Solving Bloch for span #{si} with key {key[-10:]}"):
                solve_out = solve_slice(
                    rf_complex=rf_env,
                    rf_freq=rf_freq,
                    gz=gz,
                    t_grid=t,
                    z_m=z_m,
                    b0_T=b0_T,
                    b1_scale=b1_scale,
                    device=device,
                    plot_slice_profile=False,
                    backend=solver,
                    rf_center_time=span.get("rf_center"),
                    chem_shift_Hz=chem_shift,
                )  # (3,3,Nz) real rotation matrices per z
            solution = (
                solve_out[0] if isinstance(solve_out, (list, tuple)) else solve_out
            )  # adapt to your return

            iv_kt_z = inv_k_grid[2]  # z-dimension inverse grid spacing
            # ---- build per-key artifacts ----
            if key not in pulses:
                pulses[key] = PulseArtifactsBySpecies(water=None, fat=None)
            for specieSolve in solution.Rz_solve:
                Rz = specieSolve.Rz_B33Nz
                specie = specieSolve.name.lower()
                artifact = self._build_pulse_artifacts(
                    key=key,
                    Rz=Rz,
                    z_m=z_m,
                    t=t,
                    meta=meta,
                    jb=jb,
                    phantom_data=phantom_data,
                    device=device,
                    tap_threshold=tap_threshold,
                )
                if specie == "water":
                    pulses[key].water = artifact
                elif specie == "fat":
                    pulses[key].fat = artifact
                else:
                    # Optionally handle other species or raise an error
                    self.log.warning(f"Unknown species '{specie}' encountered in solution.Rz_solve")

            def maxmag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.where(a.abs() >= b.abs(), a, b)

            nom_combined = RotTaps1D(
                pp=maxmag(
                    pulses[key].water.nominal_taps.pp,
                    (
                        pulses[key].fat.nominal_taps.pp
                        if pulses[key].fat is not None
                        else pulses[key].water.nominal_taps.pp
                    ),
                ),
                pz=maxmag(
                    pulses[key].water.nominal_taps.pz,
                    (
                        pulses[key].fat.nominal_taps.pz
                        if pulses[key].fat is not None
                        else pulses[key].water.nominal_taps.pz
                    ),
                ),
                mp=maxmag(
                    pulses[key].water.nominal_taps.mp,
                    (
                        pulses[key].fat.nominal_taps.mp
                        if pulses[key].fat is not None
                        else pulses[key].water.nominal_taps.mp
                    ),
                ),
                mz=maxmag(
                    pulses[key].water.nominal_taps.mz,
                    (
                        pulses[key].fat.nominal_taps.mz
                        if pulses[key].fat is not None
                        else pulses[key].water.nominal_taps.mz
                    ),
                ),
                zp=maxmag(
                    pulses[key].water.nominal_taps.zp,
                    (
                        pulses[key].fat.nominal_taps.zp
                        if pulses[key].fat is not None
                        else pulses[key].water.nominal_taps.zp
                    ),
                ),
                zz=maxmag(
                    pulses[key].water.nominal_taps.zz,
                    (
                        pulses[key].fat.nominal_taps.zz
                        if pulses[key].fat is not None
                        else pulses[key].water.nominal_taps.zz
                    ),
                ),
                as_dict={
                    "++": maxmag(
                        pulses[key].water.nominal_taps.pp,
                        (
                            pulses[key].fat.nominal_taps.pp
                            if pulses[key].fat is not None
                            else pulses[key].water.nominal_taps.pp
                        ),
                    ),
                    "+z": maxmag(
                        pulses[key].water.nominal_taps.pz,
                        (
                            pulses[key].fat.nominal_taps.pz
                            if pulses[key].fat is not None
                            else pulses[key].water.nominal_taps.pz
                        ),
                    ),
                    "-+": maxmag(
                        pulses[key].water.nominal_taps.mp,
                        (
                            pulses[key].fat.nominal_taps.mp
                            if pulses[key].fat is not None
                            else pulses[key].water.nominal_taps.mp
                        ),
                    ),
                    "-z": maxmag(
                        pulses[key].water.nominal_taps.mz,
                        (
                            pulses[key].fat.nominal_taps.mz
                            if pulses[key].fat is not None
                            else pulses[key].water.nominal_taps.mz
                        ),
                    ),
                    "z+": maxmag(
                        pulses[key].water.nominal_taps.zp,
                        (
                            pulses[key].fat.nominal_taps.zp
                            if pulses[key].fat is not None
                            else pulses[key].water.nominal_taps.zp
                        ),
                    ),
                    "zz": maxmag(
                        pulses[key].water.nominal_taps.zz,
                        (
                            pulses[key].fat.nominal_taps.zz
                            if pulses[key].fat is not None
                            else pulses[key].water.nominal_taps.zz
                        ),
                    ),
                },
                deltas=pulses[key].water.nominal_taps.deltas,
                dk=pulses[key].water.nominal_taps.dk,
            )
            pulses[key].combined_taps = nom_combined

        return HybridArtifacts(pulses=pulses, span_key=span_key, phi0_by_span=phi0_by_span, z_m=z_m)
