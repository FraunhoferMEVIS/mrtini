# SPDX-FileCopyrightText: 2026 Fraunhofer Institute for Digital Medicine MEVIS
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of MRTini.
#
# MRTini is research software and is not qualified for use as a medical
# device or as part of a medical device.

# mrtini/slice_selective.py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict,List, Literal
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Proton gyromagnetic ratios (defaults)
GAMMA_HZ = 42.577478e6      # Hz/T
GAMMA_RAD = 2.0*np.pi*GAMMA_HZ # rad/s/T


@dataclass
class SliceSelectiveControls:
    # waveforms on a common time grid t (seconds)
    t: np.ndarray                # (T_total,)
    rf_complex_T: np.ndarray     # (T_total,) complex64, Tesla
    gz_Tm: np.ndarray            # (T_total,) float32, Tesla/m

    
    # metadata
    flip_angle_rad: float
    B1_scale: complex
    bandwidth_hz: float
    slice_thickness_m: float
    T_pulse_s: float
    gz_area_Tm_s: float
    gz_peak_Tm: float
    tbw: float
    dt_s: float

@dataclass
class BinningControlss:
    # binning (Tesla)
    bins_T: np.ndarray           # (B,), unique off-res (Tesla)
    bin_labels: Optional[np.ndarray]  # same shape as B0_map; int labels in [1..len(edges)-1]
    bin_center_hz: Optional[Dict[int, float]]
    bin_center_map_T: Optional[Dict[int, float]]  # label -> center (Tesla)


def bin_B0_map(
    B0_map_hz: np.ndarray,
    bin_hz: float = 2.5,
    gamma_hz: float = GAMMA_HZ,
) -> BinningControlss:
    """
    Bins a B0 map (Hz) into bins of width bin_hz, returns bin centers in Tesla and
    an integer label map the same shape as B0_map with values in [1..num_bins].
    """
    B0_map_flat = B0_map_hz.flatten()
    if bin_hz <= 0.0 or B0_map_flat.size == 0:
        return BinningControlss(
            bins_T=np.zeros(1, dtype=np.float32),
            bin_labels=None,
            bin_center_hz=None,
            bin_center_map_T=None,
        )

    # binning the frequency shifts into bins of size bin_hz
    edges = np.arange(B0_map_flat.min() - bin_hz/2,
                      B0_map_flat.max() + bin_hz/2 + bin_hz,
                      bin_hz,)      # rad/s edges
    bins = np.digitize(B0_map_flat, edges, right=True)     # (Nx,Ny) integer labels
    bin_centres = (edges[:-1] + edges[1:]) / 2          # works for any non-uniform grid
    # convert from hz to tesla
    bin_centres_T = bin_centres / gamma_hz  # in Tesla
    unique_bins = np.unique(bins.flatten())
    bin_center_map_T = {int(label): float(bin_centres_T[label - 1]) for label in unique_bins}
    bin_center_map_Hz = {int(label): float(bin_centres[label - 1]) for label in unique_bins}
    return BinningControlss(
        bins_T=bin_centres_T.astype(np.float32),
        bin_labels=bins.reshape(B0_map_hz.shape).astype(np.int32),
        bin_center_hz=bin_center_map_Hz,
        bin_center_map_T=bin_center_map_T,
    )
    


@dataclass(frozen=True)
class BinningControls:
    edges: np.ndarray                  # (B+1,) bin edges in native units
    centers: np.ndarray                # (B,) bin centers in native units
    labels: Optional[np.ndarray]       # (...) int32 labels in [1..B], 0 for invalid/masked
    center_by_label: Optional[Dict[int, float]]  # label -> center (native units)

@dataclass(frozen=True)
class JointBinning:
    b0: BinningControls
    b1: BinningControls
    pair_ids: np.ndarray               # (Nvox,) int32 in [0..Npairs-1], -1 for invalid
    pairs: np.ndarray                  # (Npairs,2) labels (b0_label, b1_label)
    pair_centers: np.ndarray           # (Npairs,2) centers (b0_center_native, b1_center_native)
    voxel_groups: List[np.ndarray]     # list length Npairs, each an array of voxel indices


def _finite_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)

def _make_uniform_edges(x: np.ndarray, bin_width: float) -> np.ndarray:
    # robust edges even if x is constant
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax == xmin:
        half = bin_width/2 if bin_width > 0 else 1.0
        return np.array([xmin-half, xmin+half], dtype=np.float64)
    start = xmin - bin_width/2
    stop  = xmax + bin_width/2 + bin_width
    return np.arange(start, stop, bin_width, dtype=np.float64)


def weighted_mean(x, w, eps=1e-12):
    s = (w*x).sum()
    return float((s/(w.sum()+eps)).item())

def bin_centers_weighted(values, labels, nbins, weights=None):
    # labels in [1..nbins], values shape (N,), weights shape (N,)
    if weights is None: weights = np.ones_like(values)
    centers = np.empty(nbins,)
    for lab in range(1, nbins+1):
        m = (labels == lab)
        if not np.any(m):
            centers[lab-1] = float('nan')
        else:
            centers[lab-1] = weighted_mean(values[m], weights[m])
    return centers

def bin_map_uniform(
    values: np.ndarray,
    bin_width: float,
    *,
    mask: Optional[np.ndarray]=None,
    right: bool=True,
    PD: Optional[np.ndarray]=None,
) -> BinningControls:
    """
    Uniform-width binning. Returns labels in [1..B], 0 for masked/invalid.
    """
    v = np.asarray(values)
    if v.ndim != 1:
        raise ValueError(f"values must be 1D [Nvox], got shape {v.shape}")
    if bin_width <= 0 or v.size == 0:
        return BinningControls(edges=np.zeros(2), centers=np.zeros(1), labels=None, center_by_label=None)

    m = _finite_mask(v)
    if mask is not None:
        m &= np.asarray(mask).astype(bool)
    if not np.any(m):
        lab = np.zeros_like(v, dtype=np.int32)
        return BinningControls(edges=np.zeros(2), centers=np.zeros(1), labels=lab, center_by_label={})

    vv = v[m].astype(np.float64)
    edges = _make_uniform_edges(vv, float(bin_width))
    labels = np.digitize(vv, edges, right=right).astype(np.int32)
    # centers = bin_centers_weighted(vv, labels, len(edges)-1, PD)
    centers = (edges[:-1] + edges[1:]) / 2.0
    # digitize gives 1..len(edges)-1
    labels_all = np.zeros_like(v, dtype=np.int32)
    # clip out-of-range (can happen at numerical extremes)
    labels = np.clip(labels, 1, centers.size)
    labels_all[m] = labels
    unique = np.unique(labels)
    center_by_label = {int(k): float(centers[k-1]) for k in unique}
    return BinningControls(edges=edges.astype(np.float32), centers=centers.astype(np.float32),
                           labels=labels_all, center_by_label=center_by_label)

def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, np.float64).reshape(-1)
    w = np.asarray(w, np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m): return float(np.nanmedian(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 1.0
    x = x[m]; w = w[m]
    idx = np.argsort(x)
    x = x[idx]; w = w[idx]
    cw = np.cumsum(w)
    cutoff = 0.5 * cw[-1]
    return float(x[np.searchsorted(cw, cutoff, side="left")])

def pd_weighted_b1_ref(B1: np.ndarray, PD: np.ndarray, *, mode: str="wmedian") -> float:
    B1 = np.asarray(B1, np.float64).reshape(-1)
    PD = np.asarray(PD, np.float64).reshape(-1)
    PD = np.clip(PD, 0.0, None)
    if mode == "wmean":
        m = np.isfinite(B1) & np.isfinite(PD) & (PD > 0)
        if not np.any(m): return float(np.nanmedian(B1[np.isfinite(B1)])) if np.any(np.isfinite(B1)) else 1.0
        return float((B1[m]*PD[m]).sum() / (PD[m].sum() + 1e-12))
    return weighted_median(B1, PD)  # default robust choice

def bin_B0_map_hz(
    B0_hz: np.ndarray,
    bin_hz: float=2.5,
    PD: Optional[np.ndarray]=None,
    *,
    mask: Optional[np.ndarray]=None,
) -> BinningControls:
    return bin_map_uniform(B0_hz.astype(np.float64), bin_hz,PD=PD, mask=mask, right=True)


def bin_B1_map(
    B1: np.ndarray,
    PD: Optional[np.ndarray]=None,
    *,
    mode: Literal["scale","tesla"]="scale",
    bin_width: Optional[float]=None,
    nbins: Optional[int]=10,
    ref: Optional[float]=None,
    mask: Optional[np.ndarray]=None,
    clamp: Optional[Tuple[float,float]]=None,
) -> Tuple[np.ndarray, BinningControls]:
    """
    Returns (B1_native, controls). If mode='scale', native is unitless scale a = B1/ref.
    If mode='tesla', native is Tesla amplitude.
    Choose either bin_width or nbins (preferred: bin_width for stability).
    """
    b1 = np.asarray(B1).astype(np.float64)
    if b1.ndim != 1:
        raise ValueError(f"B1 must be 1D [Nvox], got shape {b1.shape}")

    m = _finite_mask(b1)
    if mask is not None:
        m &= np.asarray(mask).astype(bool)

    if mode == "scale":
        if ref is None:
            # robust reference: median of finite masked values
            if not np.any(m): ref = 1.0
            else: ref = pd_weighted_b1_ref(b1[m], PD[m] if PD is not None else np.ones_like(b1[m]), mode="wmedian")
            if ref == 0.0: ref = 1.0
        b1n = b1 / ref
    elif mode == "tesla":
        b1n = b1
    else:
        raise ValueError("mode must be 'scale' or 'tesla'")

    if clamp is not None:
        lo, hi = clamp
        b1n = np.clip(b1n, lo, hi)

    if bin_width is None:
        # derive bin width from nbins across observed range
        if nbins is None or nbins <= 0:
            raise ValueError("Provide bin_width or a positive nbins.")
        if not np.any(m):
            bw = 0.1
        else:
            vv = b1n[m]
            r = float(np.max(vv) - np.min(vv))
            bw = r / float(nbins) if r > 0 else 0.05
        bin_width = bw

    ctrls = bin_map_uniform(b1n, float(bin_width), PD=PD, mask=m, right=True)
    return b1n.astype(np.float32), ctrls


def build_joint_binning(
    B0_hz: np.ndarray,
    B1: np.ndarray,
    PD: Optional[np.ndarray]=None,  # for weighted centers; not needed for binning; kept for API parity
    positions_xyz: Optional[np.ndarray]=None,  # not needed for binning; kept for API parity
    *,
    b0_bin_hz: float=5,
    b1_mode: Literal["scale","tesla"]="scale",
    b1_bin_width: Optional[float]=0.05,
    b1_nbins: Optional[int]=10,
    b1_ref: Optional[float]=None,
    mask: Optional[np.ndarray]=None,
    b1_clamp: Optional[Tuple[float,float]]=None,
) -> JointBinning:
    """
    B0_hz, B1 are [Nvox]. Returns sparse pair list and voxel groups.
    """
    B0_hz = np.asarray(B0_hz).astype(np.float64).reshape(-1)
    B1 = np.asarray(B1).astype(np.float64).reshape(-1)
    if B0_hz.shape != B1.shape:
        raise ValueError(f"B0_hz and B1 must have same shape, got {B0_hz.shape} vs {B1.shape}")
    N = B0_hz.size

    m = _finite_mask(B0_hz) & _finite_mask(B1)
    if mask is not None:
        m &= np.asarray(mask).astype(bool)

    b0_ctrl = bin_B0_map_hz(B0_hz, bin_hz=b0_bin_hz, mask=m, PD=PD)
    b1_native, b1_ctrl = bin_B1_map(B1,PD=PD, mode=b1_mode, bin_width=b1_bin_width, nbins=b1_nbins,
                                   ref=b1_ref, mask=m, clamp=b1_clamp)

    b0_lab = b0_ctrl.labels if b0_ctrl.labels is not None else np.zeros(N, np.int32)
    b1_lab = b1_ctrl.labels if b1_ctrl.labels is not None else np.zeros(N, np.int32)

    valid = (b0_lab > 0) & (b1_lab > 0) & m
    pair_ids = np.full(N, -1, dtype=np.int32)
    if not np.any(valid):
        return JointBinning(b0=b0_ctrl, b1=b1_ctrl, pair_ids=pair_ids,
                            pairs=np.zeros((0,2), np.int32),
                            pair_centers=np.zeros((0,2), np.float32),
                            voxel_groups=[])

    pairs_all = np.stack([b0_lab[valid], b1_lab[valid]], axis=1).astype(np.int32)
    uniq_pairs, inv = np.unique(pairs_all, axis=0, return_inverse=True)
    # map back to full voxel array
    pair_ids[valid] = inv.astype(np.int32)

    # build groups
    voxel_idx_valid = np.nonzero(valid)[0]
    voxel_groups: List[np.ndarray] = []
    for pid in range(uniq_pairs.shape[0]):
        voxel_groups.append(voxel_idx_valid[inv == pid])

    # centers for pairs
    b0_center = b0_ctrl.center_by_label or {}
    b1_center = b1_ctrl.center_by_label or {}
    pair_centers = np.zeros((uniq_pairs.shape[0], 2), dtype=np.float32)
    for i,(l0,l1) in enumerate(uniq_pairs):
        pair_centers[i,0] = float(b0_center[int(l0)])
        pair_centers[i,1] = float(b1_center[int(l1)])

    return JointBinning(
        b0=b0_ctrl, b1=b1_ctrl,
        pair_ids=pair_ids,
        pairs=uniq_pairs.astype(np.int32),
        pair_centers=pair_centers,
        voxel_groups=voxel_groups
    )


def quantize_from_labels(labels: np.ndarray, center_by_label: Dict[int,float]) -> np.ndarray:
    """
    Replace each voxel by its bin center (quantized map). labels==0 stays NaN.
    """
    labels = np.asarray(labels).astype(np.int32).reshape(-1)
    out = np.full(labels.shape, np.nan, dtype=np.float32)
    for lab, c in center_by_label.items():
        out[labels == int(lab)] = float(c)
    return out

def _default_slr_ex(N: int, tbw: float) -> np.ndarray:
    """Fallback RF designer: Hamming-windowed sinc 'excitation' profile, unitless."""
    n = np.arange(N) - (N-1)/2
    bw = tbw / N
    x = np.pi * bw * n
    sinc = np.sinc(x/np.pi)
    win = 0.54 - 0.46*np.cos(2*np.pi*(n - n.min())/(N-1))
    return (sinc * win).astype(np.complex64)


def _trap_from_area(area_Tm_s: float, gmax_Tm: float, dgdt_Tm_s: float, dt_s: float) -> np.ndarray:
    """
    Minimal trapezoid generator by area with slew and peak limits.
    Returns gz(t) in Tesla/m sampled at dt_s. Symmetric ramp up/down; flat as needed.
    """
    area = float(abs(area_Tm_s))
    if area == 0.0:
        return np.zeros(1, dtype=np.float32)

    try:
        import pulpy as pp
        gz, meta = pp.grad.waveform.trap_grad(area, gmax_Tm, dgdt_Tm_s, dt_s) # type: ignore
    except Exception as e:
        print(f"Error importing Pulpy generating trapezoid using numpy: {e}")

        # ramp time (s) limited by gmax and slew
        t_ramp = min(gmax_Tm / dgdt_Tm_s, area / (2.0 * gmax_Tm))  # ensure non-negative flat
        g_at_ramp = min(gmax_Tm, dgdt_Tm_s * t_ramp)
        # flat time (s)
        t_flat = max(0.0, (area - g_at_ramp * t_ramp) / g_at_ramp) if g_at_ramp > 0 else 0.0

        # sample counts
        Nr = max(1, int(np.round(t_ramp / dt_s)))
        Nf = max(0, int(np.round(t_flat / dt_s)))
        # recompute exact durations to honor dt grid
        t_ramp = Nr * dt_s
        t_flat = Nf * dt_s
        g_at_ramp = min(gmax_Tm, dgdt_Tm_s * t_ramp)

        # build ramps
        up = np.linspace(0.0, g_at_ramp, Nr, endpoint=False, dtype=np.float32)
        flat = np.full(Nf, g_at_ramp, dtype=np.float32)
        down = np.linspace(g_at_ramp, 0.0, Nr, endpoint=True, dtype=np.float32)
        gz = np.concatenate([up, flat, down], dtype=np.float32)  # Tesla/m
        # scale to exact area (tiny correction)
        corr = area / np.trapz(gz, dx=dt_s)
        gz *= corr
    return gz.astype(np.float32)



def build_slice_selective(
    *,
    N: int = 512,
    tbw: float = 8.0,
    dt_s: float = 1e-6,
    slice_thickness_m: float = 5e-3,
    flip_angle_deg: float = 90.0,
    gamma_hz: float = GAMMA_HZ,
    gamma_rad: float = GAMMA_RAD,
    # gradient limits
    gmax_Tm: Optional[float] = None,      # if None, use exact needed; else clamp
    dgdt_Tm_s: float = 180.0,             # Tesla/m/s (≈ 45 mT/m per ms)
    # optional external pulse designer: returns baseband complex (unitless) length N
    rf_designer=None,                     # e.g. lambda N,tbw: prf.slr.dzrf(N,tbw,'ex','ls',d1=0.01,d2=0.01)
    # rephaser
    add_rephaser: bool = True,
    rephaser_fraction: float = 1.0,       # 1.0 → full area refocus
    # interpolation kind
    interp_kind: str = "linear",
) -> SliceSelectiveControls:
    """
    Designs slice-selective RF + slice gradient (+rephaser), scales RF to flip_angle,
    bins a B0 map into off-res bins (Tesla), and returns everything on a common time grid.

    Returns SliceSelectiveControls with rf_complex_T, gz_Tm, t, bins_T, labels, and metadata.
    """
    # 1) RF baseband (unitless)
    if rf_designer is None:
        try:
            import pulpy as pp
            import pulpy.rf as prf
            pulse0 = prf.slr.dzrf(N, int(tbw), 'ex', 'ls', d1=0.01, d2=0.01)
        except Exception:
            pulse0 = _default_slr_ex(N, tbw)
    else:
        pulse0 = rf_designer(N, tbw)
    pulse0 = np.asarray(pulse0, dtype=np.complex64)
    assert pulse0.shape == (N,), "RF designer must return length-N complex array"

    # 2) pulse timing & bandwidth
    T_pulse = N * dt_s
    bandwidth_hz = tbw / T_pulse

    # 3) slice gradient amplitude (Tesla/m)
    Gz_needed = bandwidth_hz / (gamma_hz * slice_thickness_m)
    Gz_peak = Gz_needed if gmax_Tm is None else min(Gz_needed, gmax_Tm)

    # 4) trapezoid for slice select (area = Gz_needed * T_pulse)
    area = Gz_needed * T_pulse
    dt_grad_ = dt_s
    gz_sel = _trap_from_area(area_Tm_s=area, gmax_Tm=Gz_peak, dgdt_Tm_s=dgdt_Tm_s, dt_s=dt_grad_)

    
    
    # 5) optional rephaser (negative, by fraction of selection area)
    if add_rephaser and rephaser_fraction != 0.0:
        gz_reph = _trap_from_area(area_Tm_s=abs(np.trapz(gz_sel.squeeze(), dx=dt_grad_))*rephaser_fraction,
                                  gmax_Tm=Gz_peak, dgdt_Tm_s=dgdt_Tm_s, dt_s=dt_grad_)
        gz_full = np.concatenate([gz_sel.squeeze(), -gz_reph.squeeze()]).astype(np.float32)
        t_grad  = np.linspace(0.0, T_pulse, gz_sel.size, endpoint=False)
        t_reph  = np.linspace(T_pulse, T_pulse*1.5, gz_reph.size, endpoint=True)
        t_full  = np.concatenate([t_grad, t_reph])

    else:
        gz_full = gz_sel.astype(np.float32)
        t_full  = np.linspace(0.0, gz_full.size*dt_s, gz_full.size, endpoint=False)

    # 6) resample RF (which lives on [0, T_pulse]) to t_full (RF = 0 outside [0, T_pulse])
    t_rf = np.linspace(0.0, T_pulse, N, endpoint=False)
    rf_i = interp1d(t_rf, pulse0, kind=interp_kind, bounds_error=False, fill_value=0.0)
    rf_resamp = rf_i(t_full).astype(np.complex64)  # complex, unitless

    # 7) flip-angle scaling to Tesla
    flip_angle = np.deg2rad(flip_angle_deg)
    B1_area = np.trapz(pulse0, dx=dt_s)  # complex area of unscaled baseband
    if np.abs(B1_area) == 0.0:
        raise ValueError("RF area is zero; cannot scale to flip angle.")
    B1_scale = flip_angle / (gamma_rad * B1_area)  # complex scalar
    rf_T = (rf_resamp * B1_scale).astype(np.complex64)  # Tesla
    gz_Tm = gz_full.astype(np.float32)
            # bxyz = ctrl(t)
    # plt.figure(figsize=(12,4))
    # plt.subplot(1,3,1)
    # plt.plot(t_full, rf_T.real, label="B1x (T)")
    # plt.xlabel("Time (s)"); plt.ylabel("B1 (T)"); plt.legend(); plt.grid()
    # plt.title("RF Pulse")
    # plt.subplot(1,3,2)
    # plt.plot(t_full, gz_Tm*1e3)
    # plt.xlabel("Time (s)"); plt.ylabel("Gz (mT/m)"); plt.grid()
    # plt.title("Slice Gradient")
    # plt.show()


    return SliceSelectiveControls(
        t=t_full.astype(np.float32),
        rf_complex_T=rf_T,
        gz_Tm=gz_Tm,
        flip_angle_rad=float(flip_angle),
        B1_scale=B1_scale,
        bandwidth_hz=float(bandwidth_hz),
        slice_thickness_m=float(slice_thickness_m),
        T_pulse_s=float(T_pulse),
        gz_area_Tm_s=float(np.trapz(gz_sel, dx=dt_s)),
        gz_peak_Tm=float(Gz_peak),
        tbw=float(tbw),
        dt_s=float(dt_s),
    )
