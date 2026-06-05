# mrtini/slice_selective.py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict,List, Literal
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

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


def sanity_check_quantization(
    B0_hz: np.ndarray, B1: np.ndarray, jb: JointBinning,
    *, b1_mode: str="scale", b1_ref: Optional[float]=None,
    make_plots: bool=True, title_prefix: str=""
):
    """
    Reconstruct quantized versions and optionally plot original vs quantized and errors.
    """
    if jb.b0.labels is None or jb.b1.labels is None:
        raise ValueError("No labels in binning result.")
    b0_q = quantize_from_labels(jb.b0.labels, jb.b0.center_by_label or {})
    b1_q_native = quantize_from_labels(jb.b1.labels, jb.b1.center_by_label or {})

    b0 = np.asarray(B0_hz).reshape(-1).astype(np.float32)
    b1 = np.asarray(B1).reshape(-1).astype(np.float32)

    # interpret B1 quantized back to same unit as input for comparison
    if b1_mode == "scale":
        if b1_ref is None:
            # match the binning ref if you used it externally; fallback to median
            finite = np.isfinite(b1)
            b1_ref = float(np.median(b1[finite])) if np.any(finite) else 1.0
            if b1_ref == 0.0: b1_ref = 1.0
        b1_q = b1_q_native * float(b1_ref)
    else:
        b1_q = b1_q_native

    # errors (ignore NaNs)
    def stats(err):
        e = err[np.isfinite(err)]
        if e.size == 0: return {}
        return dict(rmse=float(np.sqrt(np.mean(e**2))),
                    mae=float(np.mean(np.abs(e))),
                    p95=float(np.percentile(np.abs(e), 95)))
    b0_err = b0_q - b0
    b1_err = b1_q - b1

    out_stats = {"B0_Hz": stats(b0_err), "B1": stats(b1_err)}
    if not make_plots or plt is None:
        return out_stats

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = np.isfinite(b0) & np.isfinite(b0_q)
    ax.scatter(b0[m], b0_q[m], s=2)
    ax.set_xlabel("B0 original (Hz)")
    ax.set_ylabel("B0 quantized (Hz)")
    ax.set_title(f"{title_prefix}B0 quantization")

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    m = np.isfinite(b1) & np.isfinite(b1_q)
    ax.scatter(b1[m], b1_q[m], s=2)
    ax.set_xlabel("B1 original")
    ax.set_ylabel("B1 quantized")
    ## set x and y limits to show the same range around the ref (e.g. 1.0 for scale mode)
    if b1_mode == "scale":
        ref = float(b1_ref) if b1_ref is not None else 1.0
        r = max(np.abs(b1[m] - ref).max(), np.abs(b1_q[m] - ref).max())
        ax.set_xlim(ref - r*1.1, ref + r*1.1)
        ax.set_ylim(ref - r*1.1, ref + r*1.1)
    ax.set_title(f"{title_prefix}B1 quantization ({b1_mode})")

    fig3 = plt.figure()
    ax = fig3.add_subplot(111)
    e = b0_err[np.isfinite(b0_err)]
    ax.hist(e, bins=100)
    ax.set_title(f"{title_prefix}B0 error hist (RMSE={out_stats['B0_Hz'].get('rmse',np.nan):.3g} Hz)")
    ax.set_xlabel("Hz")

    fig4 = plt.figure()
    ax = fig4.add_subplot(111)
    e = b1_err[np.isfinite(b1_err)]
    ax.hist(e, bins=100)
    ax.set_title(f"{title_prefix}B1 error hist (RMSE={out_stats['B1'].get('rmse',np.nan):.3g})")
    ax.set_xlabel("B1 units")

    plt.show()
    
    return out_stats

    