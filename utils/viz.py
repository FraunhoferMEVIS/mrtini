# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 Fraunhofer MEVIS, Germany
#
# This file is part of MRTini.
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
def plot_slice_profiles(M_bins, z_m, *, bins=None, n_max=20,
                        title="Slice profiles per bin",
                        show_flip=False, save_path=None):
    """
    Plot |M_perp| and Mz vs z for up to n_max bins.

    M_bins : array-like, shape (B, Nz, 3) or (B, 3, Nz)
             magnetization for each bin across z (final state per z)
    z_m    : array-like, shape (Nz,)  z-grid (e.g., mm)
    bins   : optional iterable of bin indices to plot; if None, sample up to n_max spread across B
    n_max  : max number of bins to show (at least 20 if B >= 20)

    show_flip : if True, also overlays flip angle θ(z) = arccos(Mz) [deg] on a secondary y-axis
    save_path : if given, save the figure to this path
    """
    # --- convert to numpy and fix shapes ---
    def to_np(a):
        try:
            import torch
            if isinstance(a, torch.Tensor):
                return a.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(a)

    M = to_np(M_bins)
    z = to_np(z_m).reshape(-1)
    if M.ndim != 3:
        raise ValueError(f"M_bins must be 3D (B,Nz,3) or (B,3,Nz); got shape {M.shape}")

    B = M.shape[0]
    # detect layout
    if M.shape[2] == 3:          # (B, Nz, 3)
        Mx, My, Mz = M[..., 0], M[..., 1], M[..., 2]   # (B, Nz)
    elif M.shape[1] == 3:        # (B, 3, Nz) -> permute
        Mx, My, Mz = M[:, 0, :], M[:, 1, :], M[:, 2, :]
    else:
        raise ValueError("Last or second axis must have size 3 for (Mx,My,Mz).")

    Nz = Mz.shape[1]
    if z.shape[0] != Nz:
        raise ValueError(f"z_m length ({z.shape[0]}) != Nz ({Nz}).")

    # ensure z is increasing (for nicer plots)
    if z[0] > z[-1]:
        order = np.arange(Nz-1, -1, -1)
        z = z[order]
        Mx = Mx[:, order]
        My = My[:, order]
        Mz = Mz[:, order]

    Mperp = np.sqrt(Mx**2 + My**2)   # (B, Nz)

    # choose which bins to plot
    want = max(n_max, 20)  # ensure "at least 20"
    if bins is None:
        if B <= want:
            sel = np.arange(B)
        else:
            # spread indices approximately uniformly across [0, B)
            sel = np.unique(np.linspace(0, B-1, want, dtype=int))
    else:
        sel = np.asarray(list(bins), dtype=int)
        if sel.size > want:
            ## randomly sample want indices
            np.random.seed(0)
            np.random.shuffle(sel)
            
            sel = sel[:want]

    n = sel.size
    nrows = math.ceil(n / 4)
    ncols = min(4, n)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.2*nrows), squeeze=False)
    axes = axes.ravel()

    for k, b in enumerate(sel):
        ax = axes[k]
        ax.plot(z, Mperp[b], label='|M⊥|')
        ax.plot(z, Mz[b],    label='Mz', linestyle='--')

        # ax.set_title(f"bin {b} / {B} with off_res {np.round(bins.get(b, 0)*42.58*1e6)} Hz")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("magnetization (a.u.)")
        ax.grid(True, alpha=0.3)

        if show_flip:
            m = np.sqrt(Mx[b]**2 + My[b]**2 + Mz[b]**2)  # Magnitude of the magnetization vector
            flip_deg = np.degrees(np.arccos(Mz[b] / m))  # Flip angle in degrees
            phase_deg = np.degrees(np.arctan2(My[b], Mx[b]))  # Phase of the transverse magnetization

            # Overlay flip angle on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(z, flip_deg, alpha=0.4, label="Flip angle θ")
            ax2.set_ylabel("Flip angle θ (deg)", alpha=0.7)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            ax2.set_ylim(0, flip_deg.max()*1.1)

        if k == 0:
            ax.legend(loc="best")

    # hide any unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


##============== plot slice profiles for prepass debugging ================


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def plot_slice_profile(z_m, Mxyz=None, Mplus=None, *,
                             title="Slice profile",
                             unwrap_phase=False,
                             normalize_xy=False,
                             fill=True,
                             cmap="twilight",
                             linewidth=3.0):
    z = _np(z_m).reshape(-1)
    z_mm = z * 1e3

    if Mplus is None:
        M = _np(Mxyz)
        if M.shape[0] == 3 and M.shape[1] == z.size:
            M = M.T
        assert M.shape == (z.size, 3), f"Expected (Nz,3), got {M.shape}"
        Mx = np.real(M[:, 0]); My = np.real(M[:, 1]); Mz = np.real(M[:, 2])
        Mplus = (Mx + 1j * My) / math.sqrt(2.0)
    else:
        Mplus = _np(Mplus).reshape(-1)
        Mz = None if Mxyz is None else np.real(_np(Mxyz).reshape(-1, 3)[:, 2])

    Mxy = np.sqrt(2.0) * np.abs(Mplus)
    ph = np.angle(Mplus)
    if unwrap_phase:
        ph = np.unwrap(ph)
    if normalize_xy and Mxy.max() > 0:
        Mxy = Mxy / Mxy.max()

    if Mz is None:
        fig, ax_xy = plt.subplots(figsize=(8, 4), constrained_layout=True)
        axes = [ax_xy]
    else:
        fig, (ax_z, ax_xy) = plt.subplots(
            2, 1, figsize=(8, 5.6), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.2]},
            constrained_layout=True
        )
        axes = [ax_z, ax_xy]

        ax_z.plot(z_mm, Mz, color="0.15", lw=2.2)
        ax_z.axhline(0, color="0.8", lw=1.0)
        ax_z.set_ylabel(r"$M_z$")
        ax_z.set_ylim(min(-1.05, float(Mz.min()) - 0.03), max(1.05, float(Mz.max()) + 0.03))
        ax_z.spines["top"].set_visible(False)
        ax_z.spines["right"].set_visible(False)
        ax_z.grid(True, axis="x", alpha=0.12)
        ax_z.grid(False, axis="y")

    if fill:
        ax_xy.fill_between(z_mm, 0, Mxy, color="0.9", alpha=0.9, linewidth=0)

    pts = np.column_stack([z_mm, Mxy]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=linewidth)
    lc.set_array(ph[:-1])
    lc.set_clim(-np.pi, np.pi)
    ax_xy.add_collection(lc)

    ax_xy.axhline(0, color="0.8", lw=1.0)
    ax_xy.set_xlim(z_mm.min(), z_mm.max())
    ax_xy.set_ylim(-0.02, max(1.05, float(Mxy.max()) * 1.08))
    ax_xy.set_xlabel("z [mm]")
    ax_xy.set_ylabel(r"$|M_{xy}|$")
    ax_xy.spines["top"].set_visible(False)
    ax_xy.spines["right"].set_visible(False)
    ax_xy.grid(True, axis="x", alpha=0.12)
    ax_xy.grid(False, axis="y")

    axes[0].set_title(title, fontsize=16, pad=8)

    cax = inset_axes(ax_xy, width="34%", height="7%", loc="upper right", borderpad=1.0)
    cb = fig.colorbar(lc, cax=cax, orientation="horizontal")
    cb.set_label(r"phase($M_+$)", fontsize=9, labelpad=1)
    cb.set_ticks([-np.pi, 0, np.pi])
    cb.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    cb.outline.set_visible(False)

    return fig, axes


# =================== compare slice profiles for viz================
def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _to_components(z_m, Mxyz=None, Mplus=None, normalize_xy=False, unwrap_phase=False):
    z = _np(z_m).reshape(-1)
    if Mplus is None:
        M = _np(Mxyz)
        if M.shape[0] == 3 and M.shape[1] == z.size:
            M = M.T
        Mx, My, Mz = np.real(M[:,0]), np.real(M[:,1]), np.real(M[:,2])
        Mplus = (Mx + 1j*My) / math.sqrt(2.0)
    else:
        Mplus = _np(Mplus).reshape(-1)
        Mz = None if Mxyz is None else np.real(_np(Mxyz).reshape(-1, 3)[:,2])

    Mxy = np.sqrt(2.0) * np.abs(Mplus)
    ph = np.angle(Mplus)
    if unwrap_phase:
        ph = np.unwrap(ph)
    if normalize_xy:
        mx = max(Mxy.max(), 1e-12)
        Mxy = Mxy / mx
    return z*1e3, Mxy, ph, Mz

def plot_slice_profile_compare(
    z_mm,
    sim_Mxyz=None, sim_Mplus=None,
    bloch_Mxyz=None, bloch_Mplus=None,
    *,
    title="Slice-profile comparison",
    normalize_xy=False,
    unwrap_phase=False,
    show_error=True,
    linewidth=2.7,
):
    z_mm, sim_xy, sim_ph, sim_z = _to_components(
        z_mm, sim_Mxyz, sim_Mplus, normalize_xy, unwrap_phase
    )
    _, bloch_xy, bloch_ph, bloch_z = _to_components(
        z_mm, bloch_Mxyz, bloch_Mplus, normalize_xy, unwrap_phase
    )

    nrows = 4 if show_error else 3
    fig, axes = plt.subplots(
        nrows, 1, figsize=(8.8, 7.2 if show_error else 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15, 1.0, 0.75] if show_error else [1.0, 1.15, 1.0]},
        constrained_layout=True,
    )

    ax0, ax1, ax2 = axes[:3]

    # Mz
    if sim_z is not None and bloch_z is not None:
        ax0.plot(z_mm, bloch_z, lw=linewidth, label="Bloch")
        ax0.plot(z_mm, sim_z, lw=linewidth, ls="--", label="MRTini / PDG")
        ax0.set_ylabel(r"$M_z$")
        ax0.axhline(0, lw=1, color="0.82")
        ax0.legend(frameon=False, ncol=2, loc="best")
    else:
        ax0.axis("off")

    # |Mxy|
    ax1.plot(z_mm, bloch_xy, lw=linewidth, label="Bloch")
    ax1.plot(z_mm, sim_xy, lw=linewidth, ls="--", label="MRTini / PDG")
    ax1.fill_between(z_mm, 0, bloch_xy, alpha=0.12)
    ax1.set_ylabel(r"$|M_{xy}|$")
    ax1.axhline(0, lw=1, color="0.82")

    # phase
    ax2.plot(z_mm, bloch_ph, lw=linewidth, label="Bloch")
    ax2.plot(z_mm, sim_ph, lw=linewidth, ls="--", label="MRTini / PDG")
    ax2.set_ylabel(r"$\angle M_+$ [rad]")
    ax2.axhline(0, lw=1, color="0.82")
    ax2.set_ylim(-np.pi*1.08, np.pi*1.08)
    ax2.set_yticks([-np.pi, 0, np.pi])
    ax2.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

    if show_error:
        ax3 = axes[3]
        err_xy = sim_xy - bloch_xy
        ax3.plot(z_mm, err_xy, lw=2.2)
        ax3.axhline(0, lw=1, color="0.82")
        ax3.set_ylabel(r"$\Delta |M_{xy}|$")
        ax3.set_xlabel("z [mm]")
    else:
        axes[-1].set_xlabel("z [mm]")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.15)
        ax.grid(True, axis="y", alpha=0.08)

    axes[0].set_title(title, fontsize=15, pad=8)

    rms = np.sqrt(np.mean((sim_xy - bloch_xy)**2))
    maxerr = np.max(np.abs(sim_xy - bloch_xy))
    txt = rf"RMS $\Delta |M_{{xy}}|$ = {rms:.3g}" + "\n" + rf"max = {maxerr:.3g}"
    ax1.text(
        0.985, 0.92, txt,
        transform=ax1.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="gray", ec="0.85", alpha=0.92),
    )

    return fig, axes


def _phase_colored_line(ax, x, y, phase, *, cmap="twilight_shifted",
                        lw=3.0, alpha=1.0, label=None, zorder=3):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ph = 0.5 * (phase[:-1] + phase[1:])
    lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(-np.pi, np.pi))
    lc.set_array(ph)
    lc.set_linewidth(lw)
    lc.set_alpha(alpha)
    lc.set_zorder(zorder)
    ax.add_collection(lc)
    if label is not None:
        ax.plot([], [], color="0.2", lw=lw, label=label)
    return lc



def _colored_error_curve(ax, x, y, color_value, *, cmap, norm, lw=1.8, alpha=1.0):
    x = np.asarray(x); y = np.asarray(y); color_value = np.asarray(color_value)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(color_value)
    x, y, color_value = x[good], y[good], color_value[good]
    if len(x) < 2:
        return None

    pts = np.c_[x, y].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cval = 0.5 * (color_value[:-1] + color_value[1:])

    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw, alpha=alpha)
    lc.set_array(cval)
    ax.add_collection(lc)
    return lc

def _phase_line(ax, x, y, phase, *, cmap, norm, lw=3, ls="solid", alpha=1.0, zorder=3):
    x, y, phase = map(np.asarray, (x, y, phase))
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(phase)
    x, y, phase = x[good], y[good], phase[good]
    if len(x) < 2:
        return None
    pts = np.c_[x, y].reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ph = 0.5 * (phase[:-1] + phase[1:])
    lc = LineCollection(seg, cmap=cmap, norm=norm, linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(ph)
    lc.set_linestyle(ls)
    ax.add_collection(lc)
    return lc

def plot_slice_profile_comparison_2(
    z_m,
    mrtini_Mxyz,
    bloch_Mxyz,
    *,
    title="Slice profile",
    phase_threshold=0.02,
    normalize=True,
    save=None,
):
    z = np.asarray(z_m).reshape(-1) * 1e3
    A = np.asarray(mrtini_Mxyz)
    B = np.asarray(bloch_Mxyz)

    mxy_a = A[:,0] + 1j*A[:,1]
    mxy_b = B[:,0] + 1j*B[:,1]
    abs_a = np.abs(mxy_a)
    abs_b = np.abs(mxy_b)
    mz_a = A[:,2].real
    mz_b = B[:,2].real

    if normalize:
        s = max(abs_a.max(), abs_b.max(), np.abs(mz_a).max(), np.abs(mz_b).max(), 1e-12)
        abs_a, abs_b, mz_a, mz_b = abs_a/s, abs_b/s, mz_a/s, mz_b/s

    ph_a = np.angle(mxy_a)
    ph_b = np.angle(mxy_b)
    ph_a = np.where(abs_a > phase_threshold, ph_a, np.nan)
    ph_b = np.where(abs_b > phase_threshold, ph_b, np.nan)

    err_abs = np.abs(abs_a - abs_b)
    phase_diff = np.angle(mxy_a * np.conj(mxy_b))
    phase_diff = np.where((abs_a > phase_threshold) & (abs_b > phase_threshold), phase_diff, np.nan)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "text.color": "black",
        "axes.linewidth": 0.9,
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (axz, axxy) = plt.subplots(
        2, 1, figsize=(6.4, 3.1), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.08}
    )


    # Mz panel
    green = "#f7567c"
    green2 = "#ffc15e" 
    axz.plot(z, mz_a, color=green, lw=2.0, label=r"MRTini $M_z$")
    axz.plot(z, mz_b, color=green2, lw=2.0, ls=(0,(4,2)), alpha=0.85, label=r"Bloch $M_z$")
    axz.axhline(0, color="0.60", lw=0.8, ls="--")
    axz.set_ylabel(r"$M_z$")
    axz.set_ylim(min(0, np.nanmin([mz_a.min(), mz_b.min()]) - 0.05),
                 max(1.08, np.nanmax([mz_a.max(), mz_b.max()]) + 0.05))
    axz.grid(True, color="0.88", lw=0.55, ls="--")
    axz.legend(frameon=False, loc="center right")
    axz.set_title(title, pad=8)

    # Mxy panel: both curves phase-coded
    # lc_a = _phase_line(axxy, z, abs_a, ph_a, cmap=cmap, norm=norm, lw=3.2, alpha=1.0, zorder=4)
    # lc_b = _phase_line(axxy, z, abs_b, ph_b, cmap=cmap, norm=norm, lw=2.2, ls=(0,(4,2)), alpha=0.75, zorder=5)

    # Mxy panel: normal magnitude curves, no phase coding
    blue = "#5d576b"
    orange = "#f7567c"

    axxy.fill_between(z, 0, abs_a, color="0.78", alpha=0.22, zorder=1)
    axxy.plot(z, abs_a, color=blue, lw=2.7, label=r"MRTini $|M_{xy}|$", zorder=4)
    axxy.plot(z, abs_b, color=orange, lw=2.5, ls=(0, (4, 2)),
            label=r"Bloch $|M_{xy}|$", zorder=5)

    axxy.axhline(0, color="0.72", lw=0.8)
    axxy.set_xlabel("z position (mm)")
    axxy.set_ylabel(r"$|M_{xy}|$")
    axxy.set_ylim(-0.03, max(1.05, abs_a.max()*1.08, abs_b.max()*1.08))
    axxy.grid(True, color="0.88", lw=0.55, ls="--")
    axxy.legend(frameon=False, loc="upper left")

    # # Error inset in Mxy panel
    # axins = inset_axes(axxy, width="34%", height="34%", loc="upper right", borderpad=0.9)
    # axins.plot(z, err_abs, color="0.12", lw=1.3)
    # axins.fill_between(z, 0, err_abs, color="0.12", alpha=0.16)
    # axins.set_title(r"$\Delta |M_{xy}|$", fontsize=8, pad=2)
    # axins.grid(True, color="0.88", lw=0.45, ls="--")
    # axins.tick_params(labelsize=7, length=2)
    # axins.spines[["top","right"]].set_visible(False)
    
    # ---- error inset: error amplitude, phase-coded line ----
    axins = inset_axes(axxy, width="36%", height="34%", loc="upper right", borderpad=0.9)

    phase_diff = np.angle(mxy_a * np.conj(mxy_b))
    valid = (abs_a > phase_threshold) & (abs_b > phase_threshold)
    phase_diff = np.where(valid, phase_diff, np.nan)

    phase_clip = np.deg2rad(90.0)
    phase_show = np.clip(phase_diff, -phase_clip, phase_clip)

    phase_cmap = plt.get_cmap("twilight_shifted")
    phase_norm = plt.Normalize(-phase_clip, phase_clip)

    # faint fill under error
    axins.fill_between(z, 0, err_abs, color="0.25", alpha=0.12, zorder=1)

    # color-coded error curve by phase mismatch
    lc_err = _colored_error_curve(
        axins, z, err_abs, phase_show,
        cmap=phase_cmap, norm=phase_norm,
        lw=1.8, alpha=0.98
    )

    # optional subtle black outline so the curve remains readable
    axins.plot(z, err_abs, color="0.05", lw=0.55, alpha=0.45, zorder=0)

    axins.set_title(r"$\Delta |M_{xy}|$ colored by $\Delta\phi$", fontsize=7.2, pad=2)
    axins.grid(True, color="0.88", lw=0.35, ls="--")
    axins.tick_params(labelsize=6, length=2)
    axins.spines[["top", "right"]].set_visible(False)

    # deliberately relaxed scale so error does not look over-amplified
    err_max = np.nanmax(err_abs)
    axins.set_ylim(0, max(1e-6, err_max * 3.5))
    # axins.set_ylim(0, max(1e-6, 0.02))

    max_mag_err = np.nanmax(err_abs)
    max_phase_deg = np.rad2deg(np.nanmax(np.abs(phase_diff)))

    axins.text(
        0.98, 0.92,
        rf"max {max_mag_err:.2g}" + "\n" + rf"$\Delta\phi$ {max_phase_deg:.1f}°",
        transform=axins.transAxes,
        ha="right", va="top",
        fontsize=6.2,
        color="0.20"
    )

    # tiny phase scale, compact, inside inset
    cax = inset_axes(axins, width="30%", height="7%", loc="upper left", borderpad=0.75)
    cb = plt.colorbar(lc_err, cax=cax, orientation="horizontal")
    cb.set_ticks([-phase_clip, 0, phase_clip])
    cb.set_ticklabels(ticklabels=[r"$-90^\circ$", "0", r"$90^\circ$"])
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.45)
        

    for ax in (axz, axxy):
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.02)

    fig.align_ylabels([axz, axxy])

    if save:
        fig.savefig(save, dpi=400, bbox_inches="tight")
        # if save.endswith(".png"):
        #     fig.savefig(save[:-4] + ".pdf", bbox_inches="tight")

    return fig, (axz, axxy)



