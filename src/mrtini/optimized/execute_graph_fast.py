# SPDX-License-Identifier: AGPL-3.0-or-later
"""RF-batched variant of the reviewed executor; free evolution/signal physics retained.
Generated from the supplied analysis executor, with per-edge RF tensor work removed.
"""
from __future__ import annotations
import math
import time as perf_clock
import torch
from collections import defaultdict
from .fast_graph import FastRFContext,PackedGraph
try:
    from ..event_profiles import EventProfiles, requested_profile_events, check_profile_event_times
except ImportError:
    EventProfiles = None
    def requested_profile_events(*args, **kwargs):
        raise RuntimeError("Profile callbacks require the project's event_profiles module")
    def check_profile_event_times(*args, **kwargs):
        raise RuntimeError("Profile callbacks require the project's event_profiles module")

def rigid_motion(voxel_pos, motion_func):
    def voxel_motion(time):
        rot, offset = motion_func(time)
        rot = rot.to(device=voxel_pos.device)
        offset = offset.to(device=voxel_pos.device)
        return torch.einsum("vi,eij->evj",voxel_pos,rot)+offset[:,None,:]
    return voxel_motion

def execute_graph_fast(graph: Graph,
                  seq: Sequence,
                  data: SimData,
                  rep_keys: list[str],
                  art: HybridArtifacts,
                  SLICE_THICKNESS: float = 0.005,
                  Nz: int = 256,
                  min_emitted_signal: float = 1e-4,
                  min_latent_signal: float = 1e-4,
                  print_progress: bool = True,
                  return_mag_p: int | bool | None = None,
                  return_mag_z: int | bool | None = None,
                  plot_sp: bool = False,
                  collect_kz_diag: bool = False,
                  diag_to_cpu: bool = True,
                  check_mixed_slice_order: bool = True,
                  profile_callback=None,
                  profile_events=None,
                  profile_z_offsets_m=None,
                  rf_backend="auto",
                  rf_context=None,
                  performance=None,
                  vectorized=True,
                  state_batch=16,
                  adc_batch=64,
                  voxel_batch=1024,
                  ) -> torch.Tensor | list:
    """
    Execute the phase graph.
    If collect_kz_diag=True, returns:
        measured, diag
    diag contains:
        rep_state_stats : per-repetition state statistics grouped by cumulative RF slice order
        rep_emit_stats  : per-repetition emitted-signal statistics grouped by cumulative RF slice order
        measured_groups : ADC signal grouped by cumulative RF slice-order class
    profile_callback receives repetition, event, stage="event", z_m[V,Nz]
    and M[V,Nz,3], before PD/coil weighting or receiver rotation.
    Default checkpoint: middle acquired ADC sample in each ADC-bearing rep.
    profile_events may be {rep_index: [event_index, ...]} (zero-based).
    profile_z_offsets_m supplies z offsets relative to each voxel centre.
    Profile reconstruction currently supports the static-object case.

    Important:
        dist.kt_vec is still the physical total k-vector used for signal physics.
        slice_order is tracked separately in Python sidecar dictionaries and is only diagnostic.
    """
    if (vectorized and not collect_kz_diag and not plot_sp and profile_callback is None
            and return_mag_p is None and return_mag_z is None
            and data.voxel_motion is None and data.phantom_motion is None):
        from .vectorized import execute_vectorized
        return execute_vectorized(graph,seq,data,rep_keys,art,
            slice_thickness=SLICE_THICKNESS,min_emitted_signal=min_emitted_signal,
            backend=rf_backend,context=rf_context,performance=performance,
            state_batch=state_batch,adc_batch=adc_batch,voxel_batch=voxel_batch,
            print_progress=print_progress)
    # ------------------------------------------------------------
    
    def emit_profile(*, repetition, event, stage, z_m, M):
        if profile_callback is not None:
            profile_callback(repetition=repetition, event=event, stage=stage,
                             z_m=z_m, M=M)
        if plot_sp:
            from utils.viz import plot_slice_profile
            import matplotlib.pyplot as plt
            # Interactive preview of voxel 0. Callback still receives all voxels.
            profile_voxel = int(data.voxel_pos.square().sum(-1).argmin().item())
            plot_slice_profile(z_m[profile_voxel], M[profile_voxel],
                title=f"Slice profile: rep {repetition+1}, event {event}")
            plt.show()

    # Diagnostic helpers
    # ------------------------------------------------------------
    def _new_state_stat():
        return {
            "count": 0,
            "plus_count": 0,
            "z_count": 0,
            "latent_sum": 0.0,
            "emitted_sum": 0.0,
            "mag_rms_sum": 0.0,
            "mag_rms_count": 0,
            "mixed_count": 0,
        }
    def _new_emit_stat():
        return {
            "dist_count": 0,
            "signal_l2_sum": 0.0,
            "signal_l1_sum": 0.0,
            "wz_abs_sum": 0.0,
            "wz_count": 0,
            "adc_kz_mean_sum": 0.0,
            "adc_kz_abs_mean_sum": 0.0,
        }
    def _to_cpu(x):
        return x.detach().cpu() if diag_to_cpu else x.detach()
    def _group_from_order(n: int) -> str:
        a = abs(int(n))
        if a == 0:
            return "kz0"
        elif a == 1:
            return "abs1"
        elif a <= 4:
            return "abs2_4"
        else:
            return "high"
    # ------------------------------------------------------------
    # Static setup
    # ------------------------------------------------------------
    t0 = 0
    D_scaled = -1e-9 * data.D
    ONE_THIRD_2PI2 = (1.0 / 3.0) * (2.0 * torch.pi) ** 2
    SQRT2 = 1.41421356237
    abs_T1 = torch.abs(data.T1)
    abs_T2 = torch.abs(data.T2)
    abs_T2dash = torch.abs(data.T2dash)
    zero_kt = torch.zeros(4, device=data.device)
    ez = torch.zeros(4, device=data.device)
    ez[2] = 1.0
    NEG_KT_REL = {"-+", "-z"}
    NEG_MAG_REL = {"-z", "-+"}
    # Python-side metadata for Rust/PyO3 graph objects.
    # Do not attach dist.slice_order directly to Rust objects.
    slice_order_map: dict[int, int] = {}
    slice_order_mixed_map: dict[int, bool] = {}
    def set_slice_order(dist, value: int, mixed: bool = False):
        slice_order_map[id(dist)] = int(value)
        slice_order_mixed_map[id(dist)] = bool(mixed)
    def get_slice_order(dist, default: int = 0) -> int:
        return int(slice_order_map.get(id(dist), default))
    def get_slice_order_mixed(dist) -> bool:
        return bool(slice_order_mixed_map.get(id(dist), False))
    def tap_delta_from_ancestor(ancestor, deltas) -> int:
        # ancestor[3] is the support/tap index.
        return int(deltas[ancestor[3]])
    def child_slice_order_from_ancestor(ancestor, deltas) -> int:
        relation, parent, _, _ = ancestor
        parent_order = get_slice_order(parent, 0)
        tap_delta = tap_delta_from_ancestor(ancestor, deltas)
        if relation in NEG_KT_REL:
            return -parent_order + tap_delta
        else:
            return parent_order + tap_delta
    voxel_pos_func = data.voxel_motion
    if voxel_pos_func is None and data.phantom_motion is not None:
        voxel_pos_func = rigid_motion(data.voxel_pos, data.phantom_motion)
    if seq.normalized_grads:
        grad_scale = 1 / data.size
    else:
        grad_scale = torch.ones_like(data.size)
    signal: list[torch.Tensor] = []
    diag = None
    if collect_kz_diag:
        diag = {
            "rep_state_stats": [],
            "rep_emit_stats": [],
            "signal_groups": {
                "kz0": [],
                "abs1": [],
                "abs2_4": [],
                "high": [],
                "all": [],
            },
            "meta": {
                "SLICE_THICKNESS": float(SLICE_THICKNESS),
                "min_emitted_signal": float(min_emitted_signal),
                "min_latent_signal": float(min_latent_signal),
                "multi_slice": bool(data.multi_slice),
                "group_definition": {
                    "kz0": "|slice_order| == 0",
                    "abs1": "|slice_order| == 1",
                    "abs2_4": "2 <= |slice_order| <= 4",
                    "high": "|slice_order| > 4",
                },
            },
        }
    # Proton density baked into coil sensitivity.
    # shape: voxels x coils
    coil_sensitivity = (
        data.coil_sens.t().to(torch.cfloat)
        * torch.abs(data.PD).unsqueeze(1)
    )
    coil_count = int(coil_sensitivity.shape[1])
    voxel_count = data.PD.numel()
    rf = rf_context or FastRFContext(graph, art, data, backend=rf_backend)
    if rf.graph is not graph or rf.art is not art or rf.data is not data:
        raise ValueError("RF context belongs to different graph/artifacts/data")
    rf_gpu_events = []
    free_gpu_events = []
    rf_dispatch_seconds = 0.0
    host_deltas = {}
    # Initial relaxed z0 state
    graph[0][0].mag = torch.ones(
        voxel_count, dtype=torch.cfloat, device=data.device
    )
    graph[0][0].kt_vec = torch.zeros(4, device=data.device)
    set_slice_order(graph[0][0], 0, mixed=False)
    mag_p = []
    mag_z = []
    # ------------------------------------------------------------
    # Repetition loop
    # ------------------------------------------------------------
    for i, (dists, rep) in enumerate(zip(graph[1:], seq)):
        if print_progress:
            print(f"\rCalculating repetition {i+1} / {len(seq)}", end="")
        key = rep_keys[i]
        vox6 = art.pulses[key].water.vox6
        vox6_fat = art.pulses[key].fat.vox6 if art.pulses[key].fat is not None else None
        supp_map = art.pulses[key].water.supp_map
        phase = torch.as_tensor(rep.pulse.phase, device=data.device)
        shim_array = torch.as_tensor(rep.pulse.shim_array, device=data.device)
        if key not in host_deltas:
            t = art.pulses[key].water.nominal_taps
            host_deltas[key] = (float(t.dk), t.deltas.detach().cpu().tolist())
        dkz, deltas = host_deltas[key]
        # 1Tx or pTx
        if shim_array.shape[0] == 1:
            B1 = data.B1.sum(0)
        else:
            assert shim_array.shape[0] == data.B1.shape[0]
            shim = shim_array[:, 0] * torch.exp(-1j * shim_array[:, 1])
            B1 = (data.B1 * shim[:, None]).sum(0)
        phase = phase + B1.angle()
        # RF phase factors
        phi = phase
        e_p = torch.exp(1j * phi)
        e_m = e_p.conj()
        e_2p = e_p * e_p
        rf_cpu_start = perf_clock.perf_counter()
        if performance is not None and data.PD.is_cuda:
            rf_begin, rf_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            rf_begin.record()
        rf_mags, rf_plan, rf_live = rf.layer(i, key, phase)
        rf_dispatch_seconds += perf_clock.perf_counter() - rf_cpu_start
        if performance is not None and data.PD.is_cuda:
            rf_end.record(); rf_gpu_events.append((rf_begin, rf_end))
            free_begin, free_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            free_begin.record()
        # ------------------------------------------------------------
        # Event / ADC setup
        # ------------------------------------------------------------
        adc = rep.adc_usage > 0
        adc_idx = adc.nonzero(as_tuple=True)[0]
        # rep.gradm[:, 2] = torch.zeros_like(rep.gradm[:, 2])  # remove kz from gradient trajectory, as it is already included in the kt_vec updates for slice profile
        trajectory = torch.cumsum(
            torch.cat(
                [rep.gradm * grad_scale[None, :], rep.event_time[:, None]],
                1
            ),
            0
        )
        trajectory_adc = trajectory.index_select(0, adc_idx)
        rep_sig = torch.zeros(
            adc_idx.numel(),
            coil_count,
            dtype=torch.cfloat,
            device=data.device
        )
        if collect_kz_diag:
            rep_state_stats = defaultdict(_new_state_stat)
            rep_emit_stats = defaultdict(_new_emit_stat)
            rep_sig_groups = {
                "kz0": torch.zeros_like(rep_sig),
                "abs1": torch.zeros_like(rep_sig),
                "abs2_4": torch.zeros_like(rep_sig),
                "high": torch.zeros_like(rep_sig),
                "all": torch.zeros_like(rep_sig),
            }
        dt = rep.event_time
        total_time = rep.event_time.sum()
        r1 = torch.exp(-total_time / abs_T1)
        r2 = torch.exp(-total_time / abs_T2)
        adc_rot = torch.exp(1j * rep.adc_phase).unsqueeze(1)
        adc_rot_adc = adc_rot.index_select(0, adc_idx)
        T2_adc = torch.exp(-trajectory_adc[:, 3:] / abs_T2)
        # Motion phase
        motion_phase = 0
        if voxel_pos_func is not None:
            time = t0 + torch.cat([torch.zeros(1, device=data.device), trajectory[:, 3]])
            voxel_traj = (
                voxel_pos_func((time[:-1] + time[1:]) / 2)
                - data.voxel_pos[None, :, :]
            )
            motion_phase = torch.einsum(
                "evi,ei->ev",
                voxel_traj,
                rep.gradm * grad_scale[None, :]
            ).cumsum(0)
        t0 += total_time
        mag_p_rep = []
        mag_p.append(mag_p_rep)
        mag_z_rep = []
        mag_z.append(mag_z_rep)
        # ------------------------------------------------------------
        # Distribution loop
        # ------------------------------------------------------------
        profile_acc = None
        if profile_callback is not None or plot_sp:
            events = requested_profile_events(rep, i, profile_events)
            check_profile_event_times(seq, rep, i, events)
            if events:
                profile_acc = EventProfiles(data, rep, i, trajectory, events,
                    nz=Nz, z_offsets_m=profile_z_offsets_m)
        for di, dist in enumerate(dists):
            first_ancestor = rf_plan.first_ancestor(di, graph[i], rf_live)
            if dist.dist_type != "z0" and first_ancestor is None:
                continue
            dist.mag = rf_mags[di]
            if dist.dist_type == "+" and return_mag_p in [i, True]:
                mag_p_rep.append(dist.mag)
            if dist.dist_type in ["z0", "z"] and return_mag_z in [i, True]:
                mag_z_rep.append(dist.mag)
            # --------------------------------------------------------
            # Update physical kt_vec and diagnostic slice_order
            # --------------------------------------------------------
            if dist.dist_type == "z0":
                dist.kt_vec = zero_kt
                set_slice_order(dist, 0, mixed=False)
            else:
                ancestor0 = first_ancestor
                relation0 = ancestor0[0]
                parent0 = ancestor0[1]
                tap_delta0 = tap_delta_from_ancestor(ancestor0, deltas)
                base_kt = parent0.kt_vec
                offset = dkz * tap_delta0
                # Physics: total k-vector. This is what goes into signal physics.
                if relation0 in NEG_KT_REL:
                    dist.kt_vec = -base_kt + offset * ez
                else:
                    dist.kt_vec = base_kt + offset * ez
                # Diagnostic: cumulative RF slice order.
                order0 = child_slice_order_from_ancestor(ancestor0, deltas)
                mixed = False
                if collect_kz_diag and check_mixed_slice_order:
                    candidate_orders = []
                    for anc in dist.ancestors:
                        if anc[1].mag is None:
                            continue
                        candidate_orders.append(child_slice_order_from_ancestor(anc, deltas))
                    mixed = len(candidate_orders) > 0 and len(set(candidate_orders)) > 1
                set_slice_order(dist, order0, mixed=mixed)
            # --------------------------------------------------------
            # State diagnostic after kt/order update
            # --------------------------------------------------------
            if collect_kz_diag and dist.dist_type != "z0":
                n = get_slice_order(dist, 0)
                st = rep_state_stats[n]
                st["count"] += 1
                if dist.dist_type == "+":
                    st["plus_count"] += 1
                elif dist.dist_type == "z":
                    st["z_count"] += 1
                st["latent_sum"] += float(dist.latent_signal)
                st["emitted_sum"] += float(dist.emitted_signal)
                if get_slice_order_mixed(dist):
                    st["mixed_count"] += 1
                if dist.mag is not None:
                    mag_rms = torch.sqrt(torch.mean(torch.abs(dist.mag.detach()) ** 2))
                    st["mag_rms_sum"] += float(mag_rms.cpu())
                    st["mag_rms_count"] += 1
            # --------------------------------------------------------
            # Transverse states
            # --------------------------------------------------------
            if dist.dist_type == "+":
                dist_traj = dist.kt_vec + trajectory
                k2 = dist_traj[:, :3]
                k1 = torch.empty_like(k2)
                k1[0, :] = dist.kt_vec[:3]
                k1[1:, :] = k2[:-1, :]
                b = ONE_THIRD_2PI2 * dt * (k1 * k1 + k1 * k2 + k2 * k2).sum(1)
                cb = torch.cumsum(b, 0)
                final_diffusion = torch.exp(D_scaled * cb[-1])
                if profile_acc is not None:
                    profile_acc.add_transverse(dist, cb)
                if dist.emitted_signal >= min_emitted_signal:
                    adc_dist_traj = dist.kt_vec + trajectory_adc
                    cb_adc = cb.index_select(0, adc_idx)
                    diffusion_adc = torch.exp(D_scaled * cb_adc[:, None])
                    if isinstance(motion_phase, torch.Tensor):
                        adc_motion_phase = motion_phase.index_select(0, adc_idx)
                    else:
                        adc_motion_phase = motion_phase
                    T2dash = torch.exp(
                        -torch.abs(adc_dist_traj[:, 3:]) / abs_T2dash
                    )
                    # Current RF-time B0 double-count correction.
                    # Keep this only if it matches your RF kernel timing convention.
                    t = adc_dist_traj[:, 3:]
                    # t_eff = torch.clamp(t - rep.event_time[0] * 2, min=0.0)
                    rot = torch.exp(2j * torch.pi * (
                        (t * data.B0)
                        + (adc_dist_traj[:, :3] @ data.voxel_pos.T)
                        + adc_motion_phase
                    ))
                    # In-plane voxel model
                    dephasing = data.dephasing_func(
                        adc_dist_traj[:, :2],
                        data.nyquist[:2]
                    )[:, None]
                    # 2D slice collapse. This uses physical total kz.
                    wz = None
                    if not data.multi_slice:
                        wz = torch.sinc(
                            adc_dist_traj[:, 2] * SLICE_THICKNESS
                        )[:, None]
                        dephasing = dephasing * wz
                    transverse_mag = (
                        SQRT2 * dist.mag
                        * rot
                        * T2_adc
                        * T2dash
                        * diffusion_adc
                        * dephasing
                    )
                    if isinstance(graph,PackedGraph):
                        allow=torch.tensor((graph.emitted[dist.index]>=min_emitted_signal).copy(),device=data.device)
                        transverse_mag=transverse_mag*allow[rf.ids][None,:]
                    dist_signal = transverse_mag @ coil_sensitivity
                    rep_sig += dist_signal
                    # Direct ADC contribution grouped by cumulative RF slice order.
                    if collect_kz_diag:
                        n = get_slice_order(dist, 0)
                        g = _group_from_order(n)
                        rep_sig_groups[g] += dist_signal
                        rep_sig_groups["all"] += dist_signal
                        es = rep_emit_stats[n]
                        es["dist_count"] += 1
                        es["signal_l2_sum"] += float(
                            torch.linalg.vector_norm(dist_signal.detach()).cpu()
                        )
                        es["signal_l1_sum"] += float(
                            dist_signal.detach().abs().sum().cpu()
                        )
                        es["adc_kz_mean_sum"] += float(
                            adc_dist_traj[:, 2].detach().mean().cpu()
                        )
                        es["adc_kz_abs_mean_sum"] += float(
                            adc_dist_traj[:, 2].detach().abs().mean().cpu()
                        )
                        if wz is not None:
                            es["wz_abs_sum"] += float(wz.detach().abs().mean().cpu())
                            es["wz_count"] += 1
                # Propagate transverse state to end of repetition
                dist.mag = dist.mag * r2 * final_diffusion
                if isinstance(motion_phase, torch.Tensor):
                    dist.mag = dist.mag * torch.exp(
                        2j * torch.pi * motion_phase[-1, :]
                    )
                dist.kt_vec = dist_traj[-1]
            # --------------------------------------------------------
            # Longitudinal states
            # --------------------------------------------------------
            else:
                if profile_acc is not None:
                    profile_acc.add_longitudinal(dist)
                k2_z = (dist.kt_vec[:3] * dist.kt_vec[:3]).sum()
                diffusion_z = torch.exp(D_scaled * total_time * k2_z)
                dist.mag = dist.mag * r1 * diffusion_z
                if dist.dist_type == "z0":
                    if isinstance(graph,PackedGraph):
                        active=(int(graph.node_mask[dist.index])>>rf.ids)&1
                        dist.mag=dist.mag+(1-r1)*active
                    else:dist.mag = dist.mag + 1 - r1
        # All states have contributed to each requested event profile.
        # The stored states may now be at rep end; the accumulated profiles are not.
        if profile_acc is not None:
            profile_acc.emit(emit_profile)
            if collect_kz_diag:
                diag.setdefault("profile_checkpoints", []).extend(profile_acc.diagnostics())
        # ADC phase
        rep_sig *= adc_rot_adc
        if collect_kz_diag:
            for g in rep_sig_groups:
                rep_sig_groups[g] *= adc_rot_adc
        # Save diagnostics before removing ancestor magnetization
        if collect_kz_diag:
            diag["rep_state_stats"].append({
                int(k): dict(v) for k, v in rep_state_stats.items()
            })
            diag["rep_emit_stats"].append({
                int(k): dict(v) for k, v in rep_emit_stats.items()
            })
            for g in rep_sig_groups:
                diag["signal_groups"][g].append(_to_cpu(rep_sig_groups[g]))
        # Every edge comes from the immediately preceding repetition.
        # Release each parent once rather than visiting it for every child edge.
        for parent in graph[i]:
            parent.mag = None
        signal.append(rep_sig)
        if performance is not None and data.PD.is_cuda:
            free_end.record(); free_gpu_events.append((free_begin, free_end))
    if print_progress:
        print(" - done")
    measured = torch.cat(signal)
    if performance is not None:
        performance["rf_cpu_dispatch_seconds"] = rf_dispatch_seconds
        performance["rf_backend"] = rf_backend
        performance["rf_banks_prepared"] = len(rf.banks)
        if data.PD.is_cuda:
            torch.cuda.synchronize(data.PD.device)
            performance["rf_cuda_seconds"] = sum(a.elapsed_time(b) for a,b in rf_gpu_events)/1000
            performance["free_evolution_and_signal_cuda_seconds"] = sum(a.elapsed_time(b) for a,b in free_gpu_events)/1000
    if collect_kz_diag:
        diag["measured_groups"] = {
            g: torch.cat(parts, dim=0)
            for g, parts in diag["signal_groups"].items()
        }
        del diag["signal_groups"]
        return measured, diag
    if return_mag_p is not None:
        if return_mag_z is not None:
            return measured, mag_p, mag_z
        else:
            return measured, mag_p
    elif return_mag_z is not None:
        return measured, mag_z
    else:
        return measured
