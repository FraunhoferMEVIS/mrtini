"""Direct MRTini versus explicit isochromats. No adapters, manifest or NPZ handoff."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import sys

sys.path.append('./')
from config import (PHANTOM, SEQUENCES, ISO_SPIN_COUNT, EPSILONS, MAX_STATES,
                    STATE_CAPS, RF_GRID_OVERSAMPLING, RF_GRID_SWEEP,
                    MIN_STATE_MAG, MIN_STATE_MAG_SWEEP, DEVICE, OUTPUT,
                    FAT_REPLAY_FIXED, HARD_RF_CONTROL, MRTINI_ROOT, CACHE_TAG, ISO_SPIN_DIST, ISO_SEED)
from val_figures import (signal_figures, tap_figures, operator_figures, graph_figures,
                     sweep_figures, late_echo_sensitivity, hard_rf_figure,
                     min_magnitude_figure,
                     cross_sequence_figure)

HERE = Path(__file__).resolve().parent
ROOT = Path(MRTINI_ROOT).resolve()
if not (ROOT / "src/mrtini/mrtini_main.py").is_file():
    raise FileNotFoundError(f"Wrong MRtini root: {ROOT}")
sys.path.insert(0, str(ROOT))

from result_cache import RunCache

from src.voxelgridphantom.voxel_grid_phantom import VoxelGridPhantom
from src.voxelgridphantom.sim_data import SimData
from src.sequence.sequence import Sequence
from src.mrtini.mrtini_main import MRtiniSim
from src.isochromat_sim import isochromat_sim
from src.mrtini.pre_pass_rust_call import compute_graph_rust
from src.mrtini.execute_graph import execute_graph
from src.mrtini.RF_banks import build_imap
torch.manual_seed(ISO_SEED)
def path(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    if not p.is_file(): raise FileNotFoundError(p)
    return p


def load_phantom():
    # Single 1 mm isotropic voxel of nominal water at 3 T.
    phantom = VoxelGridPhantom(
        size=torch.tensor([1e-3, 1e-3, 5e-3]),
        PD=torch.ones(1, 1, 1),
        B0=torch.zeros(1, 1, 1),
        B1=torch.ones(1,1, 1, 1),
        T1=torch.full((1, 1, 1), 1.93),
        T2=torch.full((1, 1, 1), 0.099),
        T2dash=torch.full((1, 1, 1), 0.2),
        D=torch.full((1, 1, 1), 2.5e-9),
        coil_sens= torch.ones(1, 1, 1, 1),
    )
    phantom.chemical_shift = torch.zeros_like(phantom.PD)
    data = phantom.build(voxel_shape="box")
    return data.cuda() if DEVICE == "cuda" else data


def array(tensor):
    return np.asarray(tensor.detach().cpu().numpy())


def nrmse(signal, reference):
    a, b = np.asarray(signal), np.asarray(reference)
    if a.shape != b.shape: raise ValueError(f"ADC shape mismatch: {a.shape} / {b.shape}")
    return float(np.linalg.norm(a-b) / np.linalg.norm(b))


def echo_centres(seq):
    """One specified centre ADC sample for each rep that contains an ADC."""
    centre = []; offset = 0
    for rep in seq:
        count = int((rep.adc_usage > 0).sum().item())
        if count: centre.append(offset + count // 2)
        offset += count
    return np.asarray(centre, dtype=int)


def run_iso(seq, data):
    start = time.perf_counter()
    result = isochromat_sim(seq, data, spin_count=ISO_SPIN_COUNT, spin_dist="rand",
                            perfect_spoiling=False, print_progress=True)
    return array(result), time.perf_counter()-start


def pulse_diagnostics(pulse):
    water = pulse.water
    bank = water.bank
    bank.ensure_H6_bins()
    orders = array(water.nominal_taps.deltas)
    energy = sum(abs(array(v))**2 for v in water.nominal_taps.as_dict.values())
    kept = array(water.support['++']).astype(int)
    E = sum(abs(array(getattr(bank.H6_bank, key)))**2
            for key in ('pp', 'pz', 'mp', 'mz', 'zp', 'zz'))
    omitted = (E.sum(-1)-E[...,kept].sum(-1))/np.maximum(E.sum(-1), 1e-30)
    R = array(bank.Rxyz_bank)
    center = R[R.shape[0]//2,R.shape[1]//2]
    return dict(orders=orders, energy=energy, kept=kept,
                z_m=array(water.z_m),
                mxy=center[:,0,2]+1j*center[:,1,2], mz=center[:,2,2],
                omitted_by_bin=omitted, duration_s=water.t_rf,
                fat_support=(array(pulse.fat.support['++']).astype(int)
                             if pulse.fat is not None else None))


def run_mrtini(seq, data, tbw, epsilon, max_states=MAX_STATES,
               rf_grid_oversampling=RF_GRID_OVERSAMPLING, hard_rf=False,
               min_state_mag=MIN_STATE_MAG) -> dict[str, Any]:
    start = time.perf_counter()
    sim = MRtiniSim(seq=seq, device=DEVICE)
    art = sim.bake_bloch_kernels_from_sequence(
        phantom_data=data, device=DEVICE, tap_threshold=epsilon,
        global_RF=hard_rf)
    baked = time.perf_counter()

    # GammaSTAR importer: one RF/Bloch span per RF-to-RF repetition.
    rep_keys = []
    for indices in seq.mrtini_spans_by_rep:
        rf = [i for i in indices if seq.mrtini_hybrid[i]["kind"] == "bloch"]
        if len(rf) != 1: raise ValueError("Expected exactly one Bloch RF span per rep")
        rep_keys.append(art.span_key[rf[0]])
    keys = list(dict.fromkeys(rep_keys))
    ids = torch.tensor([keys.index(key) for key in rep_keys], dtype=torch.int32)
    taps = []; supports = []
    for key in keys:
        pulse = art.pulses[key]
        water = pulse.water
        if pulse.fat is None:
            taps.append(water.nominal_taps)
            supports.append(water.support)
        else:
            # Re-evaluate BOTH voxel maps on the same union support; replay
            # indexes fat coefficients through the water support map.
            union = torch.unique(torch.cat((water.support["++"],
                                            pulse.fat.support["++"])), sorted=True)
            for species in (water, pulse.fat):
                imap = build_imap(data.B0, data.B1, species.ax,
                                  device=DEVICE, clamp=True)
                species.vox6, species.supp_map = species.bank.eval_voxels_so3(imap, union)
                species.support = {rel: union for rel in species.support}
            taps.append(pulse.combined_taps)
            supports.append(water.support)

    pulses = [pulse_diagnostics(art.pulses[key]) for key in keys]
    slice_m = float(PHANTOM["slice_thickness_m"])
    rf_period_m = 1.0 / float(taps[0].dk)
    pad=rf_period_m / slice_m
    graph = compute_graph_rust_multi(
        list(seq), data, ids, taps, supports, art.z_m,
        max_state_count=max_states, min_state_mag=min_state_mag,)
    compiled = time.perf_counter()
    data.multi_slice = False
    
    signal, diag = execute_graph_anz(graph, seq, data, rep_keys, art,
                                    print_progress=True, collect_kz_diag=True,
                                    min_emitted_signal=0., min_latent_signal=0.,
                                    SLICE_THICKNESS=slice_m
                                    )
    finished = time.perf_counter()
    counts = [len(repetition) for repetition in graph[1:]]
    graph_info = {
        'plus': [sum(d.dist_type == '+' for d in rep) for rep in graph[1:]],
        'z': [sum(d.dist_type in ('z','z0') for d in rep) for rep in graph[1:]],
        'edges': [sum(len(d.ancestors) for d in rep) for rep in graph[1:]],
    }
    stats = {
        "rf_bank_s": baked-start, "graph_s": compiled-baked,
        "replay_s": finished-compiled, "total_s": finished-start,
        "peak_states": max(counts, default=0), "states_sum": sum(counts),
        "ancestry_edges": sum(graph_info['edges']),
        "unique_pulses": len(keys),
        "retained_taps": sum(len(s["++"]) for s in supports),
    }
    diag['measured_groups'] = {key: array(val)
                               for key,val in diag['measured_groups'].items()}
    return dict(signal=array(signal),stats=stats,pulses=pulses,
                graph=graph_info,kz_diag=diag,epsilon=epsilon,
                max_states=max_states,rf_grid_oversampling=rf_grid_oversampling,
                hard_rf=hard_rf,min_state_mag=min_state_mag)


def audit_sequence(case, seq, result, data):
    """Parser-level acquisition facts; inspect raw JSON for raster and spoiler units."""
    pulses = result['pulses']
    reps = list(seq)
    return dict(sequence=case['name'], file=str(path(case['path'])),
                tbw_from_config=case['tbw'], rf_repetitions=len(reps),
                adc_bearing_repetitions=sum(bool((r.adc_usage > 0).any()) for r in reps),
                adc_samples=sum(int((r.adc_usage > 0).sum()) for r in reps),
                rf_duration_s_by_unique_pulse=[p['duration_s'] for p in pulses],
                rf_z_samples_by_unique_pulse=[len(p['z_m']) for p in pulses],
                rf_offset_count_by_unique_pulse=[len(p['orders']) for p in pulses],
                rf_retained_offsets_by_unique_pulse=[len(p['kept']) for p in pulses],
                phantom_voxels=int(data.PD.numel()),
                phantom_species_ids=np.unique(array(data.species_id)).tolist(),
                phantom_B0_range=[float(array(data.B0).min()),float(array(data.B0).max())],
                phantom_B1_shape=list(data.B1.shape),
                phantom_size_m=array(data.size).tolist(),
                rf_raster_and_spoiler_units='verify GammaSTAR JSON and importer manually')


def run_all():
    configured = [case for case in SEQUENCES if case["path"]]
    if not configured: raise ValueError("Set one or more SEQUENCES[*]['path'] in config.py")
    out = HERE / OUTPUT; out.mkdir(parents=True, exist_ok=True)
    data = load_phantom()
    if bool(torch.any(data.species_id == 1).item()) and not FAT_REPLAY_FIXED:
        raise RuntimeError("execute_graph_analyze.py overwrites fat_mask with zero. "
                           "Fix that line before using a fat phantom, or use water-only SimData.")
    records, summary, audits = [], [], []
    for case in configured:
        name = case["name"]
        seq = Sequence.import_file_for_mrtini_sim(str(path(case["path"])))
        if DEVICE == "cuda": seq = seq.cuda()
        centres = echo_centres(seq)
        data = data.cuda() if DEVICE == "cuda" else data
        persistent = RunCache(
    out / "raw" / name, path(case["path"]), data,
                    tag=(CACHE_TAG, DEVICE, str(torch.__version__))
                )
        nyq = torch.as_tensor(data.nyquist, device=data.device)[:2]
        k = torch.zeros((5, 2), device=nyq.device, dtype=nyq.dtype)
        k[:, 0] = nyq[0] * nyq.new_tensor([0., .25, .5, 1., 1.5])
        native = data.dephasing_func(k, nyq).reshape(-1)
        box = torch.sinc(k / (2*nyq)).prod(-1)
        print("Columns: kx [cycles/m], native response, uniform-box response")
        print(torch.stack((k[:, 0], native, box), dim=1).detach().cpu())
        iso_result = persistent.get(
            "iso",
            dict(spins=ISO_SPIN_COUNT, spin_dist=ISO_SPIN_DIST,
                perfect_spoiling=False, seed=ISO_SEED),
            lambda: run_iso(seq, data)
        )
        if isinstance(iso_result, tuple):
            ref, seconds = iso_result
        else:
            ref, seconds = array(iso_result), 0.0
        records.append(dict(sequence=name,engine="isochromat",spins=ISO_SPIN_COUNT,
                            seconds=seconds,adc_samples=len(ref),echo_count=len(centres)))
        cache = {}
        def get(eps, cap, grid, hard_rf=False,
                min_mag=MIN_STATE_MAG) -> dict[str, Any]:
            key = (eps,cap,grid,hard_rf,min_mag)
            if key not in cache:
                r = cast(dict[str, Any], persistent.get(
                    "mrtini",
                    dict(tbw=case["tbw"], epsilon=eps, max_states=cap,
                        rf_grid_oversampling=grid, hard_rf=hard_rf,
                        min_state_mag=min_mag, collect_kz_diag=True),
                    lambda: run_mrtini(seq, data, case["tbw"], eps, cap,
                                    grid, hard_rf, min_mag)
                ))
                cache[key] = r
                records.append(dict(sequence=name,engine="mrtini",epsilon=eps,
                    max_states=cap,rf_grid_oversampling=grid,hard_rf=hard_rf,
                    min_state_mag=min_mag,
                    complex_nrmse=nrmse(r["signal"],ref),
                    seconds=r["stats"]["total_s"],adc_samples=len(r["signal"]),
                    echo_count=len(centres),**r["stats"]))
            return cache[key]
        sweeps = [get(e,MAX_STATES,RF_GRID_OVERSAMPLING) for e in EPSILONS]
        primary = sweeps[-1]
        audits.append(audit_sequence(case,seq,primary,data))
        caps = [get(EPSILONS[-1],cap,RF_GRID_OVERSAMPLING) for cap in STATE_CAPS]
        grids = [get(EPSILONS[-1],MAX_STATES,grid) for grid in RF_GRID_SWEEP]
        magnitudes = [get(EPSILONS[-1],MAX_STATES,RF_GRID_OVERSAMPLING,
                          min_mag=m) for m in MIN_STATE_MAG_SWEEP]
        signal_figures(name,seq,ref,primary,centres,out)
        tap_figures(name,primary,sweeps,out)
        operator_figures(name,primary,out)
        graph_figures(name,primary,centres,out)
        sweep_figures(name,ref,sweeps,caps,grids,out)
        late_echo_sensitivity(name,ref,centres,sweeps,caps,out)
        min_magnitude_figure(name,ref,magnitudes,out)
        if HARD_RF_CONTROL:
            hard = (get(EPSILONS[-1], MAX_STATES, RF_GRID_OVERSAMPLING, hard_rf=True)
            if HARD_RF_CONTROL else None)
            persistent.save_plot_inputs(
                name, seq, ref, seconds, primary,
                sweeps, caps, grids, magnitudes, hard
            )
            hard_rf_figure(name,ref,primary,hard,centres,out)
        summary.append(dict(sequence=name,seconds=primary["stats"]["total_s"],
                            complex_nrmse=nrmse(primary["signal"],ref),
                            echoes=len(centres),peak_states=primary["stats"]["peak_states"]))
        print(f"{name}: {len(centres)} ADC-bearing reps; {len(cache)} MRTini runs; "
              f"complex ADC NRMSE={summary[-1]['complex_nrmse']:.4g}")
    cross_sequence_figure(summary,out)
    (out / 'sequence_audit.json').write_text(json.dumps(audits,indent=2))
    fields = sorted({key for record in records for key in record})
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(records)
    (out / "settings.json").write_text(json.dumps({"phantom":PHANTOM,
        "sequences":configured,"iso_spins":ISO_SPIN_COUNT,"epsilons":EPSILONS,
        "state_caps":STATE_CAPS,"rf_grid_sweep":RF_GRID_SWEEP,
        "min_state_mag_sweep":MIN_STATE_MAG_SWEEP,"device":DEVICE,
        "hardware":torch.cuda.get_device_name() if DEVICE == 'cuda' else 'CPU',
        "pytorch":torch.__version__,
        "complex_nrmse":"||MRTini ADC - isochromat ADC||2 / ||isochromat ADC||2; all coils/samples; no scaling"
        }, indent=2))


if __name__ == "__main__": run_all()
