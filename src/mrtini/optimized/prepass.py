# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepare one native, species-separated arena prepass, not two Python calls."""
from types import SimpleNamespace
import torch
from .fast_graph import PackedGraph

RELATIONS = ('++', '+z', '-+', '-z', 'z+', 'zz')
COEFFICIENTS = ('pp', 'pz', 'mp', 'mz', 'zp', 'zz')


def build_inputs(seq, art):
    membership = getattr(seq, 'mrtini_spans_by_rep', None)
    if membership is None:
        spans = [i for i, s in enumerate(seq.mrtini_hybrid) if s['kind'] == 'bloch']
        if len(spans) != len(seq):
            raise ValueError('Need exactly one mapped Bloch RF span per repetition')
    else:
        if len(membership) != len(seq):
            raise ValueError('RF membership/repetition count mismatch')
        spans = []
        for indices in membership:
            rf = [int(i) for i in indices if seq.mrtini_hybrid[int(i)]['kind'] == 'bloch']
            if len(rf) != 1:
                raise ValueError('Need exactly one mapped Bloch RF span per repetition')
            spans.append(rf[0])
    rep_keys = [art.span_key[i] for i in spans]
    keys = list(dict.fromkeys(rep_keys))
    key_to_pid = {key: i for i, key in enumerate(keys)}
    pulse_id = torch.tensor([key_to_pid[key] for key in rep_keys], dtype=torch.int32)
    return rep_keys, keys, pulse_id


def _cpu_bank(p):
    t = p.nominal_taps
    values = {name: getattr(t, name).detach().to(device='cpu', dtype=torch.complex64).contiguous()
              for name in COEFFICIENTS}
    values.update(deltas=t.deltas.detach().to(device='cpu', dtype=torch.int32).contiguous(), dk=float(t.dk))
    support = {r: p.support[r].detach().to(device='cpu', dtype=torch.int32).contiguous() for r in RELATIONS}
    return SimpleNamespace(**values), support


def compute_graph_fast(seq, data, art, *, max_state_count=200, min_state_mag=1e-4,
                       slice_thickness=.005, species='auto', per_species_tissue=True):
    """Return (PackedGraph, rep_keys), ready for execute_graph_fast.

    species='auto' builds only species with nonzero PD. Explicit species may
    be (0,), (1,), or (0,1); masks refer to that ordered case list. No combined
    complex taps are ever constructed. T1/T2/T2dash are seconds, D is um^2/ms,
    matching the supplied native interface. z/Nz/pad belong to RF baking; they
    were unused arguments in the supplied Rust prepass and are not repeated.
    """
    import mrtini_prepass_rs as native
    if not hasattr(native, 'compute_graph_packed'):
        raise RuntimeError('Loaded a stale native engine. Reinstall MRTini from the repository root; no separate patch or fast package is required.')
    rep_keys, keys, pulse_id = build_inputs(seq, art)
    ids = getattr(data, 'species_id', torch.zeros_like(data.PD, dtype=torch.long)).detach().cpu().reshape(-1)
    pd = data.PD.detach().cpu().reshape(-1)
    if ids.numel() != pd.numel() or bool(((ids != 0) & (ids != 1)).any()):
        raise ValueError('Expected one species_id per voxel: 0 water, 1 fat')
    labels = [s for s in (0, 1) if bool(((ids == s) & (pd != 0)).any())] if species == 'auto' else list(species)
    if not labels:
        labels = [0]
    if labels not in ([0], [1], [0, 1]):
        raise ValueError('species must be auto, (0,), (1,), or (0,1)')
    if any(bool(((ids == s) & (pd != 0)).any()) for s in (0, 1) if s not in labels):
        raise ValueError('Requested graph omits a species with nonzero PD')
    taps, supports, tissues = [], [], []
    for label in labels:
        banks = []
        for key in keys:
            pulse = art.pulses[key]
            p = pulse.water if label == 0 or pulse.fat is None else pulse.fat
            banks.append(_cpu_bank(p))
        taps.append([b[0] for b in banks]); supports.append([b[1] for b in banks])
        means = []
        for name in ('T1', 'T2', 'T2dash', 'D'):
            a = getattr(data, name).detach().cpu()
            if per_species_tissue and a.numel() == ids.numel() and bool((ids == label).any()):
                a = a.reshape(-1)[ids == label]
            means.append(float(a.mean()))
            # means.append(float(a.min() if name == "D" else a.max()))
        tissues.append(tuple(means))
    payload = native.compute_graph_packed(
        list(seq), pulse_id, tissues, taps, supports, int(max_state_count), float(min_state_mag),
        tuple(map(float, data.nyquist.detach().cpu().reshape(-1))),
        tuple(map(float, data.size.detach().cpu().reshape(-1))),
        bool(seq.normalized_grads), float(slice_thickness))
    return PackedGraph(payload, case_species=labels), rep_keys
