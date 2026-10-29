"""Save actual backend returns as compressed NPZ; no backend or pickle needed to read.

Use a new CACHE_TAG after changing backend code or simulation conventions.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def save_result(filename, value):
    arrays = {}
    def pack(v):
        if hasattr(v, 'detach'):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            key = f'array_{len(arrays)}'
            arrays[key] = v
            return {'array': key}
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, complex):
            return {'complex': [v.real, v.imag]}
        if isinstance(v, dict):
            return {'dict': [[pack(k), pack(x)] for k, x in v.items()]}
        if isinstance(v, (list, tuple)):
            return {'sequence': [pack(x) for x in v], 'tuple': isinstance(v, tuple)}
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        raise TypeError(f'Cannot save {type(v).__name__}; convert it to an array or scalar')
    tree = json.dumps(pack(value))
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    # Save immediately and atomically; a failed plot cannot discard this run.
    fd, temporary = tempfile.mkstemp(dir=filename.parent, suffix='.npz')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, tree=np.asarray(tree), **arrays)
        os.replace(temporary, filename)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_result(filename):
    with np.load(filename, allow_pickle=False) as stored:
        def unpack(v):
            if not isinstance(v, dict):
                return v
            if 'array' in v:
                return stored[v['array']].copy()
            if 'complex' in v:
                return complex(*v['complex'])
            if 'dict' in v:
                return {unpack(k): unpack(x) for k, x in v['dict']}
            values = [unpack(x) for x in v['sequence']]
            return tuple(values) if v['tuple'] else values
        return unpack(json.loads(str(stored['tree'].item())))


def phantom_snapshot(data):
    """Record effective tissue/field parameters, including any user overrides."""
    result = {}
    for key in ('PD', 'T1', 'T2', 'T2dash', 'D', 'B0', 'B1', 'voxel_pos',
                'size', 'nyquist', 'coil_sens', 'species_id', 'chemical_shift', 'multi_slice'):
        if hasattr(data, key):
            v = getattr(data, key)
            result[key] = v.detach().cpu().numpy() if hasattr(v, 'detach') else v
    return result


class RunCache:
    def __init__(self, directory, sequence_file, data, tag):
        # Sequence content and effective phantom values determine the cache namespace.
        h = hashlib.sha256(Path(sequence_file).read_bytes())
        h.update(str(tag).encode())
        self.phantom = phantom_snapshot(data)
        for key, value in sorted(self.phantom.items()):
            a = np.asarray(value)
            if a.dtype.hasobject:
                raise TypeError(f'Phantom field {key} must be numeric, not an object array')
            h.update(key.encode()); h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
        self.directory = Path(directory) / h.hexdigest()[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.files = {}
        (self.directory / 'context.json').write_text(json.dumps(
            dict(sequence_file=str(Path(sequence_file).resolve()),
                 sequence_sha256=hashlib.sha256(Path(sequence_file).read_bytes()).hexdigest(),
                 cache_tag=tag),indent=2))

    def get(self, engine, options, simulate):
        key = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:20]
        filename = self.directory / f'{engine}_{key}.npz'
        if filename.exists():
            print(f'Load saved {engine}: {filename.name}')
            result = load_result(filename)
        else:
            result = simulate()
            if isinstance(result,dict):
                result['cache_options']=dict(engine=engine,**options)
            save_result(filename, result)
            print(f'Saved {engine}: {filename.name}')
        self.files[id(result)] = filename.name
        return result

    def save_plot_inputs(self, name, seq, reference, iso_seconds, primary,
                         sweeps, caps, grids, magnitudes, hard=None):
        counts = [int((rep.adc_usage > 0).sum().item()) for rep in seq]
        # Preserve native acquisition timing/gradient moments for later diagnostics.
        native = []
        elapsed = 0.0
        for rep_index, rep in enumerate(seq):
            adc = rep.adc_usage > 0
            t = rep.event_time.cumsum(0)
            native.append(dict(rep=rep_index, time_s=elapsed+t[adc],
                               k_from_rep_gradm=rep.gradm.cumsum(0)[adc],
                               adc_phase=rep.adc_phase[adc],
                               rf_angle=rep.pulse.angle, rf_phase=rep.pulse.phase))
            elapsed += float(rep.event_time.sum().item())
        save_result(self.directory / 'acquisition.npz',
                    dict(adc_counts=counts, native=native, phantom=self.phantom))
        # k_from_rep_gradm is parsed local gradient accumulation, NOT a refocused
        # coherence trajectory or a declaration that count//2 is the physical echo.
        selected = dict(primary=primary, sweeps=sweeps, caps=caps, grids=grids,
                        magnitudes=magnitudes, hard=hard)
        references = {k: ([self.files[id(r)] for r in v] if isinstance(v, list)
                           else self.files[id(v)] if v is not None else None)
                      for k, v in selected.items()}
        # Reference is separately saved once, without duplicating it in every MRTini file.
        save_result(self.directory / 'reference.npz', dict(signal=reference, seconds=iso_seconds))
        metadata = dict(name=name, adc_counts=counts, **references)
        (self.directory / 'plot_inputs.json').write_text(json.dumps(metadata, indent=2))
