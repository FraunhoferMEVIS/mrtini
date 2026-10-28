"""python replot.py --raw output/raw --out output_replot

No MRTini, MRzero or GPU imports. Replots the most recently selected runs
in every saved context; add --name rare_tbw52 to select a sequence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from result_cache import load_result
from figures import (signal_figures, tap_figures, operator_figures, graph_figures,
                     sweep_figures, late_echo_sensitivity, hard_rf_figure,
                     min_magnitude_figure, cross_sequence_figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='output/raw')
    parser.add_argument('--out', default='output_replot')
    parser.add_argument('--name')
    args = parser.parse_args()
    found = sorted(Path(args.raw).rglob('plot_inputs.json'))
    if not found:
        raise SystemExit('No saved plot_inputs.json. Run validation.py with caching installed first.')
    summaries=[]
    for filename in found:
        meta = json.loads(filename.read_text())
        name = meta['name']
        if args.name and name != args.name:
            continue
        folder = filename.parent
        out = Path(args.out) / name / folder.name
        out.mkdir(parents=True, exist_ok=True)
        def read(key):
            value = meta[key]
            if isinstance(value, list):
                return [load_result(folder / item) for item in value]
            return load_result(folder / value) if value else None
        ref = load_result(folder / 'reference.npz')['signal']
        primary = read('primary')
        sweeps, caps, grids, mags = [read(k) for k in ('sweeps', 'caps', 'grids', 'magnitudes')]
        counts = np.asarray(meta['adc_counts'], dtype=int)
        offsets = np.r_[0, np.cumsum(counts[:-1])]
        centres = (offsets+counts//2)[counts > 0]
        signal_figures(name, counts, ref, primary, centres, out)
        tap_figures(name, primary, sweeps, out)
        operator_figures(name, primary, out)
        graph_figures(name, primary, centres, out)
        sweep_figures(name, ref, sweeps, caps, grids, out)
        late_echo_sensitivity(name, ref, centres, sweeps, caps, out)
        min_magnitude_figure(name, ref, mags, out)
        hard = read('hard')
        if hard is not None:
            hard_rf_figure(name, ref, primary, hard, centres, out)
        summaries.append(dict(sequence=f'{name} ({folder.name[:6]})',
                              seconds=primary['stats']['total_s'],
                              complex_nrmse=float(np.linalg.norm(primary['signal']-ref)/np.linalg.norm(ref)),
                              echoes=len(centres),peak_states=primary['stats']['peak_states']))
        print(f'Replotted {name}: {out}')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    cross_sequence_figure(summaries,root)


if __name__ == '__main__':
    main()
