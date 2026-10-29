"""Run once beside YOUR validation.py: python install_cache.py

Adds saving/reuse to the supplied simple runner; preserves your imports,
phantom overrides, sequence loader and config. Stops if its anchors differ.
"""
from pathlib import Path

folder=Path(__file__).resolve().parent
target=folder/'validation.py'
text=target.read_text()
if 'persistent = RunCache' in text:
    raise SystemExit('Cache is already installed; no edits made.')

replacements=[
    ('import csv\n', 'import csv\nimport config as settings\nfrom result_cache import RunCache\n'),
    ('        ref, seconds = run_iso(seq, data)\n',
     '''        persistent = RunCache(out / 'raw' / name, path(case['path']), data,
                              tag=(getattr(settings, 'CACHE_TAG', 'v1'), DEVICE,
                                   str(torch.__version__)))
        seed = getattr(settings, 'ISO_SEED', 0)
        def simulate_iso():
            torch.manual_seed(seed)
            return run_iso(seq, data)
        ref, seconds = persistent.get('iso',
            dict(spins=ISO_SPIN_COUNT, spin_dist='z', perfect_spoiling=False, seed=seed),
            simulate_iso)
'''),
    ('                r = run_mrtini(seq,data,case["tbw"],eps,cap,grid,hard_rf,min_mag)\n',
     '''                r = persistent.get('mrtini',
                    dict(tbw=case['tbw'], epsilon=eps, max_states=cap,
                         rf_grid_oversampling=grid, hard_rf=hard_rf,
                         min_state_mag=min_mag, collect_kz_diag=True),
                    lambda: run_mrtini(seq,data,case['tbw'],eps,cap,grid,hard_rf,min_mag))
'''),
    ('        signal_figures(name,seq,ref,primary,centres,out)\n',
     '''        hard = (get(EPSILONS[-1],MAX_STATES,RF_GRID_OVERSAMPLING,hard_rf=True)
                if HARD_RF_CONTROL else None)
        persistent.save_plot_inputs(name,seq,ref,seconds,primary,
                                    sweeps,caps,grids,magnitudes,hard)
        signal_figures(name,seq,ref,primary,centres,out)
'''),
]
for old,new in replacements:
    if text.count(old)!=1:
        raise SystemExit(f'Cannot find a unique runner anchor:\n{old}\n'
                         'No edits made. Apply the four additions in INSTALL.md manually.')
    text=text.replace(old,new,1)

backup=folder/'validation.py.before_cache'
if backup.exists():
    raise SystemExit('Backup already exists; no edits made. Keep it or rename it before installing.')
backup.write_bytes(target.read_bytes())
target.write_text(text)
print('Updated validation.py; original saved as validation.py.before_cache')
