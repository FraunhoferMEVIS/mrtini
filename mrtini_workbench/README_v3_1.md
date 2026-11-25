# Workbench 3.1

Extract into the MRTini repository root, replacing the v3 workbench files. Restart `python workbench.py`. No simulator source changes or new dependencies.

Simulation model in the protocol inspector (native mode only): Order-local RF sends `global_RF=True` to the existing Bloch bake API; off sends `False` (default, slice-profile enabled). State budget presets are 100, 200, 300, 400, 600, 800, plus editable integer values 1–20000. The value passes directly as `max_state_count`; it is the prepass cap, not a guarantee of the union size or total nodes. Lower caps can change the numerical signal. Both controls invalidate prepared engine/graph, not phantom cache. Protocol and exported native metadata record the RF mode and cap.

Sequence: paper-style color-coded RF/Gx/Gy/Gz/ADC, independent signed vertical scales, phase-referenced signed RF and optional original real/imaginary components, flip | phase annotations, event-average gradients as continuous held steps, cursor inspection, active/full window and repetition navigation. The plot depicts the commanded sequence, not the effective order-local RF approximation. Imported sequences still display RF markers where envelopes are unavailable. Default window is two RF repetitions.

Offline tests: `cd mrtini_workbench` then `python -m pytest -q test_native_sequences.py test_phantom_cache.py test_analysis_views.py test_model_controls.py` (31 passed). JavaScript syntax checks passed. Native CUDA and visual browser verification unavailable in this environment.
