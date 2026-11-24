# MRtini Workbench v2 — native sequence generation

Extract the archive into MRtini_release beside `sim_run.py`. No existing simulator source files are replaced. Use your existing CUDA environment and optimized v3.1 (or compatible later version).

```sh
python -m pip install -r mrtini_workbench/requirements.txt
python mrtini_workbench/check_native_sequences.py
python workbench.py
```

Open http://127.0.0.1:8765. The app binds only to localhost. Native mode is selected automatically if source files are found. Built-in native sequence generation is selected by default; disable it to import an existing gammaSTAR JSON preset unchanged. The optional `--adapter` external JSON recipe from v1 is retained. No pypulseq dependency is required.

## Native implementation

`native_sequences.py` extends the actual `native_sequence()` and `make_design()` conventions recovered from the October 6 thesis package `chapter03_mini_sequences_dropin.zip` (`mini_sequences.py`). The same API also appeared in the native SMS and receiver rebuttal studies. It creates `Sequence(normalized_grads=False, mrtini_hybrid=[], mrtini_rep_bounds=..., mrtini_spans_by_rep=[])`, sets RF usage, phase, angle, waveform, RF centre, and full centre-to-centre physical gradient moments. The 10 us held RF raster, Hamming-windowed sinc shape, Hz/m gradient units, RF amplitude in Hz and ADC at event ends follow that source. RF duration is rounded to the RF raster; actual duration is exported. No JSON schema is guessed. The normal MRtini Bloch RF baking, graph prepass, optimized GPU execution and `_reconstruct` are used. No new graph pruning rule or precision reduction. Slice thickness is shared across the generated RF, phantom, graph and prepared executor.

- FLASH: one excitation and one readout per TR, linear ky, 117-degree quadratic RF spoiling, slice rewinder and post-readout spoiler.
- RARE: excitation plus one finite refocusing pulse per echo, CPMG phase, matched slice crushers, x/y rewinds, linear ky. TE of first echo equals echo spacing; centre k-space occurs at the corresponding echo in the train (not necessarily first TE).
- SE-EPI: interleaved multishot, one excitation/refocusing pair per shot, selectable lines per shot, alternating x readout, ky blips and post-shot spoiler. TE refers to the kx=0 sample on the middle readout of each shot. Defaults: TE 100 ms, ESP 4.5 ms, 16 lines per shot. The requested 64/128/256 matrix sets both acquisition and phantom sampling for generated sequences. Imported presets retain their own ADC matrix.

These are newly designed research acquisitions, not reproductions of the shipped scanner protocols. They use held rectangular gradient lobes with 1 ms prephasers/rewinders/crushers, not hardware slew-limited ramps. No scanner hardware constraints or SAR model. Invalid RF/ADC/gradient/TR timing is rejected; no hidden TE/TR adjustment. Encoding FOV changes gradient moments; the physical BrainWeb extent is not rescaled. Reconstruction uses the existing native function, without guessing its normalization. Independent image display scaling does not support quantitative intensity comparison; use exported complex data. Native mode selects a BrainWeb-compatible NPZ from data/ and samples 31 slices, then selects one. No native PD map export. Preview is a separately labelled approximate CPU model, not MRtini; it does not model finite RF, diffusion, full pathways or intra-readout evolution.

## Verification

`python -m unittest discover -s mrtini_workbench -p test_native_sequences.py -v` checks the design against an API double, not the installed simulator: all three sequences at 64/128/256, exact nominal RF integral, full ADC count, independent primary-coherence k-space recursion, nominal kz=0, ADC times, parameter changes, and timing error rejection. `check_native_sequences.py` checks the actual installed CPU Sequence constructor and get_kspace trajectory. Native mode also rejects mismatched trajectory/ADC counts instead of silently using an alternative trajectory. These assume the recovered physical xy convention; a local mismatch needs inspection of sequence.py. CUDA, real Bloch RF baking and full native acquisition/reconstruction have not been executed here. For scientific use, compare native ADC amplitude and phase and images against the reference executor for each sequence and independent Bloch references as appropriate.

## Performance and export

Identical configurations reuse the prepared engine; changed configurations rebuild it. CUDA Graph replay is opt-in. The dynamic Triton kernel cache persists as supported by Triton, but cache hit counts are not instrumented. Stage timings reflect actual work; CUDA event times may include idle intervals. Native progress is printed to the server terminal, browser displays stage progress. Result NPZ contains complex ADC signal, native k-space, complex reconstruction, settings, metadata, timings, and for built-in native sequences expected xy trajectory and ADC times. Runs are serialized; files remain under mrtini_workbench/runs/. Do not expose the server to the network. Research software only.
