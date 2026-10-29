# MRtini tests with independent numerical references

Replace the previous `tests/` directory with this directory. No production source
changes are included. Your uploaded API changes are retained: `execute_graph.py`,
`execute_graph()`, `bake_bloch_kernels()`, and the bank-based native bridge.

```bash
python -m pip install -r tests/requirements-test-mrtini.txt
python tests/run_mrtini_tests.py --fail-under 0
```

Start with a zero coverage threshold while investigating correctness failures.
The default runner still enforces 90% when `--fail-under` is omitted; no claim is
made that this revised suite reaches that threshold. Coverage cannot establish
physical correctness and does not measure the numerical independence of tests.
Reports are written under `tests/reports/mrtini/`.

## Independent checks

`mrtini_pytest/reference_math.py` imports NumPy and SciPy, never MRtini. It defines:

- Explicit Cartesian-to-normalized-complex basis transforms.
- Direct Fourier analysis/synthesis with physical sample coordinates.
- Proper-rotation checks, complex-unitarity checks, and six-channel completion.
- Exact piecewise-constant gradient-area integration, including split intervals.
- An adaptive DOP853 Bloch ODE reference for finite shaped RF and gradients.

Actual production outputs are compared with these references. Tests never call
MRtini's diagnostic, `gold`, legacy-basis, phase-fit, or quantization-check helpers.
The functions under test remain real production implementations.

The references have 12 self-tests, including deliberately wrong Fourier signs,
odd-grid shifts, reflection/scaling matrices, and wrong-side demodulation.
These self-tests validate the reference harness; they are not simulator tests.
A static audit test prevents reintroducing the removed checker calls.

## Explicit conventions

These conventions are part of the expected behavior, not inferred from the
production result and not tuned to make a test pass:

- Cartesian column vectors and active rotation `R = exp(+[omega]x dt)`.
- `m+ = (Mx+i My)/sqrt(2)`, `m- = (Mx-i My)/sqrt(2)`.
- Coefficient names specify source -> target:
  `pp=C00, pz=C20, mp=C01, mz=C21, zp=C02, zz=C22`.
- Fourier analysis uses `exp(-2*pi*i*k*z)`; synthesis uses the positive sign.
- A full N-point DFT grid has N distinct modes, with one even-grid Nyquist mode.
- The RF solver uses interval-start, zero-order-held controls on a uniform raster.
- Demodulation removes the pre/post gradient rotations on the corresponding
  right/left sides. Its Cartesian input/output layout is `(species,bank,3,3,z)`.
- RF carrier/FM is encoded in the complex RF envelope in the independent ODE;
  it is not also added as a detuning. B0 and species shifts enter the z field.

A different intentional representation must be specified and transformed
explicitly. Do not flip a reference sign merely because an assertion fails.
Discrete tap/convolution tests establish correctness on the stated finite,
periodic sample grid. They do not prove convergence to a continuous slice model.

## What changed

See `AUDIT.md` for the detailed audit. In particular:

- Removed diagnostic-print tests and legacy "gold" helper tests.
- Replaced MRtini-generated expected bank/tap values with independent transforms.
- Added complex Fourier phase, reconstruction, Parseval, odd/even, and nonzero
  coordinate-origin checks; odd-grid projection is no longer avoided.
- Replaced the malformed old demodulation test with a noncommuting frame-change
  test using asymmetric gradients and non-grid RF centers.
- Added shaped-RF solver comparisons against independently integrated Bloch ODEs.
- Replaced all-ones, nonphysical RF coefficient fixtures with an independently
  generated ideal 90-degree rotation for the executor tests.
- Removed the global `np.unique` workaround; production binning now sees the
  actual installed NumPy behavior.
- Historical regression checks now fail normally. No `xfail` decorators remain.

## Scope and limitations

The importer and native bridge use explicit boundary doubles; these verify
Python conversion contracts, not native code. Phantom type imports are isolated;
executor fixtures supply small deterministic numerical phantoms. UI/profiler
imports are isolated. Numerical production functions are not mocked.

Shape, serialization, caching, and argument-order tests are useful software
contracts but do not establish physical correctness. The revised suite still
does not validate the native graph builder, real gammaSTAR files, CUDA, scanner
measurements, full image reconstruction, diffusion comprehensively, or fat signal
handling. Those require separate integration and physics validation.

The uploaded coverage HTML contained source text from its report-generation run.
That text was reconstructed into a separate temporary tree for verification.
`audit_results/` records the resulting run and its source provenance. It is not
proof of the state of a release checkout edited after that report was generated.
The earlier 90%-plus reference reports and compiled caches are not republished.
