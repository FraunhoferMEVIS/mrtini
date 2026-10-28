# Verification of the revised suite

- Independent reference and static-audit self-tests: 12 passed.
- Executor tests with physical ideal RF coefficients: 21 passed.
- Full suite against source recovered from the uploaded coverage HTML: 198 passed, 17 failed, 16 warnings.
- Combined statement/branch coverage on that captured source: 84.37%.
- No xfail markers and no NumPy numerical monkeypatches.

## Remaining failures in that captured source

| Group | Count | Evidence |
| --- | ---: | --- |
| Legacy Bloch RHS versus active-rotation convention | 2 | Cross-product sign differs from the SO(3)/spinor generator convention. This does not establish that the legacy RHS is used by the current solver. |
| Finite shaped RF at a non-midpoint center | 2 | Direct demodulation checks and midpoint ODE checks pass; solve_slice omits rf_center_time at its demodulation call. |
| Historical metadata/event/cache behavior | 8 | Ordinary regression failures, previously XFAIL. |
| Odd-grid baked projection | 3 | Direct complex DFT mismatch for each normalization. |
| Even-grid Fourier modes | 1 | N+1 modes include both equivalent Nyquist representatives instead of N unique DFT modes. |
| Spatially varying RF taps on an odd grid | 1 | Independent complex tap/reconstruction check exposes the same projection permutation issue. |

The 17 failures are not 17 independent production bugs; parameterizations and
multiple checks can expose the same problem. These observations apply to the
source captured in the report. They do not establish the state of later edits.

The source calls check_unitarity_Rch, check_demod, check_rotations, gold6_from_Rxyz,
legacy basis helpers, phase-fit helpers, and sanity_check_quantization are not
invoked by any revised test. Core production algorithms are still called as the
subjects under test. Removing unsupported diagnostic tests lowers coverage;
coverage is reported as measured and is not used as a correctness oracle.
