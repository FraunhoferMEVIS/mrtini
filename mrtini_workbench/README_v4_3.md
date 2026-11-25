# Workbench 4.3 — live design, reconstruction and DICOM reference

Extract into the repository root, replacing workbench files (keep local runs). Run
`python -m pip install -r mrtini_workbench/requirements.txt` in the MRTini environment.
Install `torchkbnufft` in the same PyTorch environment for non-Cartesian NUFFT.
Launch `python workbench.py`, open http://127.0.0.1:8765 and hard refresh.

Design: live preview is on by default. After a 240 ms pause following an edit or a
completed drop, the actual design compiler updates the complete first-shot strip
and sampled nominal full-acquisition trajectory. Previous valid plots remain
visible while compiling or on invalid drafts; Run design is enabled only for a
validated current revision. Manual Validate & preview remains available. Live
preview never invokes the GPU acquisition. Independent gradient axis duration
and offset are available under Parallel gradients; arbitrary simultaneous RF
and ADC scheduling is not supported. EPI and inward/outward spiral readouts are
available. Research waveforms are not hardware safety checked.

Reconstruction: in Execution choose Auto (FFT if all samples are on the
Cartesian grid, otherwise Kaiser–Bessel NUFFT), NUFFT adjoint, NUFFT + iterative
DCF, Direct Fourier reference, or legacy MRTini. The generated native sequence
trajectory is checked against the nominal design before simulation. Explicit
reconstruction uses xy cycles/m and the positive-phase forward convention;
FFT and direct reference agree numerically in synthetic tests. Images are
indexed [ky,kx]. Imported acquisitions keep legacy reconstruction in Auto
unless an explicit reconstruction FOV is entered and require that FOV for
explicit methods; verify that imported trajectory units are cycles/m. No
iterative inversion or implicit density compensation. The direct reference is
bounded to 200 million sample-pixel-coil products. NUFFT requires torchkbnufft;
no silent substitution if it is absent. Native CUDA sign/orientation integration
still needs verification against your installed simulator.

DICOM reference: open from Acquisition. Select local monochrome DICOM files,
choose series and slice/frame, and adjust window/level. The loopback Python
server decodes files in memory, does not persist them in run history and returns
no patient identity tags. Use de-identified research data; pixel data can still
contain burned-in identifiers. This is an independent image reference, not a
registered comparison or a DICOM export. Compressed images require appropriate
pydicom decoder plugins; frame-specific rescale and modality LUTs are rejected.

Validation: `cd mrtini_workbench && python -m pytest -q` — 63 passed, 2 skipped
(pydicom and torchkbnufft not installed in the test environment). Browser tests
using inline assets and an API bridge to the real FastAPI compiler covered
numeric edits, invalid timing recovery, synthetic drag/drop events, paused and
manual preview. No native CUDA device available. No Rust or Triton changes.
