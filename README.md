# MRTini: A cocktail of Bloch and Phase Distribution Graphs (PDG) for fast and accurate MRI Pulse Sequence simulation

<img src="data/MRTini_logo.png" alt="MRTini logo" width="800">

MRTini includes slice-profile RF propagation, species-separated graph construction and accelerated graph execution in one Python/Rust installation. The native engine is built in release mode by pip. There is no separate optimization package or patch step.


## Quick start

Use Python **3.10** for the supplied `mrstar` wheels. Core-engine development also supports Python 3.11–3.12 when a sequence interpreter is not needed. For source installation, install Rust/Cargo first. Run commands below from the repository root, in your selected Python environment.

NVIDIA CUDA setup (Linux or native Windows x86-64):

```bash
python scripts/setup_mrtini.py --device cuda --dev --recon-tools
```

CPU setup:

```bash
python scripts/setup_mrtini.py --device cpu --dev
```

The setup script selects the official torch CPU/CUDA wheel, installs MRTini and its application requirements together, builds the existing Rust crate once, installs the compatible `mrstar` wheel from `wheels/`, and runs an installation check. `--recon-tools` additionally fetches/installs the gammaSTAR reconstruction helpers. On Windows the CUDA extra installs `triton-windows`; on Linux it installs Triton. The release baseline is torch 2.8.0, CUDA 12.9 and Triton 3.4.x. It does not install both Triton distributions on one platform.

For core-engine tests without the platform-specific sequence interpreter:

```bash
python scripts/setup_mrtini.py --device cpu --dev --skip-mrstar
```

`--dry-run` prints the installation plan without changing the environment. A container whose image already has the correct torch build uses `--skip-torch`. Previously installed standalone `mrtini-fast-pdg` / `mrtini-prepass-rs` prototype distributions are replaced by the integrated MRTini installation; restart Python afterwards.

The authoritative runtime dependencies are in `pyproject.toml`. Existing application requirements are retained in `requirements-application.txt` and included through `requirements.txt`. When adding a dependency later, update the package metadata and applicable requirements file together. Private wheels/local reconstruction tools remain explicit setup inputs rather than invented PyPI package versions.

## Recommended setup: VS Code Dev Container

Host requirements:

- Docker Desktop / Docker Engine and VS Code with the Dev Containers extension.
- Python 3.10+ available as `python`, and Git, for the cross-platform gammaSTAR preparation hook.
- For the CUDA configuration: an NVIDIA GPU/driver and Docker GPU access (NVIDIA Container Toolkit on a Linux host).

Open the repository and run **Dev Containers: Reopen in Container**. Select the CUDA configuration or the CPU configuration when prompted. The CPU alternative is `.devcontainer/cpu/devcontainer.json` and does not request GPU access.

Both configurations use a Python 3.10 virtual environment, an appropriate torch 2.8 build, Rust/Cargo caches, and the same `scripts/setup_mrtini.py` used locally. Post-create setup builds the integrated native engine and runs CPU release tests. CUDA setup also compiles and checks a small fused GPU RF kernel before declaring setup complete.

The initialization hook fetches gammaSTAR to one canonical path on every platform:

```text
external/gammaSTAR/sequenceServer/gammastar_recon
```

The reconstruction server is a sibling Compose service. Inside the MRTini container its endpoint is `gammastar-recon:9002`, configured through `GAMMASTAR_RECON_HOST` / `GAMMASTAR_RECON_PORT`. The host endpoint is `localhost:9002`. `localhost` inside the development container refers to that container itself.

Plotting defaults to **Agg** for headless operation. Save figures or use notebook inline plotting. GUI plotting is optional (`python -m pip install -e ".[gui]"`) and additionally needs a working display; it is not required to install or test the simulator.

To repeat setup in an already running container:

```bash
bash .devcontainer/post_create.sh
```

After changing dependencies or container files, use **Dev Containers: Rebuild Container**. Cargo/pip caches reduce repeated build work. The first CUDA RF call may include Triton JIT compilation; subsequent calls reuse its kernel cache.

## Simulation API

Keep sequence import, phantom preparation and RF-bank baking as before. With `seq`, `data` and baked RF artifacts `art`:

```python
from src.mrtini.simulation import compute_graph, execute_graph

graph, rep_keys = compute_graph(
    seq, data, art,
    max_state_count=NMAX,
    min_state_mag=1e-4,
    slice_thickness=SLICE_THICKNESS,
    per_species_tissue=True,
)
measured = execute_graph(
    graph, seq, data, rep_keys, art,
    SLICE_THICKNESS=SLICE_THICKNESS,
    rf_backend="auto",
    print_progress=True,
)
```

The existing `src` import prefix is retained to avoid a repository-wide package rename. The implementation lives inside MRTini. Existing callers of `compute_graph_fast` / `execute_graph_fast` remain supported through `src.mrtini.optimized`; the historical `mrtini_prepass_rs` import is included in the same wheel.

Water and fat use their own nominal RF taps, supports, coefficients, representative magnitudes and graph scores. No combined complex-tap envelope is used. `species="auto"` chooses species with nonzero PD. `per_species_tissue=True` uses separate tissue means for prepass scoring; the compatibility default remains `False` (the original whole-phantom means). Execution uses the actual voxel tissue maps either way. State caps remain per species and per state family, so the union can be larger than a water-only graph.

CPU execution uses fused native RF accumulation or matrix multiplication as appropriate. CUDA uses the matrix paths or the fused Triton RF kernel. If a general CUDA RF layer needs Triton and it is missing, execution stops with a setup error instead of silently taking a very slow path. `rf_backend="grouped"` remains an explicit portable alternative for diagnosis. Native/Triton RF kernels are inference paths; torch RF paths are selected for inputs requiring gradients. This does not make graph selection differentiable end to end.

The static-object path batches RF application, free evolution and ADC evaluation. Motion, profile callbacks, k-z diagnostics and magnetization-history requests use the retained state-loop path with accelerated RF. `vectorized=False` explicitly selects that path. RF-bank and dense-matrix setup can be reused via `FastRFContext` when graph/data/banks are unchanged.

To run your existing driver:

```bash
python sim_run.py
```

The driver must supply its sequence/phantom data as usual. Use `pathlib.Path` for portable filenames rather than Windows-only backslash paths inside Linux containers.

## Installation and numerical tests

The new release tests are additive; they do not replace your existing `tests/mrtini_pytest/` suite.

Installation/native smoke test:

```bash
python -m src.mrtini.doctor --kernel
```

CPU release checks:

```bash
python -m pytest -c tests/pytest.release.ini -m "not cuda"
```

On the actual CUDA machine, check the fused backend and complete complex ADC path:

```bash
python -m src.mrtini.doctor --require-cuda --kernel
python -m pytest -c tests/pytest.release.ini -m cuda
```

CUDA tests need real CUDA hardware and are reported as skipped when it is absent. A passing CPU job does not certify the CUDA kernel. The checks cover independent water/fat supports, species masks, conjugation, spatially varying phase/coefficient maps, empty RF rows, native-versus-independent prepass retention, and measured complex ADC comparisons against the supplied original executor and independent species runs.

The included CI job builds one wheel on Linux and Windows and tests it outside the checkout. It expects the corresponding Python 3.10 `mrstar` wheels in `wheels/`. The wheel-isolation check is enabled in CI through `MRTINI_TEST_INSTALLED_WHEEL=1`; it is normally skipped for an editable development install. GPU checks should additionally run on your cluster before release.

For your real sequence/phantom rather than the small synthetic regressions:

```python
from src.mrtini.simulation import benchmark_case
report, measured = benchmark_case(
    seq, data, art,
    max_state_count=NMAX,
    slice_thickness=SLICE_THICKNESS,
    compare_reference=True,
)
print(report)
```

This reports first/warm execution times and complex ADC differences. The explicit edge reference can be slow on a large graph. It supplements your isochromat validation.

## Optional gammaSTAR reconstruction

Fetch source and install reconstruction helpers with `--recon-tools`, or fetch/update the source separately:

```bash
python scripts/fetch_gammastar_recon.py
```

The `.sh` and `.ps1` wrappers call this same helper. It preserves local tracked modifications by refusing to overwrite a modified checkout, retains upstream licenses, and records the fetched commit in `external/GAMMASTAR_RECON_COMMIT.txt`. The default ref is `main`; set `GAMMASTAR_REF` to the tested full commit/tag for a pinned release. An existing checkout is reused during normal package setup.

For a local Python process, start the server from the host:

```bash
docker compose -f docker/compose.gammastar-recon.yaml up --build
```

For server logs in devcontainer mode, run from the host:

```bash
docker compose -f .devcontainer/docker-compose.yml logs gammastar-recon
```

`mrtini-doctor --mrstar --reconstruction` actually imports the requested optional tools; it does not report missing modules as OK.

## External Simulation Resources

### gammaSTAR Sequence Data

MRTini supports pulse sequences exported from the [gammaSTAR online framework](https://gammastar.mevis.fraunhofer.de/), developed at Fraunhofer MEVIS.

The raw sequence descriptions (`*_raw.json`) are included in this repository. However, the corresponding gammaSTAR `sequenceData` JSON files are excluded due to their large file sizes.

These files are required when using the gammaSTAR reconstruction server and can be obtained by configuring and exporting the corresponding sequences directly from the public gammaSTAR online framework.

The exported files should be placed in:

`src/sequence/gstar_seqs/`

### External Resources

- [gammaSTAR Online Framework](https://gammastar.mevis.fraunhofer.de/)
- [gammaSTAR – Fraunhofer MEVIS](https://www.mevis.fraunhofer.de/en/portfolio/solutions/gammastar.html)

## Troubleshooting

- **Stale native API:** restart Python after installation. Run the doctor and inspect its printed native-engine path. Install from the MRTini root, rather than building an old standalone extension elsewhere.
- **No matching mrstar wheel:** use Python 3.10 and the correct OS/architecture wheel. `--skip-mrstar` is intended for core-engine work, not sequence import.
- **CUDA unavailable:** check `torch.version.cuda`, `torch.cuda.is_available()` and host/Docker GPU access. CPU setup is available independently.
- **Missing/mismatched Triton:** rerun the CUDA setup in the same Python environment. For the release torch 2.8 baseline, Linux needs Triton 3.4.x and Windows needs `triton-windows` 3.4.x. Then run the real GPU kernel check.
- **Slow first repetition:** enable execution progress. Kernel compilation and initial RF-bank preparation occur before steady-state timings; the repetition counter is not an edge-level progress meter.
- **Qt/display error:** retain `MPLBACKEND=Agg` in headless runs and remove application-level hardcoded `QtAgg` overrides unless a display has been configured.
- **Reconstruction connection:** use `localhost:9002` from the host or `gammastar-recon:9002` inside the Compose network.

## License

MRTini is licensed under the GNU Affero General Public License v3.0 or later
(`AGPL-3.0-or-later`). See [LICENSE](LICENSE).

Copyright (C) 2026 Fraunhofer Institute for Digital Medicine MEVIS

Contributors:
- Snawar Hussain
- Daniel C. Hoinkiss
- Jörn Huber
- Vincent Kuhlen
- Matthias Günther

Contact:
- Daniel C. Hoinkiss: daniel.hoinkiss@mevis.fraunhofer.de
- Snawar Hussain: snawar.hussain@mevis.fraunhofer.de

## Citation

If you use MRTini in your research, please cite our paper:

**[Slice-Profile-Enabled Phase Distribution Graphs for MRI Simulation](https://arxiv.org/abs/2606.09233)**

<details>
<summary>Show BibTeX</summary>

```bibtex
@article{hussain2026sliceprofile,
  title   = {Slice-Profile-Enabled Phase Distribution Graphs for MRI Simulation},
  author  = {Snawar Hussain, Daniel C. Hoinkiss, J{\"o}rn Huber, Vincent Kuhlen, Matthias G{\"u}nther},
  journal = {arXiv preprint arXiv:2606.09233},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.09233},
  url     = {https://arxiv.org/abs/2606.09233}
}
```

</details>


