
# MRTini : A cocktail of Bloch and EPG for fast and accurate MRI simulation
<img src="data/MRTini_logo.png" alt="Logo" width="800">

This repository contains the simulation workflow for MRTini.  


## Setup

MRTini can be run in two ways:

1. **Recommended:** VS Code Dev Container with Docker Compose.
2. **Manual/local:** Python 3.10 environment with local dependencies installed manually.

The devcontainer route is the most reproducible setup. It builds the MRTini development environment, installs the Python/Rust dependencies, installs the `mrstar` sequence-interpreter wheel, builds the Rust prepass module, and starts the gammaSTAR reconstruction server as a sibling Docker service.

---

## 1. Recommended setup: VS Code Dev Container

### Requirements

Install the following on the host machine:

* Docker Desktop
* VS Code
* VS Code extension: **Dev Containers**
* NVIDIA driver + NVIDIA Container Toolkit, if GPU/CUDA execution is required

The devcontainer uses:

* Python `3.10`
* PyTorch `2.8.0+cu129`
* Rust + `maturin`
* `mrstar` Linux wheel from `wheels/`
* gammaSTAR reconstruction server from `gammastar_recon/`

Expected repository layout:

```text
MRTini/
├── .devcontainer/
│   ├── devcontainer.json
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── post_create.sh
├── external/
│   └── gammastar_recon
|       └──docker/Dockerfile
├── src/
├── wheels/
│   └── mrstar-...-cp310-cp310-manylinux_2_34_x86_64.whl
├── requirements.txt
└── README.md
```

The `gammastar_recon/` directory should contain the vendored gammaSTAR reconstruction server source corresponding to the tested upstream commit. If the directory was copied from gammaSTAR, keep the upstream license/notice files and record the source commit in a metadata file such as `gammastar_recon/UPSTREAM_COMMIT.txt`.

### Start the devcontainer

Open the repository in VS Code and run:

```text
Ctrl+Shift+P → Dev Containers: Reopen in Container
```

VS Code will build and start two Docker services:

```text
mrtini-dev        # MRTini development container
gammastar-recon   # gammaSTAR reconstruction server
```

### Run the simulation inside the devcontainer
inside the devcontainer terminal, run the `sim_run.py` file
it will run the simulation and reconstruct using nufft and save 
the output in the `recons/` directory

everything should work out at this point, if you encounter any issues, please refer to the some extra infor and trouble shooting section below.

### Devcontainer Information
The MRTini container is configured with:

```text
PYTHONPATH=/workspaces/mrtini/src:/workspaces/mrtini
GAMMASTAR_RECON_HOST=gammastar-recon
GAMMASTAR_RECON_PORT=9002
```

So, inside the devcontainer, MRTini talks to the reconstruction server at:

```text
gammastar-recon:9002
```

Do **not** use `localhost:9002` from inside the devcontainer; inside Docker, `localhost` refers to the current container, not the reconstruction-server container.

### Rebuild the devcontainer

If dependencies or `.devcontainer/` files changed, rebuild with:

```text
Ctrl+Shift+P → Dev Containers: Rebuild Container
```

For a full clean rebuild:

```text
Ctrl+Shift+P → Dev Containers: Rebuild Container Without Cache
```

If the container opened but the post-create setup failed, the setup script can be rerun manually inside the devcontainer:

```bash
cd /workspaces/mrtini
bash .devcontainer/post_create.sh
```

Check the gammaSTAR recon server connection:

```bash
python - <<'PY'
import os
import socket

host = os.getenv("GAMMASTAR_RECON_HOST", "gammastar-recon")
port = int(os.getenv("GAMMASTAR_RECON_PORT", "9002"))

print(f"Testing gammaSTAR recon at {host}:{port}")

with socket.create_connection((host, port), timeout=5):
    print("gammaSTAR recon server is reachable.")
PY
```

---

## 2. Local setup without devcontainer ( skip this if using devcontainer )

The local setup is useful if you do not want to use the VS Code devcontainer but still want to run MRTini directly on your machine.

### Requirements

Install locally:

* Python `3.10`
* Rust/Cargo
* Docker Desktop, only if using gammaSTAR reconstruction
* NVIDIA driver, if using CUDA/GPU execution
* [gammaSTAR reconstruction server](https://github.com/FraunhoferMEVIS/gammaSTAR/tree/main/sequenceServer/gammastar_recon), if using gammaSTAR reconstruction

Create and activate a Python 3.10 environment using your preferred method.


### Install PyTorch

For CUDA 12.9 wheels:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu129 "torch==2.8.0+cu129"
```

If CUDA 12.9 wheels are not available on your platform, install a compatible PyTorch build from the official PyTorch selector and keep `torch` excluded from `requirements.txt` installation.

### Install Python requirements

Install the regular requirements while excluding `torch` and `mrpy_recon_tools`, since those are handled separately:

Linux/macOS:

```bash

python -m pip install -r requirements.txt
```

### Install the `mrstar` sequence-interpreter wheel

Linux:

```bash
python -m pip install --force-reinstall wheels/mrstar-*-cp310-*-manylinux*_x86_64.whl
```

Windows:

```powershell
python -m pip install --force-reinstall (Get-ChildItem wheels\mrstar-*-cp310-*-win_amd64.whl | Select-Object -First 1).FullName
```

### Build and install the Rust prepass

```bash
cd src/rust_prepass
maturin develop -r
cd ../..
```

This should build the Rust prepass module and install it into the current Python environment.

---

## 3. Running a simulation

Example:

```bash
python sim_run.py
```

When using paths inside the devcontainer, prefer Linux-style paths or `pathlib`:

```python
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1]

sequenceData_path = (
    REPO_SRC
    / "sequence"
    / "gstar_seqs"
    / "RARE_default_sequence_sequenceData.json"
)
```

Avoid Windows-style paths such as:

```python
r"src\sequence\gstar_seqs\RARE_default_sequence_sequenceData.json"
```

inside Linux containers.

---


## 4. Running the gammaSTAR reconstruction server for 3D/2D sequence reconstruction

MRTini can optionally send simulated acquisitions to a gammaSTAR reconstruction server. The server listens on port `9002`.

### Devcontainer mode

When using the devcontainer, the reconstruction server is started as a sibling service through `.devcontainer/docker-compose.yml`.

Inside the devcontainer, use:

```text
GAMMASTAR_RECON_HOST=gammastar-recon
GAMMASTAR_RECON_PORT=9002
```

This is already set by the devcontainer Compose file.

### Local mode

### pull in the gammaSTAR-recon source

run the `scripts/fetch_gammastar_recon.sh` script if on Linux/macOS and `scripts\fetch_gammastar_recon.ps1` if on Windows to pull in the gammaSTAR reconstruction server source into the `external/gammastar_recon/` directory.

### Install gammaSTAR `mrpy_recon_tools`

 `external/gammastar_recon/` is present in the repository, install the reconstruction tools from the vendored source:

```bash
python -m pip install external/gammastar_recon/mrpy_tools/reconstruction/mrpy_recon_tools
```

On Windows PowerShell:

```powershell
python -m pip install .\external\gammastar_recon\mrpy_tools\reconstruction\mrpy_recon_tools
```

For local Python runs, start the reconstruction server with Docker Compose:

```bash
docker compose -f docker/compose.gammastar-recon.yaml up --build
```

The local Python process should connect to:

```text
localhost:9002
```

---

## 4. Reconstruction host

The reconstruction host depends on where the Python process is running:

| Python process location                             |    Reconstruction host |   Port |
| --------------------------------------------------- | ---------------------: | -----: |
| Host machine, local Python                          |            `localhost` | `9002` |
| Devcontainer, recon service in same Compose network |      `gammastar-recon` | `9002` |
| Devcontainer, recon server manually started on host | `host.docker.internal` | `9002` |

MRTini should read these values from environment variables:

```text
GAMMASTAR_RECON_HOST
GAMMASTAR_RECON_PORT
```

Default local behavior should be:

```text
GAMMASTAR_RECON_HOST=localhost
GAMMASTAR_RECON_PORT=9002
```

---

## 6. Plotting inside the devcontainer

The devcontainer may run Matplotlib in a non-interactive backend depending on display forwarding. For reliable figure generation, save figures instead of relying on GUI popups:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3])

fig.savefig("out/figure.png", dpi=300, bbox_inches="tight")
plt.close(fig)
```

If using VS Code notebooks or Python `# %%` cells, inline plotting is usually more reliable than GUI `plt.show()` windows.

---

## 7. Troubleshooting

### `mrpy_recon_tools==1.7` cannot be installed from PyPI

`mrpy_recon_tools` is not installed from PyPI in this setup. It is installed from the gammaSTAR source tree:

```bash
python -m pip install gammastar_recon/mrpy_tools/reconstruction/mrpy_recon_tools
```

Therefore, `requirements.txt` should not rely on:

```text
mrpy_recon_tools==1.7
```

### `torch.cuda.is_available()` is `False`

Check that Docker has GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

If this fails, the issue is host-side NVIDIA/Docker configuration, not MRTini.

### Devcontainer cannot connect to gammaSTAR recon

Inside the devcontainer, test:

```bash
python - <<'PY'
import os
import socket

host = os.getenv("GAMMASTAR_RECON_HOST", "gammastar-recon")
port = int(os.getenv("GAMMASTAR_RECON_PORT", "9002"))

print(f"Testing {host}:{port}")
with socket.create_connection((host, port), timeout=5):
    print("OK")
PY
```

If this fails, check whether the reconstruction container is running:

```bash
docker ps
docker logs gammastar-recon
```

Run these Docker commands from the host terminal.

### `plt.show()` does not open a window

In a headless/container environment, Matplotlib may use a non-interactive backend. Save figures to disk or use VS Code notebooks/interactive cells.

### Re-run devcontainer setup

Inside the devcontainer:

```bash
cd /workspaces/mrtini
bash .devcontainer/post_create.sh
```

Or rebuild from VS Code:

```text
Ctrl+Shift+P → Dev Containers: Rebuild Container
```
## License

MRTini is licensed under the GNU Affero General Public License v3.0 only
(`AGPL-3.0-only`). See [LICENSE](LICENSE).

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


