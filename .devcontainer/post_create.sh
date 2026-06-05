#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/mrtini

echo "[MRtini] Python:"
python --version

echo "[MRtini] Installing requirements, excluding torch and mrpy_recon_tools..."
awk '
  /^torch==/ { next }
  /^mrpy_recon_tools==/ { next }
  NF { print }
' requirements.txt > /tmp/requirements.mrtini.txt

python -m pip install -r /tmp/requirements.mrtini.txt

echo "[MRtini] Installing mrstar sequence-interpreter Linux wheel..."
MRSTAR_WHEEL="$(find wheels -name 'mrstar-*-cp310-*-manylinux*_x86_64.whl' | head -n 1 || true)"

if [ -z "$MRSTAR_WHEEL" ]; then
  echo "ERROR: No Linux cp310 mrstar wheel found in wheels/"
  echo "Expected something like:"
  echo "  wheels/mrstar-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl"
  exit 1
fi

python -m pip install --force-reinstall "$MRSTAR_WHEEL"

echo "[MRtini] Fetching gammaSTAR source inside container for mrpy_recon_tools..."
GAMMASTAR_DIR="/opt/gammaSTAR"

if [ ! -d "$GAMMASTAR_DIR/.git" ]; then
  git clone --filter=blob:none --sparse https://github.com/FraunhoferMEVIS/gammaSTAR.git "$GAMMASTAR_DIR"
fi

git -C "$GAMMASTAR_DIR" sparse-checkout set sequenceServer/gammastar_recon/mrpy_tools/reconstruction/mrpy_recon_tools
git -C "$GAMMASTAR_DIR" checkout main

MRPY_RECON_TOOLS_DIR="$GAMMASTAR_DIR/sequenceServer/gammastar_recon/mrpy_tools/reconstruction/mrpy_recon_tools"

if [ ! -d "$MRPY_RECON_TOOLS_DIR" ]; then
  echo "ERROR: mrpy_recon_tools source directory not found:"
  echo "$MRPY_RECON_TOOLS_DIR"
  find "$GAMMASTAR_DIR/sequenceServer/gammastar_recon" -maxdepth 5 -type d || true
  exit 1
fi

echo "[MRtini] Installing mrpy_recon_tools from gammaSTAR source..."
python -m pip install "$MRPY_RECON_TOOLS_DIR"

echo "[MRtini] Building and installing Rust prepass..."
cd /workspaces/mrtini/src/rust_prepass
maturin build --release
python -m pip install --force-reinstall target/wheels/*.whl

cd /workspaces/mrtini

echo "[MRtini] Installing MRtini editable package..."
python -m pip install -e .

echo "[MRtini] Sanity check..."
python - <<'PY'
import sys
import torch
import mrstar
import matplotlib

assert sys.version_info[:2] == (3, 10), sys.version

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("mrstar: OK")
print("mrpy_recon_tools: OK")
print("Matplotlib backend:", matplotlib.get_backend())
PY

echo "[MRtini] Setup complete."