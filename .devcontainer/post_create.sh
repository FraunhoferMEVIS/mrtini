#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/mrtini

echo "[MRTini] Python:"
python --version

echo "[MRTini] Installing requirements, excluding torch and mrpy_recon_tools..."
awk '
  /^torch==/ { next }
  /^mrpy_recon_tools==/ { next }
  NF { print }
' requirements.txt > /tmp/requirements.mrtini.txt

python -m pip install -r /tmp/requirements.mrtini.txt

echo "[MRTini] Installing mrstar sequence-interpreter Linux wheel..."
MRSTAR_WHEEL="$(find wheels -name 'mrstar-*-cp310-*-manylinux*_x86_64.whl' | head -n 1 || true)"

if [ -z "$MRSTAR_WHEEL" ]; then
  echo "ERROR: No Linux cp310 mrstar wheel found in wheels/"
  echo "Expected something like:"
  echo "  wheels/mrstar-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl"
  exit 1
fi

python -m pip install --force-reinstall "$MRSTAR_WHEEL"
GAMMASTAR_RECON_DIR="/workspaces/mrtini/external/gammastar_recon"
MRPY_PROCESSING_TOOLS_DIR="$GAMMASTAR_RECON_DIR/mrpy_tools/reconstruction/mrpy_recon_tools"

if [ -d "$MRPY_PROCESSING_TOOLS_DIR" ]; then
  echo "[MRTini] Installing mrpy_processing_tools..."
  python -m pip install -e "$MRPY_PROCESSING_TOOLS_DIR"
fi

echo "[MRTini] Building and installing Rust prepass..."
cd /workspaces/mrtini/src/rust_prepass
maturin build --release
python -m pip install --force-reinstall target/wheels/*.whl

cd /workspaces/mrtini

echo "[MRTini] Sanity check..."
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

echo "[MRTini] Setup complete."