#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/mrtini
python scripts/setup_mrtini.py \
  --device "${MRTINI_DEVICE:-cuda}" --skip-torch --dev --recon-tools
python tests/run_mrtini_tests.py
echo '[MRTini] Container ready. The integrated native engine and CPU release checks passed.'
