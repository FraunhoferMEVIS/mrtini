#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/mrtini
python scripts/setup_mrtini.py \
  --device "${MRTINI_DEVICE:-cuda}" --skip-torch --dev --recon-tools
python -m pytest -c tests/pytest.release.ini -m 'not cuda'
echo '[MRtini] Container ready. The integrated native engine and CPU release checks passed.'
