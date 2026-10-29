#!/usr/bin/env bash
set -euo pipefail
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$script_root/fetch_gammastar_recon.py" "$@"
