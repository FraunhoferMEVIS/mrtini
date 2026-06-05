#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/gammaSTAR"
REPO="https://github.com/FraunhoferMEVIS/gammaSTAR.git"

# Pin this to the exact commit you tested for the paper.
GAMMASTAR_REF="${GAMMASTAR_REF:-main}"

if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --sparse "$REPO" "$DEST"
fi

git -C "$DEST" fetch origin "$GAMMASTAR_REF"
git -C "$DEST" checkout "$GAMMASTAR_REF"
git -C "$DEST" sparse-checkout set sequenceServer/gammastar_recon

echo "gammaSTAR recon available at:"
echo "$DEST/sequenceServer/gammastar_recon"