#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="$ROOT/external"

REPO="https://github.com/FraunhoferMEVIS/gammaSTAR.git"
GAMMASTAR_REF="${GAMMASTAR_REF:-main}"

SOURCE="$EXTERNAL/gammaSTAR"
SUBDIR="sequenceServer/gammastar_recon"
RECON_DIR="$SOURCE/$SUBDIR"

mkdir -p "$EXTERNAL"

if [ ! -d "$SOURCE/.git" ]; then
  git clone --filter=blob:none --sparse "$REPO" "$SOURCE"
fi

git -C "$SOURCE" fetch origin "$GAMMASTAR_REF"
git -C "$SOURCE" checkout FETCH_HEAD
git -C "$SOURCE" sparse-checkout set "$SUBDIR"

if [ ! -d "$RECON_DIR" ]; then
  echo "ERROR: gammaSTAR recon directory not found:"
  echo "$RECON_DIR"
  exit 1
fi

echo "gammaSTAR recon ref:"
git -C "$SOURCE" rev-parse --short HEAD
echo "gammaSTAR recon available at:"
echo "$RECON_DIR"