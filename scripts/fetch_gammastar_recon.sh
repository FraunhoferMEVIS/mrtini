#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="$ROOT/external"

REPO="https://github.com/FraunhoferMEVIS/gammaSTAR.git"
GAMMASTAR_REF="${GAMMASTAR_REF:-main}"   # set to tested tag/commit/branch for release

SOURCE="$EXTERNAL/.gammaSTAR_source"
SUBDIR="sequenceServer/gammastar_recon"
DEST="$EXTERNAL/gammastar_recon"

mkdir -p "$EXTERNAL"

if [ ! -d "$SOURCE/.git" ]; then
  git clone --filter=blob:none --sparse "$REPO" "$SOURCE"
fi

git -C "$SOURCE" fetch origin "$GAMMASTAR_REF"
git -C "$SOURCE" checkout FETCH_HEAD
git -C "$SOURCE" sparse-checkout set "$SUBDIR"

rm -rf "$DEST"
mkdir -p "$DEST"
cp -a "$SOURCE/$SUBDIR/." "$DEST/"

echo "gammaSTAR recon ref:"
git -C "$SOURCE" rev-parse --short HEAD
echo "gammaSTAR recon available at:"
echo "$DEST"