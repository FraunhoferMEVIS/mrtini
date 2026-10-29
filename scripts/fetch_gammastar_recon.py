#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch one gammaSTAR source layout on Windows/Linux; never copy/delete trees."""
from __future__ import annotations
import argparse, os, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO="https://github.com/FraunhoferMEVIS/gammaSTAR.git"
SUBDIR="sequenceServer/gammastar_recon"

def git(*args):
    return subprocess.run(["git",*map(str,args)],check=True,text=True,stdout=subprocess.PIPE).stdout.strip()

def fetch(root=ROOT,ref=None):
    root=Path(root);ref=ref or os.environ.get("GAMMASTAR_REF","main")
    external=root/"external";source=external/"gammaSTAR";external.mkdir(parents=True,exist_ok=True)
    if not (source/".git").exists():
        if source.exists() and any(source.iterdir()):raise RuntimeError(f"Refusing to overwrite a non-Git directory: {source}")
        git("clone","--filter=blob:none","--sparse",REPO,source)
    origin=git("-C",source,"remote","get-url","origin").removesuffix(".git").rstrip("/")
    if origin!=REPO.removesuffix(".git"):raise RuntimeError(f"Unexpected gammaSTAR origin: {origin}")
    if git("-C",source,"status","--porcelain","--untracked-files=no"):
        raise RuntimeError(f"gammaSTAR has local tracked changes; commit or stash them first: {source}")
    git("-C",source,"fetch","--depth=1","origin",ref)
    git("-C",source,"checkout","--detach","FETCH_HEAD")
    git("-C",source,"sparse-checkout","set",SUBDIR)
    recon=source/SUBDIR
    if not (recon/"docker/Dockerfile").is_file():raise RuntimeError(f"No reconstruction Dockerfile at {recon}")
    commit=git("-C",source,"rev-parse","HEAD")
    (external/"GAMMASTAR_RECON_COMMIT.txt").write_text(commit+"\n",encoding="utf-8")
    print(f"gammaSTAR reconstruction source: {recon}\nCommit: {commit}",flush=True)
    return recon

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--ref");args=p.parse_args(argv);fetch(ref=args.ref)

if __name__=="__main__":
    try:main()
    except (RuntimeError,subprocess.CalledProcessError) as e:raise SystemExit(str(e)) from e
