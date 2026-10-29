#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One user-facing setup command; the main MRtini install builds Rust once."""
from __future__ import annotations
import argparse, importlib.metadata, os, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH_BASELINE = "2.8.0"
TORCH_INDEX = {"cpu":"https://download.pytorch.org/whl/cpu",
               "cuda":"https://download.pytorch.org/whl/cu129"}

def run(args, *, cwd=ROOT, dry_run=False):
    print("[MRtini] " + subprocess.list2cmdline(list(map(str,args))),flush=True)
    if not dry_run:subprocess.run(list(map(str,args)),cwd=cwd,check=True)

def supported_platform(device, system=None, machine=None):
    system=system or platform.system();machine=(machine or platform.machine()).lower()
    if device=="cuda" and (system not in ("Linux","Windows") or machine not in ("x86_64","amd64")):
        raise RuntimeError("The CUDA release baseline supports Linux/Windows x86-64. Use --device cpu for this platform.")

def compatible_mrstar(wheels_dir):
    # Imported only after the main package has installed packaging.
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename, InvalidWheelFilename
    supported=list(sys_tags());rank={tag:i for i,tag in enumerate(supported)};choices=[]
    for path in sorted(Path(wheels_dir).glob("mrstar-*.whl")):
        try:name,version,_,tags=parse_wheel_filename(path.name)
        except InvalidWheelFilename:continue
        matches=[rank[t] for t in tags if t in rank]
        if name=="mrstar" and matches:choices.append((version,-min(matches),path.name,path))
    if not choices:
        raise RuntimeError(f"No compatible mrstar wheel in {wheels_dir} for Python {sys.version_info.major}.{sys.version_info.minor} on {platform.system()}/{platform.machine()}. The supplied release wheels target Python 3.10. Use --skip-mrstar only for core-engine tests.")
    return max(choices)[-1]

def project_requirement(root, dev, cuda):
    extras=(["dev"] if dev else [])+(["cuda"] if cuda else [])
    return str(root)+(f"[{','.join(extras)}]" if extras else "")

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--dev",action="store_true")
    p.add_argument("--skip-torch",action="store_true",help="Keep a preinstalled torch 2.8 build (e.g. the container image).")
    p.add_argument("--skip-mrstar",action="store_true",help="Core-engine development only; sequence interpreter will not be installed.")
    p.add_argument("--recon-tools",action="store_true",help="Fetch gammaSTAR and install reconstruction helpers from source.")
    p.add_argument("--torch-index",help="Override the official CPU/CUDA wheel index for a local mirror.")
    p.add_argument("--dry-run",action="store_true",help="Print commands without installation, imports or downloads.")
    return p

def main(argv=None):
    args=parser().parse_args(argv);supported_platform(args.device)
    if not (3,10)<=sys.version_info[:2]<(3,13):raise RuntimeError("Use Python 3.10–3.12; packaged mrstar wheels in the release target 3.10.")
    if not (ROOT/"src/rust_prepass/Cargo.toml").is_file():raise RuntimeError("Run this script from a complete MRtini checkout, after the maintainer has integrated the native sources.")
    if not (ROOT/"requirements-application.txt").is_file():raise RuntimeError("Missing requirements-application.txt. The maintainer must first integrate and commit the release files.")
    print(f"[MRtini] Interpreter: {sys.executable}\n[MRtini] Device: {args.device}",flush=True)
    pip=[sys.executable,"-m","pip"]
    if args.skip_torch and not args.dry_run:
        import torch
        if torch.__version__.split("+")[0]!=TORCH_BASELINE:raise RuntimeError(f"--skip-torch requires torch {TORCH_BASELINE}; found {torch.__version__}. Use the normal setup command to select the release baseline.")
        if args.device=="cuda" and torch.version.cuda is None:raise RuntimeError("--skip-torch selected with a CPU-only torch build. Install the CUDA build first.")
    elif not args.skip_torch:
        run(pip+["install",f"torch=={TORCH_BASELINE}{'+cu129' if args.device=='cuda' else ('+cpu' if platform.system() in ('Windows','Linux') else '')}","--index-url",args.torch_index or TORCH_INDEX[args.device]],dry_run=args.dry_run)
    # Remove only the superseded prototype distributions, before the single
    # integrated install. They own the same historical native import path.
    for distribution in ("mrtini-fast-pdg","mrtini-prepass-rs"):
        try:importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:continue
        run(pip+["uninstall","-y",distribution],dry_run=args.dry_run)
    run(pip+["install","-r",str(ROOT/"requirements.txt"),"-e",project_requirement(ROOT,args.dev,args.device=="cuda")],dry_run=args.dry_run)
    if not args.skip_mrstar:
        if args.dry_run:print("[MRtini] Select/install the compatible mrstar wheel from wheels/",flush=True)
        else:run(pip+["install",str(compatible_mrstar(ROOT/"wheels"))])
    if args.recon_tools:
        source=ROOT/"external/gammaSTAR/sequenceServer/gammastar_recon"
        if not source.is_dir():run([sys.executable,str(ROOT/"scripts/fetch_gammastar_recon.py")],dry_run=args.dry_run)
        recon=ROOT/"external/gammaSTAR/sequenceServer/gammastar_recon/mrpy_tools/reconstruction/mrpy_recon_tools"
        if not args.dry_run and not recon.is_dir():raise RuntimeError(f"Reconstruction helper source missing: {recon}")
        run(pip+["install",str(recon)],dry_run=args.dry_run)
    doctor=[sys.executable,"-m","src.mrtini.doctor","--kernel"]
    if args.device=="cuda":doctor.append("--require-cuda")
    if not args.skip_mrstar:doctor.append("--mrstar")
    if args.recon_tools:doctor.append("--reconstruction")
    run(doctor,dry_run=args.dry_run)
    print("[MRtini] Setup complete." if not args.dry_run else "[MRtini] Dry run complete; no installation performed.",flush=True)

if __name__=="__main__":
    try:main()
    except (RuntimeError,subprocess.CalledProcessError) as e:raise SystemExit(f"[MRtini] Setup failed: {e}") from e
