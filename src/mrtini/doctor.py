# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installation checks; optional real CUDA compilation and algebra check."""
from __future__ import annotations
import argparse, importlib, importlib.metadata, json, platform, sys

TRITON_MINOR={"2.4":1,"2.5":1,"2.6":2,"2.7":3,"2.8":4,"2.9":5,
              "2.10":6,"2.11":6,"2.12":7,"2.13":7,"2.14":8,"2.15":9}

def expected_triton(torch_version):
    key=".".join(str(torch_version).split("+")[0].split(".")[:2])
    if key not in TRITON_MINOR:raise RuntimeError(f"No declared Triton compatibility for torch {torch_version}; use the release torch 2.8 baseline.")
    return f"3.{TRITON_MINOR[key]}"

def kernel_smoke(device="cuda"):
    import numpy as np
    import torch
    from types import SimpleNamespace as NS
    from .optimized.fast_graph import RFPlan,RFBank,apply_rf
    V=8;ids=torch.arange(V,device=device)%2
    def pulse(keys,shift):
        a=torch.arange(V*len(keys),dtype=torch.float32,device=device).reshape(V,len(keys))*.03+shift
        return NS(supp_map={k:j for j,k in enumerate(keys)},vox6=NS(**{n:torch.complex(a+(j+1)*.1,a*.2-j*.01) for j,n in enumerate(('pp','pz','zp','zz','mp','mz'))}))
    bank=RFBank(NS(water=pulse([0,2],.1),fat=pulse([1,2],.2)),ids,ids)
    plan=RFPlan([0,2,2,6],[0,1,0,1,0,1],[0,2,4,5,1,3],[0,1,2,1,0,2],[1,2,3,2,1,3],2)
    a=torch.arange(2*V,dtype=torch.float32,device=device).reshape(2,V)*.07
    parent=torch.complex(a,a*.2+.1);phase=torch.linspace(-.4,.5,V,device=device)
    reference=apply_rf(plan,bank,parent,phase,'reference')
    backend='triton' if device=='cuda' else 'native'
    print(f"[MRTini doctor] Checking {backend} RF kernel"+(' (first call includes JIT compilation)...' if device=='cuda' else '...'),flush=True)
    out=apply_rf(plan,bank,parent,phase,backend)
    if device=='cuda':torch.cuda.synchronize()
    torch.testing.assert_close(out,reference,atol=3e-5,rtol=3e-5)
    return {"backend":backend,"maximum_absolute_error":float((out-reference).abs().max()),
            "relative_error":float(torch.linalg.vector_norm(out-reference)/torch.linalg.vector_norm(reference))}

def check(*,require_cuda=False,kernel=False,mrstar=False,reconstruction=False):
    import torch
    import mrtini_prepass_rs as native
    missing=[n for n in ('compute_graph_multi','compute_graph_packed','apply_rf_cpu') if not hasattr(native,n)]
    if missing:raise RuntimeError(f"Loaded a stale native engine at {native.__file__}; missing {missing}. Reinstall MRTini from the repository root.")
    from .optimized import compute_graph_fast,execute_graph_fast
    info={"python":sys.version.split()[0],"platform":platform.platform(),"torch":torch.__version__,
          "torch_cuda":torch.version.cuda,"native_engine":getattr(native,'__native_file__',native.__file__),"cuda_available":torch.cuda.is_available()}
    try:info['mrtini_distribution']=importlib.metadata.version('MRTini')
    except importlib.metadata.PackageNotFoundError:raise RuntimeError("MRTini is not installed as a distribution. Run scripts/setup_mrtini.py.")
    if require_cuda:
        if not torch.cuda.is_available():raise RuntimeError("CUDA is unavailable to torch. Check the selected torch build, NVIDIA driver and Docker GPU access, or use --device cpu.")
        try:triton=importlib.import_module('triton')
        except ImportError as e:raise RuntimeError("The fused CUDA backend is unavailable. Install the MRTini [cuda] extra in the release torch 2.8 environment (Windows installs triton-windows).") from e
        expected=expected_triton(torch.__version__)
        if not str(triton.__version__).startswith(expected+'.'):raise RuntimeError(f"torch {torch.__version__} expects Triton {expected}.x; found {triton.__version__}")
        from .optimized.triton_rf import rf_triton
        info.update(triton=triton.__version__,gpu=torch.cuda.get_device_name(0))
    if mrstar:
        importlib.import_module('mrstar');info['mrstar']='imported'
    if reconstruction:
        for name in ('mrpy_ismrmrd_tools','mrpy_io_tools'):
            importlib.import_module(name);info[name]='imported'
    if kernel:info['rf_kernel_check']=kernel_smoke('cuda' if require_cuda else 'cpu')
    return info

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('require-cuda','kernel','mrstar','reconstruction'):p.add_argument('--'+flag,action='store_true')
    args=p.parse_args(argv)
    try:info=check(require_cuda=args.require_cuda,kernel=args.kernel,mrstar=args.mrstar,reconstruction=args.reconstruction)
    except (RuntimeError,ImportError,AssertionError) as e:print(f"[MRTini doctor] FAILED: {e}",file=sys.stderr,flush=True);return 1
    print(json.dumps(info,indent=2),flush=True);print('[MRTini doctor] OK',flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
