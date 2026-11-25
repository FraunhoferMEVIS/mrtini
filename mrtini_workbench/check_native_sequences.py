"""Run from the MRTini root to check the installed Sequence API before GPU work.
No CUDA needed for these construction and trajectory checks.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from native_sequences import build_native_sequence
from physics import Protocol

def main():
    for name,kw in [('flash',dict(tr_ms=35,te_ms=8,flip_deg=20)),('rare',dict(tr_ms=1800,te_ms=12,esp_ms=12,flip_deg=90)),('epi',dict(tr_ms=1500,te_ms=100,esp_ms=4.5,flip_deg=90,etl=16))]:
        p=Protocol(sequence=name,resolution=64,backend='native',native_edit=True,**kw)
        seq,d=build_native_sequence(p,return_design=True)
        k=seq.get_kspace(); k=k.detach().cpu().numpy() if hasattr(k,'detach') else np.asarray(k)
        if k.ndim!=2:raise RuntimeError(f'{name}: unsupported get_kspace shape {k.shape}')
        if k.shape[0]!=len(d.adc_times) and k.shape[1]==len(d.adc_times):k=k.T
        if k.shape[0]!=len(d.adc_times) or k.shape[1]<2:raise RuntimeError(f'{name}: unexpected native trajectory shape {k.shape}')
        err=float(np.max(np.abs(k[:,:2]-d.expected_k)))
        print(f'{name}: {len(seq)} RF-centred repetitions; {len(k)} ADCs; max native vs design xy difference {err:.6g} cycles/m (assuming physical xyz convention)')
        if not np.allclose(k[:,:2],d.expected_k,rtol=0,atol=.02):raise RuntimeError('Native get_kspace convention differs from the design. Inspect sequence.py; do not substitute a guessed trajectory.')
    print('Installed Sequence construction and native xy trajectory checks passed.')
if __name__=='__main__':main()
