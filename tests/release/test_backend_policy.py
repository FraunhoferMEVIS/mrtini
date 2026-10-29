# SPDX-License-Identifier: AGPL-3.0-or-later
import builtins
from types import SimpleNamespace as NS
import pytest
import torch
from src.mrtini.optimized.fast_graph import apply_rf

def test_missing_fused_cuda_backend_fails_before_slow_fallback(monkeypatch):
    # Only import availability/dispatch are controlled. No numerical solver is mocked.
    original=builtins.__import__
    def unavailable(name,*args,**kwargs):
        if name.endswith('triton_rf') or name=='triton':raise ModuleNotFoundError('Triton deliberately unavailable')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',unavailable)
    parent=NS(is_cuda=True,requires_grad=False,element_size=lambda:8)
    bank=NS(uniform=False,case_indices=[None],table=NS(requires_grad=False))
    plan=NS(children=10,parents=10,src=range(20))
    with pytest.raises(RuntimeError,match='will not silently fall back'):
        apply_rf(plan,bank,parent,torch.zeros(3),'auto')
