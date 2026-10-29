# SPDX-License-Identifier: AGPL-3.0-or-later
import json,time,unittest
from types import SimpleNamespace as NS
import numpy as np
import torch
from src.mrtini.optimized.fast_graph import RFPlan,RFBank,apply_rf,rf_grouped

def fixture(P=32,D=24,V=128,E=500,K=7,species=2,uniform=True,dtype=torch.complex64,seed=7,device="cpu"):
    rng=np.random.default_rng(seed);gen=torch.Generator().manual_seed(seed)
    def rand(*s):return torch.complex(torch.randn(*s,generator=gen),torch.randn(*s,generator=gen)).to(dtype=dtype,device=device)
    ids=torch.arange(V,device=device)%species
    def artifact(keys):
        vals={}
        for n in ['pp','pz','zp','zz','mp','mz']:
            x=rand(1 if uniform else V,len(keys));vals[n]=x.expand(V,-1).clone()
        return NS(vox6=NS(**vals),supp_map={int(k):j for j,k in enumerate(keys)})
    w=artifact(range(K));f=artifact(range(1,K-1)) if species==2 else None
    bank=RFBank(NS(water=w,fat=f),ids,ids)
    dest=np.sort(rng.integers(0,D,E));rows=np.r_[0,np.cumsum(np.bincount(dest,minlength=D))]
    plan=RFPlan(rows,rng.integers(0,P,E),rng.integers(0,6,E),rng.integers(0,K,E),rng.integers(1,1<<species,E),P)
    return plan,bank,rand(P,V),torch.randn(V,generator=gen,dtype=torch.float64 if dtype==torch.complex128 else torch.float32).to(device)

class Tests(unittest.TestCase):
    def test_dense(self):
        for species in [1,2]:
            for dtype in [torch.complex64,torch.complex128]:
                plan,bank,parent,phase=fixture(species=species,dtype=dtype)
                ref=apply_rf(plan,bank,parent,phase,'reference');out=apply_rf(plan,bank,parent,phase,'dense')
                torch.testing.assert_close(out,ref,rtol=3e-5 if dtype==torch.complex64 else 1e-12,atol=3e-5 if dtype==torch.complex64 else 1e-12)
    def test_grouped(self):
        for species in [1,2]:
            plan,bank,parent,phase=fixture(species=species,uniform=False)
            ref=apply_rf(plan,bank,parent,phase,'reference');out=apply_rf(plan,bank,parent,phase,'grouped')
            torch.testing.assert_close(out,ref,rtol=3e-5,atol=3e-5)
            torch.testing.assert_close(rf_grouped(plan,bank,parent,phase,chunk_voxels=17,chunk_edges=8),ref,rtol=3e-5,atol=3e-5)
    def test_sparse(self):
        for species in [1,2]:
            plan,bank,parent,phase=fixture(species=species)
            torch.testing.assert_close(apply_rf(plan,bank,parent,phase,'sparse'),apply_rf(plan,bank,parent,phase,'reference'),rtol=3e-5,atol=3e-5)
    def test_native_fused_cpu(self):
        for species in [1,2]:
            for uniform in [False,True]:
                plan,bank,parent,phase=fixture(species=species,uniform=uniform)
                torch.testing.assert_close(apply_rf(plan,bank,parent,phase,'native'),apply_rf(plan,bank,parent,phase,'reference'),rtol=3e-5,atol=3e-5)
    def test_empty_rows_and_scalar_phase(self):
        plan,bank,parent,_=fixture(D=100,E=20,species=2);phase=torch.tensor(.37)
        torch.testing.assert_close(apply_rf(plan,bank,parent,phase,'dense'),apply_rf(plan,bank,parent,phase,'reference'),rtol=3e-5,atol=3e-5)
    def test_no_dense_approximation(self):
        plan,bank,parent,phase=fixture(uniform=False)
        self.assertFalse(bank.uniform)
        with self.assertRaises(ValueError):apply_rf(plan,bank,parent,phase,'dense')
