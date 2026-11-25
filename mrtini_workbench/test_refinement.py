"""Regression coverage for issues observed in the native v5 interface."""
import numpy as np
import torch
from research import history_rf
from physics import Protocol,reconstruct
from dataclasses import replace
import pytest


def step(H,relations,coeff):
    r=torch.tensor(relations);src=torch.zeros(len(r),dtype=torch.long);dst=torch.arange(len(r))
    return history_rf(H,src,dst,r,torch.tensor(coeff,dtype=torch.complex64),len(r))


def test_history_mechanisms_and_conjugation():
    H=torch.zeros((1,6),dtype=torch.complex64);H[0,0]=1
    F=step(H,[2],[.2+.4j]);assert F[0,1]!=0
    echo=step(F,[4],[1j]);torch.testing.assert_close(echo[0,2],1j*F[0,1].conj())
    Z=step(F,[1],[.5j]);assert Z[0,4]!=0
    stimulated=step(Z,[2],[.7]);torch.testing.assert_close(stimulated[0,5],.7*Z[0,4])
    train=step(echo,[4],[1]);torch.testing.assert_close(train[0,3],echo[0,2].conj())


def test_history_conserves_merged_complex_signal():
    rng=np.random.default_rng(7);r=torch.tensor(rng.integers(0,6,500));src=torch.tensor(rng.integers(0,25,500));dst=torch.tensor(rng.integers(0,16,500));H=torch.tensor(rng.normal(size=(25,6))+1j*rng.normal(size=(25,6)),dtype=torch.complex64);c=torch.tensor(rng.normal(size=500)+1j*rng.normal(size=500),dtype=torch.complex64)
    M=H.sum(1);ref=torch.zeros(16,dtype=M.dtype);ref.index_add_(0,dst,torch.where(r>=4,M[src].conj(),M[src])*c);out=history_rf(H,src,dst,r,c,16)
    torch.testing.assert_close(out.sum(1),ref,rtol=1e-5,atol=1e-5)


def test_auto_offgrid_uses_density_compensation(monkeypatch):
    # A mocked NUFFT adapter verifies dispatch and weighting; this is not a real NUFFT numerical test.
    import sys
    from types import SimpleNamespace as NS
    called=[]
    class Adj:
        def __init__(self,im_size):self.n=im_size[0]
        def to(self,dev):return self
        def __call__(self,inp,om):
            called.append(inp.detach().cpu().numpy());return torch.ones((1,inp.shape[1],self.n,self.n),dtype=inp.dtype,device=inp.device)
    def dcf(om,im_size):return torch.full((1,1,om.shape[-1]),2.,device=om.device)
    monkeypatch.setitem(sys.modules,'torchkbnufft',NS(KbNufftAdjoint=Adj,calc_density_compensation_function=dcf))
    p=Protocol(resolution=8);k=np.array([[.12,.24],[1.2,2.5]]);s=np.ones((2,1),complex)
    assert 'iterative DCF' in reconstruct(s,k,p)['method'];np.testing.assert_equal(called[-1],2)
    assert 'no DCF' in reconstruct(s,k,replace(p,recon='nufft'))['method'];np.testing.assert_equal(called[-1],1)


def test_default_analysis_selects_central_active_water_and_first_adc(tmp_path,monkeypatch):
    import server
    from types import SimpleNamespace as NS
    i='b'*32;d=tmp_path/'runs'/i;d.mkdir(parents=True)
    np.savez(d/'maps.npz',x_m=[-.01,0,.01],y_m=[-.01,0,.01],water_index=[[-1,-1,-1],[-1,3,4],[-1,-1,-1]],water_PD=[[0,0,0],[0,.8,1],[0,0,0]],fat_index=np.full((3,3),-1),fat_PD=np.zeros((3,3)))
    monkeypatch.setattr(server,'ROOT',tmp_path);monkeypatch.setattr(server,'runner',NS(engine=NS(steps=[NS(adc=[]),NS(adc=[0,1,2,3])])));monkeypatch.setattr(server,'analysis_job',i);monkeypatch.setattr(server,'active',None);monkeypatch.setattr(server,'jobs',{i:{'status':'done','result':{'preview':False}}})
    assert server.analysis_default(i)==dict(x=1,y=1,species=0,rep=1,adc=2,repetitions=2,adc_count=4)
