import json
from types import SimpleNamespace as NS
from dataclasses import replace
import numpy as np
import pytest
import torch
from physics import Protocol
from native_sequences import design,build_native_sequence
from test_native_sequences import SequenceDouble,Usage
from research import rf_diagnostics,finite_rf_profile
from figure_export import render_figure


def p(blocks,**kw):
    recipe={'blocks':blocks,'shots':0,'phase_step_deg':0};recipe.update(kw.pop('recipe',{}))
    return Protocol(sequence='custom',backend='native',native_edit=True,resolution=64,custom_recipe=json.dumps(recipe),**kw)


def check(p):
    seq,d=build_native_sequence(p,native=(torch,SequenceDouble,Usage),return_design=True);k=np.zeros(3);read=[];times=[]
    for i,r in enumerate(seq):
        k=np.zeros(3) if r.pulse.usage==Usage.EXCIT else -k
        kt=k+np.cumsum(r.gradm.numpy().astype(float),axis=0);a=r.adc_usage.numpy()>0
        read.extend(kt[a]);times.extend(seq.mrtini_rep_bounds[i][0]+np.cumsum(r.event_time.numpy().astype(float))[a]);k=kt[-1]
    np.testing.assert_allclose(np.asarray(read)[:,:2],d.expected_k,atol=.002,rtol=0);np.testing.assert_allclose(np.asarray(read)[:,2],0,atol=.002,rtol=0);np.testing.assert_allclose(times,d.adc_times,atol=2e-6,rtol=0)
    return seq,d

@pytest.mark.parametrize('style',['cartesian','radial','spiral'])
def test_readouts(style):
    seq,d=check(p([{'type':'excitation','flip_deg':20},{'type':'delay','duration_ms':3},{'type':'readout','style':style}]))
    assert len(d.adc_times)==4096


def test_refocusing_train_and_interleaving():
    seq,d=check(p([{'type':'excitation'},{'type':'delay','duration_ms':3},{'type':'refocusing'},{'type':'delay','duration_ms':2},{'type':'readout','lines':8,'alternating':True,'rewind':False}],tr_ms=500,recipe={'interleaved':True}))
    assert len(np.unique(np.round(d.expected_k,5),axis=0))==4096


def test_balanced_centres():
    seq,d=check(p([{'type':'excitation','balanced':True,'flip_deg':35},{'type':'readout','rewind':True}],tr_ms=12,recipe={'phase_step_deg':180}))
    for r in seq[:-1]:np.testing.assert_allclose(r.gradm.sum(0),0,atol=.002)


def test_reject_timing_and_no_rf():
    with pytest.raises(ValueError):design(p([{'type':'excitation'},{'type':'delay','duration_ms':100},{'type':'readout'}]))
    with pytest.raises(ValueError):design(p([{'type':'readout'}]))


def test_inversion():
    seq,d=check(p([{'type':'inversion','flip_deg':180},{'type':'delay','duration_ms':300},{'type':'excitation'},{'type':'readout'}],tr_ms=500));assert len(seq)==128


def test_rf_equilibrium_and_finite_profile():
    angle=np.pi/3; coef=torch.tensor([np.cos(angle/2)**2,0,np.sin(angle)/np.sqrt(2),np.cos(angle),np.sin(angle/2)**2,0],dtype=torch.complex64).reshape(6,1)
    eng=NS(steps=[NS(bank=NS(tap_map={0:0},table=coef,K=1),phase=torch.zeros(1))],rf=NS(physical=torch.zeros(1)),pos=torch.zeros(1,3),workbench_rep_keys=['p'],art=NS(pulses={'p':NS(water=NS(nominal_taps=NS(deltas=torch.tensor([0]),dk=100.))) }),data=NS(B0=torch.zeros(1),B1=torch.ones(1,1,dtype=torch.complex64)),B0=torch.zeros(1),seq=[NS(pulse=NS(shim_array=torch.tensor([[1.,0.]])))])
    d=rf_diagnostics(eng,0);assert d['period_mm']==10;np.testing.assert_allclose(d['equilibrium_mxy'],np.sin(angle),atol=1e-6);np.testing.assert_allclose(d['equilibrium_mz'],np.cos(angle),atol=1e-6)
    # 90 degree, 1 ms, on-resonance, no gradient: exact rotation.
    pulse=NS(pulses=[{'span':dict(t=[0,.001],rf_amp=[250,0],rf_phase=[0,0],rf_freq=[0,0],gx=[0,0],gy=[0,0],gz=[0,0])}]);f=finite_rf_profile(eng,pulse,0,0,.015);np.testing.assert_allclose(f['mxy'],1,atol=1e-6);np.testing.assert_allclose(f['mz'],0,atol=1e-6)
    assert render_figure('rf',d,'svg').lstrip().startswith(b'<?xml');assert render_figure('rf',d,'pdf').startswith(b'%PDF')


def test_design_endpoint():
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as c:
        r=c.post('/api/design',json={'protocol':vars(p([{'type':'excitation'},{'type':'readout'}]))});assert r.status_code==200,r.text;assert r.json()['samples']==4096
