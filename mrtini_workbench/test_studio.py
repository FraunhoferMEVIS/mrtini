import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from physics import Protocol
from native_sequences import build_native_sequence,design
from timed_design import scanner_report,expand
from test_native_sequences import SequenceDouble,Usage
from postprocess import process

def config(events,shots=2):
    return Protocol(sequence='custom',backend='native',native_edit=True,resolution=64,custom_recipe=json.dumps(dict(mode='timed',shots=shots,events=events)))

def events():
    return [dict(channel='rf',start_ms=0,duration_ms=1.5,flip_deg=20),dict(channel='gz',start_ms=0,duration_ms=1.5,moment=800),dict(channel='gz',start_ms=1.5,duration_ms=1,moment=-400),dict(channel='gx',start_ms=3,duration_ms=1,moment=-10),dict(channel='gy',start_ms=3,duration_ms=1,moment=10,shot_step=5),dict(channel='gx',start_ms=4,duration_ms=2,moment=100),dict(channel='adc',start_ms=4,samples=64,dwell_us=12)]

def test_timed_native_moments():
    p=config(events());seq,d=build_native_sequence(p,native=(torch,SequenceDouble,Usage),return_design=True);out=[]
    for r in seq:
        k=np.cumsum(r.gradm.numpy().astype(float),axis=0);out.extend(k[r.adc_usage.numpy()>0]);assert len(r.raw_waveform['gz'])==151
    np.testing.assert_allclose(np.array(out)[:,:2],d.expected_k,atol=.002,rtol=0)
    np.testing.assert_allclose(np.array(out)[:,2],0,atol=.002)
    assert len(d.adc_times)==128
    assert scanner_report(d,{'gradient_mtm':1})['gradient_exceeded']

def test_overlap_rejected():
    ev=events();ev[-1]['start_ms']=.2
    with pytest.raises(ValueError,match='ADC during RF'):design(config(ev))

def test_repeat():
    r=expand([dict(channel='repeat',start_ms=1,period_ms=2,count=3,events=[dict(channel='gx',start_ms=.5,duration_ms=1)])]);assert [e['start_ms'] for e in r]==[1.5,3.5,5.5]

def test_independent_reconstruction():
    p=Protocol(resolution=64);k=np.array([[0,0],[1,0]])/(p.fov_mm/1000);s=np.array([[1+0j],[2+1j]]);before=s.copy();r=process(s,k,p.__dict__,{},dict(matrix=128,method='auto',noise_sigma=.1,seed=2));assert r['magnitude'].shape==(128,128);np.testing.assert_array_equal(s,before)
    a=process(s,k,p.__dict__,{},dict(matrix=64,method='auto',psf=True));assert a['magnitude'].shape==(64,64);assert np.isfinite(a['magnitude']).all()

def test_imported_units_guard():
    p=Protocol(backend='native',native_edit=False);s=np.ones((2,1),complex);k=np.zeros((2,2));
    with pytest.raises(ValueError,match='explicitly verify'):process(s,k,p.__dict__,{},dict(method='auto'))

def test_direct_inverse_recovers_point():
    from inverse import direct_cg
    n=4;fov=.2;kx,ky=np.meshgrid((np.arange(n)-n//2)/fov,(np.arange(n)-n//2)/fov);k=np.c_[kx.ravel(),ky.ravel()];a=np.zeros((n,n));a[1,2]=1
    s=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a))).ravel()*n*n;r=direct_cg(s,k,n,fov,regularization=1e-7)
    np.testing.assert_allclose(r['magnitude'],a,atol=1e-6)
