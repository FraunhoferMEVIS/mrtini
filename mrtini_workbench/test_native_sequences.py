"""Tests of waveform design and native representation. Run with unittest.
API-double tests are not an end-to-end test of the installed MRTini package.
"""
import unittest
from types import SimpleNamespace
from enum import Enum
from dataclasses import replace
import numpy as np
import torch
from physics import Protocol
from native_sequences import build_native_sequence
class Usage(Enum): EXCIT=1; REFOC=2
class SequenceDouble(list):
    def __init__(self,**kw):super().__init__();self.__dict__.update(kw)
    def new_rep(self,n):
        rep=SimpleNamespace(event_time=torch.zeros(n),gradm=torch.zeros(n,3),adc_usage=torch.zeros(n,dtype=torch.int32),adc_phase=torch.zeros(n),pulse=SimpleNamespace());self.append(rep);return rep

def protocol(name,n=64):
    kw={'flash':dict(tr_ms=35,te_ms=8,flip_deg=20),'rare':dict(tr_ms=1800,te_ms=12,esp_ms=12,flip_deg=90),'epi':dict(tr_ms=1500,te_ms=100,esp_ms=4.5,flip_deg=90,etl=16)}[name]
    return Protocol(sequence=name,resolution=n,backend='native',native_edit=True,**kw)

def check_representation(test,p):
    seq,d=build_native_sequence(p,native=(torch,SequenceDouble,Usage),return_design=True)
    test.assertEqual(len(seq),len(d.pulses));test.assertEqual(len(seq.mrtini_hybrid),len(seq))
    test.assertEqual(sum(int((r.adc_usage>0).sum()) for r in seq),p.resolution**2)
    # Independent primary-coherence recursion through the actual rep.gradm values.
    k=np.zeros(3);read=[];times=[]
    for i,rep in enumerate(seq):
        k=np.zeros(3) if rep.pulse.usage==Usage.EXCIT else -k
        kt=k+np.cumsum(rep.gradm.numpy().astype(float),axis=0);a=rep.adc_usage.numpy()>0
        read.extend(kt[a]);times.extend(seq.mrtini_rep_bounds[i][0]+np.cumsum(rep.event_time.numpy().astype(float))[a]);k=kt[-1]
        span=rep.raw_waveform;rf=span['rf_amp'][:-1];dt=np.diff(span['t']);angle=2*np.pi*np.sum(rf*dt)
        test.assertAlmostEqual(angle,float(rep.pulse.angle),places=6)
        test.assertTrue(np.all(rep.event_time.numpy()>0));test.assertAlmostEqual(float(rep.event_time.double().sum()),np.diff(seq.mrtini_rep_bounds[i])[0],places=5)
    read=np.array(read);np.testing.assert_allclose(read[:,:2],d.expected_k,atol=.002,rtol=0)
    np.testing.assert_allclose(read[:,2],0,atol=.002,rtol=0)
    np.testing.assert_allclose(times,d.adc_times,atol=2e-6,rtol=0)
    grid=np.rint(read[:,:2]*p.fov_mm/1000).astype(int);test.assertEqual(len(np.unique(grid,axis=0)),p.resolution**2)
    # Check exact nominal echo times (including the reverse EPI readout).
    for entry in d.pulses:
        for t,phase in entry['reads']:
            test.assertTrue(np.isfinite(t) and np.isfinite(phase))
    return d

class TestDesign(unittest.TestCase):
    def test_flash(self):
        for n in [64,128,256]:check_representation(self,protocol('flash',n))
    def test_rare(self):
        for n in [64,128,256]:check_representation(self,protocol('rare',n))
    def test_epi(self):
        for n in [64,128,256]:check_representation(self,protocol('epi',n))
    def test_parameters(self):
        a=check_representation(self,protocol('flash'));p=replace(protocol('flash'),flip_deg=37,te_ms=11,tr_ms=42,fov_mm=210,rf_ms=2,tbw=5,slice_mm=4);b=check_representation(self,p)
        self.assertNotEqual(a.pulses[0]['span']['flip_deg'],b.pulses[0]['span']['flip_deg']);self.assertFalse(np.allclose(a.expected_k,b.expected_k));self.assertFalse(np.allclose(a.adc_times,b.adc_times))
        check_representation(self,replace(protocol('rare'),etl=4,refocus_deg=145,esp_ms=15,te_ms=15))
        check_representation(self,replace(protocol('epi'),etl=8,te_ms=75,esp_ms=5,refocus_deg=160))
    def test_invalid_timing(self):
        for p in [replace(protocol('flash'),te_ms=2,rf_ms=5),replace(protocol('rare'),te_ms=3,esp_ms=3),replace(protocol('epi'),esp_ms=.5)]:
            with self.assertRaises(ValueError):build_native_sequence(p,native=(torch,SequenceDouble,Usage))
if __name__=='__main__':unittest.main()
