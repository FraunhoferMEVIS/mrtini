"""Python importer tested with an explicit native-parser contract double.
This verifies event conversion, not the native parser or real gammaSTAR files.
"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

class ParserContract:
    def __init__(self,adc): self.adc=adc
    def duration(self): return .010
    def encounter(self,name,t):
        intervals=[(.001,.002),(.005,.006)] if name=='rf' else []
        return next(((a,b) for a,b in intervals if a>=t),None)
    def events(self,name,start,end):
        if name=='adc' and self.adc: return [v for v in [.003,.004,.007,.008] if start<v<end]
        if name.startswith('grad'): return [v for v in [.0025,.0045,.0065,.009] if start<v<end]
        return []
    def integrate(self,times):
        dt=np.diff(times)
        return NS(gradient=NS(x=2*dt,y=3*dt,z=4*dt))
    def integrate_one(self,a,b):
        return NS(pulse=NS(angle=torch.tensor([.5 if a<.004 else 3.]),phase=torch.tensor([.2])),
                  gradient=NS(x=2*(b-a),y=3*(b-a),z=4*(b-a)))
    def raw_waveforms(self,a,b):
        t=np.linspace(a,b,5)
        return dict(t=t,rf_amp=np.ones(5),rf_phase=np.zeros(5),rf_freq=0.,gx=np.zeros(5),gy=np.zeros(5),gz=np.ones(5),adc_active=np.zeros(5,dtype=bool),adc_phase=np.zeros(5),adc_freq=np.zeros(5))
    def sample(self,t):
        active=((t>.0025)&(t<.0041))|((t>.0065)&(t<.0081)) if self.adc else np.zeros(len(t),dtype=bool)
        return NS(adc=NS(active=active,phase=np.zeros(len(t))))
    def hybrid(self,pad,flag):
        return [dict(kind='epg',t0=0.,t1=.001),dict(kind='bloch',t0=.001,t1=.002),
                dict(kind='epg',t0=.002,t1=.005),dict(kind='bloch',t0=.005,t1=.006),dict(kind='epg',t0=.006,t1=.010)]

@pytest.mark.parametrize('adc',[False,True])
@pytest.mark.parametrize('exact',[False,True])
def test_importer_converts_native_parser_contract(seqmod,monkeypatch,adc,exact):
    monkeypatch.setattr(seqmod.mrstar,'load_gstar',lambda name:ParserContract(adc),raising=False)
    s=seqmod.Sequence.import_file_for_mrtini_sim('contract-only.json',exact_trajectories=exact,print_stats=True)
    assert len(s)==2 and s.normalized_grads is False
    assert s[0].pulse.usage==seqmod.PulseUsage.EXCIT and s[1].pulse.usage==seqmod.PulseUsage.REFOC
    assert s.mrtini_bloch_indices==[1,3] and s.mrtini_epg_indices==[0,2,4]
    assert s.mrtini_rep_bounds==[(.0015,.0055),(.0055,.010)]
    for rep in s:
        torch.testing.assert_close(rep.gradm[:,0],2*rep.event_time)
        torch.testing.assert_close(rep.gradm[:,1],3*rep.event_time)
        torch.testing.assert_close(rep.gradm[:,2],4*rep.event_time)
        if not adc: assert not rep.adc_usage.any()
    assert s.mrtini_hybrid[1]['rf_center']==pytest.approx(.0015)
    if adc: assert sum(int(r.adc_usage.sum()) for r in s)>0
