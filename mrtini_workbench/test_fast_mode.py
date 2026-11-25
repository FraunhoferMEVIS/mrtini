from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from physics import Protocol
from native import NativeRunner


def test_mode_does_not_invalidate_physics(tmp_path):
    f=tmp_path/'phantom.npz';f.write_bytes(b'phantom');r=NativeRunner(tmp_path)
    p=Protocol(backend='native',native_edit=True)
    assert r.prepared_key(p,None,f)==r.prepared_key(replace(p,diagnostics=False),None,f)
    with pytest.raises(ValueError,match='diagnostics'):replace(p,diagnostics='false').check()


def test_fast_native_skips_diagnostics_and_preserves_adc(tmp_path,monkeypatch):
    import server
    from types import SimpleNamespace as NS
    sig=np.arange(4,dtype=np.complex64)[:,None];k=np.zeros((4,2));img=np.ones((2,2),np.complex64)
    def simulate(*args):return sig,k,{},dict(magnitude=abs(img),phase=np.angle(img),complex_image=img,kspace=abs(img),method='fixture')
    monkeypatch.setattr(server,'runner',NS(run=simulate,engine=NS(data=None,rf=NS(physical=None),graph=None)))
    monkeypatch.setattr(server,'ROOT',tmp_path);monkeypatch.setattr(server,'active',None);monkeypatch.setattr(server,'jobs',{})
    calls=[]
    def maps(*args):calls.append('maps');return dict(PD=np.ones((2,2)))
    def overview(*args):calls.append('overview');return []
    def snapshot(g,path):calls.append('snapshot');np.savez(path,kind=[0])
    monkeypatch.setattr(server,'native_maps',maps);monkeypatch.setattr(server,'overview',overview);monkeypatch.setattr(server,'snapshot_graph',snapshot)
    for mode,i in [(False,'a'*32),(True,'b'*32)]:
        d=tmp_path/'runs'/i;d.mkdir(parents=True);j=dict(id=i,directory=str(d));server.jobs[i]=j
        server.run(j,Protocol(backend='native',native_edit=True,diagnostics=mode))
        assert j['status']=='done',j
        assert (d/'maps.npz').exists()==mode and (d/'graph_nodes.npz').exists()==mode
        assert ('diagnostics' in j['metrics'])==mode
        assert j['metrics']['diagnostics_enabled']==mode
        assert 'export' in j['metrics'] and 'display' in j['metrics']
        assert ('phantom' in j['result']['images'])==mode
        with np.load(d/'result.npz') as a:np.testing.assert_array_equal(a['signal'],sig);np.testing.assert_array_equal(a['complex_image'],img)
        if not mode:
            assert calls==[]
            from fastapi import HTTPException
            with pytest.raises(HTTPException,match='Fast acquisition'):server.analysis_ready(i)
    assert calls==['maps','overview','snapshot']
