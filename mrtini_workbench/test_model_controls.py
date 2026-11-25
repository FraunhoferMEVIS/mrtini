from dataclasses import replace
import pytest
from physics import Protocol
from native import NativeRunner, PhantomCache

@pytest.mark.parametrize('edit',[True,False])
def test_model_and_budget_invalidate_prepared_not_phantom(tmp_path,edit):
    seq=tmp_path/'seq.json';seq.write_text('{}');ph=tmp_path/'phantom.npz';ph.write_bytes(b'fixture')
    p=Protocol(backend='native',native_edit=edit)
    r=NativeRunner(tmp_path)
    key=lambda q:r.prepared_key(q,None if edit else seq,ph)
    assert key(p)!=key(replace(p,global_rf=True))
    assert key(p)!=key(replace(p,max_states=300))
    assert key(p)==key(replace(p,cuda_graph=True))
    c=PhantomCache();calls=[]
    def build():calls.append(1);return {'data':42}
    for q in (p,replace(p,global_rf=True),replace(p,max_states=300)):
        c.get(c.identity(ph,q.resolution,q.slice_index,q.slice_mm*1e-3),build)
    assert len(calls)==1

@pytest.mark.parametrize('n',[1,100,200,300,400,600,800,1375,20000])
def test_state_budget_valid(n):
    assert Protocol(max_states=n,global_rf=True).check().max_states==n

@pytest.mark.parametrize('opts',[{'max_states':0},{'max_states':20001},{'max_states':2.5},{'max_states':True},{'max_states':'100'},{'global_rf':'false'}])
def test_model_controls_reject_invalid(opts):
    with pytest.raises(ValueError):Protocol(**opts).check()
