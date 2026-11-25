"""Offline tests of native graph serialization and timeline representation."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from analysis_views import physical_masks,graph_window,sequence_window
from native_sequences import design
from test_native_sequences import protocol


def graph_fixture(labels=(0,1)):
    # 1 initial equilibrium state, 3 states in each of two layers.
    masks=np.array([3,1,2,3,1,2,3],dtype=np.uint8)
    if len(labels)==1:masks=np.ones_like(masks)
    return NS(case_species=list(labels),offsets=np.array([0,1,4,7]),node_mask=masks,kind=np.array([2,0,1,0,0,1,0]),kt_end=np.arange(28).reshape(7,4).astype(float),kt_start=np.zeros((7,4)),emitted=np.ones((7,len(labels)),np.float32),latent=np.ones((7,len(labels)),np.float32),rows=np.array([0,0,1,2,3,5,6,7]),src=np.array([0,0,0,0,2,1,2]),rel=np.zeros(7,dtype=int),tap=np.zeros(7,dtype=int),mask=np.array([1,2,3,1,3,2,3],dtype=np.uint8) if len(labels)==2 else np.ones(7,dtype=np.uint8))

def test_union_membership():
    v=graph_window(graph_fixture());assert v['totals']=={'water':2,'fat':2,'shared':3};assert v['total_edges']==7
    assert {e['mask'] for e in v['edges']}=={1,2,3};assert len(v['edges'])==7
    assert next(e for e in v['edges'] if e['target']==4)['source']==1
    for mode,allowed in [('water',{1,3}),('fat',{2,3}),('shared',{3}),('water_only',{1}),('fat_only',{2})]:
        g=graph_window(graph_fixture(),mode=mode);assert all(n['mask'] in allowed for n in g['nodes']);assert all(e['mask'] in allowed for e in g['edges'])
        assert all(e['source'] in {n['id'] for n in g['nodes']} and e['target'] in {n['id'] for n in g['nodes']} for e in g['edges'])

def test_single_fat_case_maps_bit_zero_to_fat():
    assert physical_masks(np.array([1],np.uint8),[1]).tolist()==[2]
    v=graph_window(graph_fixture((1,)));assert v['totals']=={'water':0,'fat':7,'shared':0};assert all(e['mask']==2 for e in v['edges'])
    assert not graph_window(graph_fixture((1,)),mode='water')['nodes']

def test_edge_cap_and_window():
    v=graph_window(graph_fixture(),start=1,window=2,edge_limit=1);assert len(v['edges'])==1;assert v['candidate_edges']==4;assert v['display_limited'];assert v['layers'][1]['edge_counts']=={'water':1,'fat':1,'shared':2}

@pytest.mark.parametrize('name',['flash','rare','epi'])
def test_strip(name):
    d=design(protocol(name));s=sequence_window(design=d,start=0,window=4)
    assert len(s['pulses'])==4;assert s['lo']<s['hi'];assert s['gradient_unit']=='mT/m';assert all(s['lo']<=t<=s['hi'] for t in s['adc'])
    assert sum(len(p['t']) for p in s['pulses'])>4
    for pulse in s['pulses']:
        assert len(pulse['t'])==len(pulse['real'])==len(pulse['imag'])
    np.testing.assert_allclose(s['pulses'][0]['flip'],protocol(name).flip_deg)
    assert s['adc']==[float(t) for p in d.pulses[:4] for t,_ in p['reads']]
