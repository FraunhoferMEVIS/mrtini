"""Acquisition-wide RF history uses native ADC event order, never a sequence label."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from research import adc_windows, contribution_overview, voxel_replay


def test_window_discovery():
    for a, expected in [([],[]),([0,0],[]),([0,1,1,0,1,0,1,1,1],[(0,2),(2,3),(3,6)]),([1,1,1],[(0,3)])]:
        rep=NS(adc_usage=torch.tensor(a));step=NS(adc=[0]*sum(a));assert adc_windows(rep,step)==expected
    with pytest.raises(ValueError,match='ADC event count'):
        adc_windows(NS(adc_usage=torch.tensor([1,0,1])),NS(adc=[0]))


@pytest.mark.parametrize('packed',[False,True])
def test_overview_matches_single_adc_and_production(packed):
    fixture=pytest.importorskip('test_scaling').fixture
    prepare=pytest.importorskip('optimized').prepare_gpu_execution
    eng=prepare(*fixture(13,'generic',packed,n_states=11),kernel_backend='torch')
    baseline=eng.run().clone();v=2
    result=contribution_overview(eng,v)
    assert result['windows'];assert result['max_absolute_residual']<1e-5
    for r in result['windows']:
        expected=voxel_replay(eng,v,r['rep'],adc_index=r['adc_in_rep'],trace_paths=False)['contributions']
        np.testing.assert_allclose(r['real'],[c['real'] for c in expected['history_classes']],rtol=1e-5,atol=2e-7)
        np.testing.assert_allclose(r['imag'],[c['imag'] for c in expected['history_classes']],rtol=1e-5,atol=2e-7)
        np.testing.assert_allclose(r['near_real'],[c['near_zero_time_real'] for c in expected['history_classes']],atol=2e-7)
        np.testing.assert_allclose(r['near_imag'],[c['near_zero_time_imag'] for c in expected['history_classes']],atol=2e-7)
        assert r['adc_index']==expected['adc_index']
        assert r['adc_start']<=r['adc_index']<r['adc_stop']
    torch.testing.assert_close(eng.run(),baseline,rtol=0,atol=0)


def test_storage_history_survives_unsampled_layers():
    pytest.importorskip('test_scaling');prepare=pytest.importorskip('optimized').prepare_gpu_execution
    from test_pathway_inspector import stimulated_fixture
    eng=prepare(*stimulated_fixture(),kernel_backend='torch')
    r=contribution_overview(eng,2);assert len(r['windows'])==1
    w=r['windows'][0];assert w['rep']==2;assert abs(complex(w['real'][5],w['imag'][5]))>0
    assert w['residual']<1e-6


def test_two_windows_within_one_rf_interval():
    fixture=pytest.importorskip('test_scaling').fixture
    prepare=pytest.importorskip('optimized').prepare_gpu_execution
    g,seq,d,keys,art=fixture(13,'generic',False,n_states=11)
    # Use the actual event count of this sequence, and create a gap in its ADC mask.
    rep=seq[1];a=rep.adc_usage.clone();assert a.numel()>=4
    a.zero_();a[0]=1;a[2:4]=1;rep.adc_usage=a
    eng=prepare(g,seq,d,keys,art,kernel_backend='torch');rows=[r for r in contribution_overview(eng,2)['windows'] if r['rep']==1]
    assert [(r['samples'],r['adc_in_rep']) for r in rows]==[(1,0),(2,2)]
    for row in rows:
        q=voxel_replay(eng,2,1,adc_index=row['adc_in_rep'],trace_paths=False)['contributions']
        np.testing.assert_allclose([row['sum_real'],row['sum_imag']],[q['sum_real'],q['sum_imag']],atol=2e-7)
