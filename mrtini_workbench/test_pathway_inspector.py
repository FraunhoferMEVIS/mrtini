"""Scientific regression checks for contextual RF and state ancestry inspection."""
import numpy as np
import pytest
from physics import Protocol
from native_sequences import design
from analysis_views import rf_context


def test_flash_context_has_real_rephaser():
    d=design(Protocol(sequence='flash',resolution=64))
    c=rf_context(d,0);a=c['local'];assert c['role']=='excitation/preparation'
    assert a['area_rf']>0 and a['area_after']<0
    np.testing.assert_allclose(a['area_after'],-a['area_rf']/2,rtol=1e-8)
    assert any(g[4]<0 for g in a['gradients'])
    assert a['lo']<c['t0'] and a['hi']>c['t1']
    assert all(g[0]>=a['lo'] and g[1]<=a['hi'] for g in a['gradients'])


def test_refocusing_context_uses_actual_signed_moments():
    d=design(Protocol(sequence='rare',resolution=64,tr_ms=1000,te_ms=12,esp_ms=12))
    c=rf_context(d,1);assert c['role']=='refocusing'
    a=c['local'];assert a['lo']<c['t0'] and a['hi']>c['t1']
    # Match actual compiled gradient areas (not an assumed negative rephaser).
    for name,lo,hi in [('before',a['lo'],c['t0']),('rf',c['t0'],c['t1']),('after',c['t1'],a['hi'])]:
        np.testing.assert_allclose(a['area_'+name],d.area(lo,hi)[2],atol=1e-6)


def test_ancestry_coordinates_and_class_contributions():
    # Optional recovered prepared-engine fixture; not a CUDA test.
    fixture=pytest.importorskip('test_scaling').fixture
    prepare=pytest.importorskip('optimized').prepare_gpu_execution
    from research import voxel_replay
    eng=prepare(*fixture(13,'generic',True,n_states=11),kernel_backend='torch')
    d=voxel_replay(eng,2,1,slab_thickness_m=.005,adc_index=3)['contributions']
    plain=voxel_replay(eng,2,1,slab_thickness_m=.005,adc_index=3,trace_paths=False)['contributions']
    assert not plain['ancestry_options'];assert d['ancestry_options']
    np.testing.assert_allclose([d['sum_real'],d['sum_imag']],[plain['sum_real'],plain['sum_imag']],atol=1e-7)
    assert d['history_sum_error']<1e-6
    for option in d['ancestry_options']:
        trace=option['trace'];steps=trace['steps'];assert steps
        assert steps[-1]['rep']==1 and steps[-1]['kind']=='F+'
        s=eng.steps[1];row=steps[-1]['local'];k=int((s.rows==row).nonzero()[0]);expected=(s.k0[k]+s.adc[3]).numpy()
        np.testing.assert_allclose(trace['adc_kt'],expected,rtol=1e-6)
        np.testing.assert_allclose(trace['z_weight'],np.sinc(expected[2]*.005),rtol=1e-6)
        np.testing.assert_allclose(steps[-1]['curve'][-1][1],expected[2],rtol=1e-6)
        assert all(a['time_ms']<=b['time_ms'] for a,b in zip(steps,steps[1:]))


def stimulated_fixture():
    """Small, deliberately hand-constructed graph with physically consistent
    F -> conjugated Z -> F storage/recall; not a Bloch or Rust-prepass test.
    Reuse only the prepared executor's data/bank fixture, replace topology and
    gradient events. The recalled state reaches kz=0 and tau=0 at the ADC.
    """
    from types import SimpleNamespace as NS
    import torch
    from test_scaling import fixture
    g,seq,data,_,art=fixture(13,'uniform',False,n_states=8)
    for name in ('PD','T1','T2','T2dash'):
        getattr(data,name).fill_(1.)
    data.B0.zero_();data.B1.fill_(1);data.voxel_pos.zero_();data.coil_sens.fill_(1);data.D.zero_()
    for name in ('pp','pz','zp','zz','mp','mz'):
        t=getattr(art.pulses['rf'].water.vox6,name);t.zero_();t[:,1]=.6 if name in ('zp','mz','mp') else .2
    def state(kind,anc=()):return NS(dist_type=kind,mag=None,kt_vec=None,emitted_signal=.5 if kind=='+' else 0.,latent_signal=.5,ancestors=list(anc))
    eq=state('z0');f=state('+',[('z+',eq,0,1)]);z=state('z',[('-z',f,0,1)]);ref=state('+',[('-+',f,0,1)]);rec=state('+',[('z+',z,0,1)]);f2=state('+',[('++',ref,0,1)])
    graph=[[eq],[f],[z,ref],[rec,f2]]
    for j,r in enumerate(seq):
        r.pulse.phase=0.;r.gradm=torch.tensor([[0.,0.,500.]],dtype=torch.float32);r.event_time=torch.tensor([.006]);r.adc_usage=torch.tensor([1 if j==2 else 0]);r.adc_phase=torch.zeros(1)
    return graph,seq,data,['rf']*3,art


def test_recalled_state_approaches_signal_band():
    pytest.importorskip('test_scaling');prepare=pytest.importorskip('optimized').prepare_gpu_execution
    from research import voxel_replay
    eng=prepare(*stimulated_fixture(),kernel_backend='torch');d=voxel_replay(eng,2,2,slab_thickness_m=.005,adc_index=0)['contributions'];item=next(x for x in d['ancestry_options'] if x['category']==5);t=item['trace'];steps=t['steps']
    assert [x['kind'] for x in steps]==['F+','Z','F+'];assert [x['relation'] for x in steps]==['z+','-z','z+']
    np.testing.assert_allclose([s['start_kt'][2] for s in steps],[0,-500,-500],atol=1e-5)
    np.testing.assert_allclose([s['end_kt'][2] for s in steps],[500,-500,0],atol=1e-5)
    assert abs(t['adc_kt'][3])<1e-8 and abs(t['z_weight']-1)<1e-6
    assert d['history_sum_error']<1e-6
