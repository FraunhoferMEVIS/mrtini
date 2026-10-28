import logging
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from reference_math import GAMMA_HZ, channel_six, dft_matrix
from scipy.spatial.transform import Rotation


def make_span(**kw):
    s=dict(kind='bloch',t=np.linspace(0,.001,5),rf_amp=np.full(5,100.),rf_phase=np.zeros(5),gz=np.full(5,10000.),rf_freq=0.)
    s.update(kw);return s

@pytest.mark.parametrize('phase,amp,constant', [([0.,2*math.pi],[1.,1.],True),([0.,1.],[1.,1.],False),([0.,1.],[1.,0.],True),([1.,2.],[0.,0.],True),([.7],[1.],True)])
def test_constant_phase_wraps_and_ignores_zero_amplitude(main,phase,amp,constant):
    ok,phi=main.is_constant_phase(np.array(phase),np.array(amp));assert bool(ok)==constant
    assert np.isfinite(phi)


def test_hash_frequency_tag_and_phase_factoring(main):
    assert main._freq_tag(3.)==main._freq_tag(np.full(5,3.))
    assert main._freq_tag([0.,1.]).startswith('sha1:')
    assert main._hash_array([1.,2.])==main._hash_array(np.array([1.,2.]))
    a=main._bloch_kernel_signature(make_span())
    b=main._bloch_kernel_signature(make_span(rf_phase=np.full(5,.7)))
    assert a[0]==b[0] and b[-1]['phi0']==pytest.approx(.7)
    c=main._bloch_kernel_signature(make_span(rf_phase=np.linspace(0,1,5)))
    assert c[0]!=a[0] and not c[-1]['const_phase']
    for change in [dict(rf_amp=np.full(5,200.)),dict(gz=np.full(5,20000.)),dict(rf_freq=10.),dict(t=np.linspace(0,.002,5))]:
        assert main._bloch_kernel_signature(make_span(**change))[0]!=a[0]

@pytest.mark.parametrize('fm',[None,[],20.])
def test_complex_rf_frequency_phase_and_units(main,fm):
    s=make_span(rf_freq=100.,rf_fm=fm,rf_phase=.3)
    rf=main.build_complex_rf(s)
    f=100.+(20. if fm==20. else 0.)
    expected=(100./GAMMA_HZ)*np.exp(1j*(.3+2*math.pi*f*s['t']))
    np.testing.assert_allclose(rf,expected,atol=1e-12,rtol=1e-6)


def test_signature_modulation_and_single_sample(main):
    a=main._bloch_kernel_signature(make_span(rf_fm=10.))
    expected=100/GAMMA_HZ*np.exp(2j*np.pi*10*np.linspace(0,.001,5))
    np.testing.assert_allclose(a[1],expected,rtol=1e-6)
    a=main._bloch_kernel_signature(make_span(t=np.array([0.]),rf_amp=[1.],rf_phase=[0.],gz=[0.]))
    assert a[1].shape==(1,)
    assert main.build_complex_rf(make_span(t=[0.],rf_amp=[1.],rf_phase=0.)).shape==(1,)


def test_zgrid_is_odd_uniform_centered_and_periodic(main):
    z=main.make_z_grid_slice_select(.005,2,4,.002,2e5,device='cpu')
    assert len(z)%2==1 and z[len(z)//2]==0
    if len(z)>1:
        dz=z[1]-z[0]
        torch.testing.assert_close(torch.diff(z),torch.full_like(z[1:],dz),atol=1e-8,rtol=1e-5)
        assert (len(z)*dz).item()==pytest.approx(.01,rel=1e-5)


def test_cartesian_pair_order_and_kgrid(main,repetition):
    b0,b1,n0,n1=main.make_pairs([1.,2.],[.8,1.,1.2])
    assert (n0,n1)==(2,3)
    torch.testing.assert_close(b0,torch.tensor([1.,1.,1.,2.,2.,2.]))
    torch.testing.assert_close(b1,torch.tensor([.8,1.,1.2,.8,1.,1.2]))
    k=main.k_grid_from_seq([repetition]);assert k.shape==(4,) and torch.isfinite(k).all()
    repetition.gradm.zero_();repetition.event_time.zero_()
    torch.testing.assert_close(main.k_grid_from_seq([repetition]),torch.full((4,),10.))


def test_logger_idempotence_and_stage_exception(main,tmp_path):
    name='mrtini.unit.'+tmp_path.name
    log=main.setup_logger(name,log_dir=tmp_path)
    try:
        assert main.setup_logger(name,log_dir=tmp_path) is log
        assert len(log.handlers)==2
        with main.log_stage(log,'success'): pass
        with pytest.raises(RuntimeError,match='sentinel'):
            with main.log_stage(log,'failure'): raise RuntimeError('sentinel')
        text=(tmp_path/'run.log').read_text()
        assert 'success done' in text and 'failure FAILED' in text
    finally:
        for h in log.handlers[:]: h.close();log.removeHandler(h)


def test_simulator_requires_sequence(main,monkeypatch,tmp_path):
    log=logging.getLogger('unit-simulator')
    monkeypatch.setattr(main,'setup_logger',lambda **kw:log)
    sim=main.MRtiniSim(device='cpu')
    with pytest.raises(ValueError,match='sequence is None'):
        sim.bake_bloch_kernels(phantom_data=None,device='cpu')


def test_build_artifacts_real_bank_pipeline(main,bank,monkeypatch,sm):
    monkeypatch.setattr(main,'setup_logger',lambda **kw:logging.getLogger('unit-artifacts'))
    sim=main.MRtiniSim(device='cpu')
    jb=SimpleNamespace(b0=SimpleNamespace(centers=np.array([-10.,10.],dtype=np.float32)),b1=SimpleNamespace(centers=np.array([.8,1.2],dtype=np.float32)))
    data=SimpleNamespace(B0=torch.tensor([-10.,10.]),B1=torch.tensor([.8,1.2]),PD=torch.tensor([1.,2.]))
    R=bank.Rxyz_bank.reshape(4,len(bank.z_m),3,3).permute(0,2,3,1)
    out=sim._build_pulse_artifacts(key='unit',Rz=R,z_m=bank.z_m,t=np.array([0.,.001]),meta={'phi0':.4},jb=jb,phantom_data=data,device='cpu',tap_threshold=1e-5)
    assert out.key=='unit' and out.t_rf==pytest.approx(.001)
    assert set(out.support)=={'++','+z','-+','-z','z+','zz'}
    assert out.vox6.pp.shape[0]==2
    # Test the baked VALUES against Cartesian rotations + an independent DFT.
    z=out.z_m.detach().cpu().numpy();k=out.bank.kgrid.kz.detach().cpu().numpy()
    reference=channel_six(Rotation.from_rotvec([[0.,0.,0.],[.9,0.,0.]]).as_matrix())
    for name in ['pp','pz','mp','mz','zp','zz']:
        full=np.broadcast_to(reference[name][:,None],(2,len(z)))@dft_matrix(z,k)
        expected_nominal=(full[0]+2*full[1])/3
        np.testing.assert_allclose(getattr(out.nominal_taps,name).detach().cpu(),expected_nominal,atol=4e-6,rtol=4e-6)
        selected=np.empty((2,len(out.supp_map)),dtype=complex)
        for full_index,compact_index in out.supp_map.items(): selected[:,compact_index]=full[:,full_index]
        np.testing.assert_allclose(getattr(out.vox6,name).detach().cpu(),selected,atol=4e-6,rtol=4e-6)

@pytest.mark.parametrize('global_rf',[False,True])
def test_bake_real_bloch_to_bank_and_cache_reuse(main,seqmod,monkeypatch,global_rf):
    monkeypatch.setattr(main,'setup_logger',lambda **kw:logging.getLogger('unit-bake'))
    seq=seqmod.Sequence(normalized_grads=False)
    rep=seq.new_rep(1);rep.event_time[:]=.002;rep.gradm[0,2]=1.
    a=make_span();b=make_span(rf_phase=np.full(5,.4))
    seq.mrtini_hybrid=[{'kind':'epg'},a,b]
    data=SimpleNamespace(B0=torch.tensor([0.,10.]),B1=torch.tensor([.8,1.2]),PD=torch.ones(2),chemical_shift=torch.tensor([0.,-412.]))
    sim=main.MRtiniSim(device='cpu',seq=seq)
    art=sim.bake_bloch_kernels(phantom_data=data,device='cpu',global_RF=global_rf,tbw=4.,z_span_m=.005)
    assert art.span_key[1]==art.span_key[2] and len(art.pulses)==1
    assert art.phi0_by_span[2]==pytest.approx(.4)
    pulse=art.pulses[art.span_key[1]]
    assert pulse.water is not None and pulse.fat is not None
    assert pulse.combined_taps.pp.shape==pulse.water.nominal_taps.pp.shape
    assert torch.isfinite(pulse.water.vox6.pp).all()
    seq.mrtini_hybrid=[]
    with pytest.raises(ValueError,match="No 'bloch' span"):
        sim.bake_bloch_kernels(phantom_data=data,device='cpu')
