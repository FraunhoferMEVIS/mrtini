"""Behavioral regression contracts, not claims that the current source has defects.
These checks run normally; failures are never automatically marked expected.
Projection and demodulation are covered by the independent numerical tests.
"""
import math
import numpy as np
import pytest
import torch



@pytest.mark.regression
def test_repetition_clone_preserves_effective_z(repetition):
    repetition.gradm_z_eff=torch.tensor([7.,8.,9.])
    torch.testing.assert_close(repetition.clone().gradm_z_eff,repetition.gradm_z_eff)

@pytest.mark.regression
def test_repetition_cpu_preserves_waveform(repetition):
    repetition.raw_waveform={'sentinel':'waveform'}
    assert isinstance(repetition.cpu().raw_waveform,dict)
    assert repetition.cpu().raw_waveform==repetition.raw_waveform

@pytest.mark.regression
def test_sequence_clone_preserves_hybrid_metadata(seqmod,repetition):
    s=seqmod.Sequence([repetition],mrtini_hybrid=[{'kind':'epg'}],mrtini_rep_bounds=[(0.,1.)],mrtini_spans_by_rep=[[0]])
    c=s.clone()
    assert c.mrtini_hybrid==s.mrtini_hybrid
    assert c.mrtini_rep_bounds==s.mrtini_rep_bounds and c.mrtini_spans_by_rep==s.mrtini_spans_by_rep

@pytest.mark.regression
def test_sequence_default_metadata_is_independent(seqmod):
    a,b=seqmod.Sequence(),seqmod.Sequence()
    assert a.mrtini_hybrid is not b.mrtini_hybrid
    assert a.mrtini_rep_bounds is not b.mrtini_rep_bounds
    assert a.mrtini_spans_by_rep is not b.mrtini_spans_by_rep

@pytest.mark.regression
def test_collapse_keeps_schema(seqmod):
    s=dict(time=torch.ones(3),gradm=torch.ones(3,3),adc_mask=torch.tensor([False,True,False]),adc_phase=torch.zeros(3))
    assert set(seqmod.Sequence.soa_collapse_keep_adcs(s))==set(s)

@pytest.mark.regression
def test_collapse_preserves_adc_count(seqmod):
    s=dict(time=torch.ones(3),gradm=torch.ones(3,3),adc_mask=torch.tensor([True,False,True]),adc_phase=torch.zeros(3))
    assert seqmod.Sequence.soa_collapse_keep_adcs(s)['adc_mask'].sum()==2


@pytest.mark.regression
def test_fm_waveforms_have_distinct_cache_keys(main):
    s=dict(kind='bloch',t=np.linspace(0,.001,5),rf_amp=np.ones(5)*100,rf_phase=np.zeros(5),gz=np.zeros(5),rf_freq=0.)
    assert main._bloch_kernel_signature(dict(s,rf_fm=0.))[0]!=main._bloch_kernel_signature(dict(s,rf_fm=100.))[0]

@pytest.mark.regression
def test_frequency_array_is_supported(main):
    s=dict(kind='bloch',t=np.linspace(0,.001,5),rf_amp=np.ones(5),rf_phase=np.zeros(5),gz=np.zeros(5),rf_freq=np.arange(5.))
    assert np.isfinite(main._bloch_kernel_signature(s)[1]).all()

"""Additional checks for the source fixes; the original independent tests are unchanged."""
import numpy as np
import pytest
import torch


def base_span():
    return dict(kind='bloch',t=np.linspace(0,.001,5),rf_amp=np.ones(5)*100.,
                rf_phase=np.zeros(5),gz=np.zeros(5),rf_freq=0.)


def test_cache_includes_nonmidpoint_frame_reference(main):
    s=base_span()
    a=main._bloch_kernel_signature(dict(s,rf_center=.0002))[0]
    b=main._bloch_kernel_signature(dict(s,rf_center=.0006))[0]
    assert a!=b


def test_cache_does_not_quantize_away_weak_rf(main):
    s=base_span()
    a=main._bloch_kernel_signature(dict(s,rf_amp=np.ones(5)))[0]
    b=main._bloch_kernel_signature(dict(s,rf_amp=np.ones(5)*2))[0]
    assert a!=b

@pytest.mark.parametrize('fm',[None,np.array(0.),np.array(20.)])
def test_cache_accepts_missing_and_scalar_array_modulation(main,fm):
    result=main._bloch_kernel_signature(dict(base_span(),rf_freq=np.array(10.),rf_fm=fm))
    expected=10.+(0. if fm is None else float(fm))
    np.testing.assert_allclose(result[2],expected)
    assert np.isfinite(result[1]).all()


def test_clone_cpu_preserve_all_sequence_metadata_and_effective_gradient(seqmod,repetition):
    repetition.gradm_z_eff=torch.tensor([7.,8.,9.])
    repetition.raw_waveform={'samples':np.array([1.,2.])}
    seq=seqmod.Sequence([repetition],mrtini_hybrid=[{'kind':'epg','values':np.array([3.])}],
        mrtini_rep_bounds=[(0.,1.)],mrtini_spans_by_rep=[[0]])
    seq.mrtini_pad=.00015;seq.mrtini_epg_indices=[0]
    # Metadata tensors can be non-leaf differentiable tensors.
    leaf=torch.tensor([1.],requires_grad=True)
    seq.metadata_tensor=leaf*2
    for cloned in [seq.clone(),seq.cpu()]:
        torch.testing.assert_close(cloned[0].gradm_z_eff,repetition.gradm_z_eff)
        assert cloned.mrtini_pad==seq.mrtini_pad
        assert cloned.mrtini_epg_indices==seq.mrtini_epg_indices
        cloned.mrtini_hybrid[0]['values'][0]=99
        assert seq.mrtini_hybrid[0]['values'][0]==3
        cloned[0].raw_waveform['samples'][0]=99
        assert seq[0].raw_waveform['samples'][0]==1
        torch.testing.assert_close(cloned.metadata_tensor,seq.metadata_tensor)
        assert cloned.metadata_tensor is not seq.metadata_tensor


def test_collapse_preserves_adc_times_phases_and_gradient_positions(seqmod):
    t=torch.tensor([.1,.2,.3,.4,.5]);G=torch.arange(15,dtype=torch.float32).reshape(3,5)
    mask=torch.tensor([False,True,False,True,False]);phi=torch.tensor([0.,.2,0.,.4,0.])
    inp=dict(time=t,gradm=G,adc_mask=mask,adc_phase=phi)
    out=seqmod.Sequence.soa_collapse_keep_adcs(inp)
    torch.testing.assert_close(out['time'].cumsum(0)[out['adc_mask']],t.cumsum(0)[mask])
    torch.testing.assert_close(out['gradm'].cumsum(1)[:,out['adc_mask']],G.cumsum(1)[:,mask])
    torch.testing.assert_close(out['adc_phase'][out['adc_mask']],phi[mask])
    torch.testing.assert_close(out['time'].sum(),t.sum())
    torch.testing.assert_close(out['gradm'].sum(1),G.sum(1))
