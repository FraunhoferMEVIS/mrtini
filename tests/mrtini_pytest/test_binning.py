import numpy as np
import pytest
from reference_math import GAMMA_HZ, GAMMA_RAD

@pytest.mark.parametrize('x,w,result',[([3,1,2],[1,8,1],1),([1,3],[0,0],2),([np.nan],[1],1)])
def test_weighted_median(sm,x,w,result): assert sm.weighted_median(x,w)==result

@pytest.mark.parametrize('mode,result',[('wmean',2.5),('wmedian',3.)])
def test_pd_weighted_b1_ref(sm,mode,result): assert sm.pd_weighted_b1_ref([1,3],[1,3],mode=mode)==pytest.approx(result)

@pytest.mark.parametrize('x,result',[([np.nan],1.),([1,3],2.)])
def test_pd_ref_fallback(sm,x,result): assert sm.pd_weighted_b1_ref(x,np.zeros(len(x)),mode='wmean')==result


def test_weighted_centers_with_empty_bin(sm):
    v=np.array([1.,3.,8.]); labels=np.array([1,1,3]);w=np.array([1.,3.,1.])
    c=sm.bin_centers_weighted(v,labels,3,w)
    np.testing.assert_allclose(c,[2.5,np.nan,8.],equal_nan=True)
    np.testing.assert_allclose(sm.bin_centers_weighted(v,labels,3),[2.,np.nan,8.],equal_nan=True)

@pytest.mark.parametrize('right',[False,True])
def test_uniform_binning_mask_quantization_error_bound(sm,right):
    x=np.array([-5.,0.,4.,np.nan,np.inf,100.]);mask=np.array([1,1,1,1,1,0])
    c=sm.bin_map_uniform(x,2.,mask=mask,right=right)
    assert c.labels.tolist()[3:]==[0,0,0]
    q=sm.quantize_from_labels(c.labels,c.center_by_label)
    assert np.isnan(q[3:]).all()
    assert np.max(abs(q[:3]-x[:3]))<=1+1e-6
    assert c.labels.dtype==np.int32

@pytest.mark.parametrize('x,width',[(np.array([]),2),(np.array([1.]),0),(np.array([1.]),-1)])
def test_disabled_binning(sm,x,width): assert sm.bin_map_uniform(x,width).labels is None


def test_uniform_binning_constant_invalid_and_shape(sm):
    c=sm.bin_map_uniform(np.full(3,2.),.1)
    assert c.labels.tolist()==[1,1,1]
    np.testing.assert_allclose(c.centers,[2.])
    c=sm.bin_map_uniform(np.array([np.nan,np.inf]),.1)
    assert c.labels.tolist()==[0,0] and c.center_by_label=={}
    with pytest.raises(ValueError,match='1D'): sm.bin_map_uniform(np.zeros((2,2)),1)
    np.testing.assert_allclose(sm._make_uniform_edges(np.array([2.]),0),[1.,3.])

@pytest.mark.parametrize('mode',['scale','tesla'])
@pytest.mark.parametrize('explicit_width',[True,False])
def test_b1_modes_reference_and_clamp(sm,mode,explicit_width):
    x=np.array([1.,2.,3.,np.nan])
    y,c=sm.bin_B1_map(x,mode=mode,ref=2.,bin_width=.1 if explicit_width else None,nbins=4,clamp=(.75,1.25))
    expected=np.clip(x/2 if mode=='scale' else x,.75,1.25)
    np.testing.assert_allclose(y,expected,equal_nan=True)
    assert c.labels[-1]==0

@pytest.mark.parametrize('kw,match',[({'mode':'invalid'},'mode'),({'nbins':0},'positive nbins')])
def test_b1_errors(sm,kw,match):
    with pytest.raises(ValueError,match=match): sm.bin_B1_map(np.ones(3),**kw)


def test_b1_automatic_reference_edge_cases(sm):
    y,c=sm.bin_B1_map(np.array([1.,2.,3.]),PD=np.array([1.,8.,1.]),bin_width=.1)
    np.testing.assert_allclose(y,[.5,1.,1.5])
    for x in [np.zeros(3),np.array([np.nan,np.inf])]:
        y,c=sm.bin_B1_map(x)
        assert len(y)==len(x)
    with pytest.raises(ValueError,match='1D'): sm.bin_B1_map(np.ones((2,2)))


def test_joint_pairs_partition_valid_voxels_and_match_centers(sm):
    b0=np.array([0.,0.,10.,10.,np.nan,3.]);b1=np.array([1.,1.,.8,1.2,1.,1.])
    mask=np.array([1,1,1,1,1,0])
    jb=sm.build_joint_binning(b0,b1,b1_ref=1.,b0_bin_hz=5,b1_bin_width=.1,mask=mask)
    assert jb.pair_ids[4:].tolist()==[-1,-1]
    np.testing.assert_array_equal(np.sort(np.concatenate(jb.voxel_groups)),np.arange(4))
    for pid,group in enumerate(jb.voxel_groups):
        assert (jb.pair_ids[group]==pid).all()
        l0,l1=jb.pairs[pid]
        np.testing.assert_allclose(jb.pair_centers[pid],[jb.b0.center_by_label[l0],jb.b1.center_by_label[l1]])
    # Reconstruct directly from labels; no production quantization checker.
    q0=np.full(len(b0),np.nan);q1=np.full(len(b1),np.nan)
    for i in range(4):
        q0[i]=jb.b0.center_by_label[int(jb.b0.labels[i])]
        q1[i]=jb.b1.center_by_label[int(jb.b1.labels[i])]
    assert np.max(np.abs(q0[:4]-b0[:4]))<=2.5+1e-6
    assert np.max(np.abs(q1[:4]-b1[:4]))<=.05+1e-6
    assert np.isnan(q0[4:]).all() and np.isnan(q1[4:]).all()


def test_joint_empty_shape_error_and_missing_labels(sm):
    with pytest.raises(ValueError,match='same shape'): sm.build_joint_binning([0,1],[1])
    jb=sm.build_joint_binning([np.nan],[1.])
    assert jb.pairs.shape==(0,2) and jb.voxel_groups==[]
    assert jb.pair_ids.tolist()==[-1]
    assert jb.pair_centers.shape==(0,2)

@pytest.mark.parametrize('width',[0.,2.])
def test_legacy_b0_conversion_hz_to_tesla(sm,width):
    c=sm.bin_B0_map(np.array([[0.,1.],[2.,3.]]),bin_hz=width)
    if width==0: assert c.bin_labels is None
    else:
        for lab,hz in c.bin_center_hz.items(): assert c.bin_center_map_T[lab]==pytest.approx(hz/GAMMA_HZ)


@pytest.mark.parametrize('rephaser',[False,True])
def test_slice_waveform_scaling_bandwidth_and_rephaser(sm,rephaser,monkeypatch):
    c=sm.build_slice_selective(N=128,tbw=4,dt_s=1e-5,slice_thickness_m=.005,flip_angle_deg=30.,
                               rf_designer=lambda n,tbw:np.ones(n,dtype=complex),add_rephaser=rephaser)
    assert c.t.shape==c.rf_complex_T.shape==c.gz_Tm.shape
    assert np.all(np.diff(c.t)>0)
    assert c.bandwidth_hz==pytest.approx(4/(128e-5))
    assert c.gz_peak_Tm==pytest.approx(c.bandwidth_hz/(GAMMA_HZ*.005))
    assert c.flip_angle_rad==pytest.approx(np.pi/6)
    assert abs(c.B1_scale)*GAMMA_RAD*(127e-5)==pytest.approx(np.pi/6)
    assert np.isfinite(c.rf_complex_T).all() and np.isfinite(c.gz_Tm).all()
    assert np.any(c.gz_Tm<0)==rephaser
    assert c.gz_area_Tm_s==pytest.approx(c.gz_peak_Tm*c.T_pulse_s,rel=1e-5)


def test_slice_default_designer_and_reject_invalid_rf(sm,monkeypatch):
    c=sm.build_slice_selective(N=64,tbw=4,dt_s=2e-5,add_rephaser=False,gmax_Tm=.03)
    assert np.isfinite(c.B1_scale)
    with pytest.raises(AssertionError,match='length-N'):
        sm.build_slice_selective(N=64,rf_designer=lambda n,tbw:np.ones(3))
    with pytest.raises(ValueError,match='RF area is zero'):
        sm.build_slice_selective(N=64,dt_s=2e-5,rf_designer=lambda n,tbw:np.zeros(n))



def test_joint_binning_handles_numpy_inverse_index_shape(sm):
    # Test unmodified NumPy and production code: no monkeypatch or expected failure.
    result=sm.build_joint_binning([0.,10.],[1.,1.],b1_ref=1.)
    assert result.pair_ids.shape==(2,)
    assert result.pair_ids.dtype==np.int32
