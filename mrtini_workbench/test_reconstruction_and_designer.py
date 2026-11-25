"""Trajectory-aware recon and live compiler numerical contracts."""
from dataclasses import replace
import json
import numpy as np
import pytest
from physics import Protocol,reconstruct
from analysis_views import sequence_window
from test_composer_and_profiles import p,check

@pytest.mark.parametrize('style',['epi','spiral_in'])
def test_new_readouts(style):
    _,d=check(p([{'type':'excitation'},{'type':'readout','style':style,'lines':4,'rewind':False}],tr_ms=100))
    assert len(d.adc_times)==4096

def test_parallel_gradient_preserves_moments_and_preview():
    _,d=check(p([{'type':'excitation'},{'type':'parallel_gradient','kx':100,'ky':-70,'kz':10,'kx_duration_ms':2,'ky_duration_ms':1,'ky_offset_ms':.5,'kz_offset_ms':1},{'type':'readout'}],tr_ms=35))
    w=sequence_window(design=d,shot_end=.035)
    g=np.array(w['gradients']);assert np.all(g[1:,0]>=g[:-1,1]-1e-11)
    np.testing.assert_allclose(np.sum((g[:,1]-g[:,0])[:,None]*g[:,2:],axis=0)*42577.47892,d.area(0,.035),atol=1e-5)
    assert len(w['adc'])==64

def test_full_shot_preview():
    blocks=[{'type':'excitation'}]+[b for _ in range(6) for b in [{'type':'refocusing'},{'type':'readout'}]]
    _,d=check(p(blocks,tr_ms=200))
    w=sequence_window(design=d,shot_end=.2)
    assert len(w['pulses'])==7 and len(w['adc'])==6*64 and w['hi']==.2

@pytest.mark.parametrize('n',[8,16])
def test_cartesian_fft_matches_direct(n):
    q=Protocol(resolution=n,fov_mm=256);a=np.arange(n)-n//2;Y,X=np.meshgrid(a,a,indexing='ij');k=np.c_[X.ravel(),Y.ravel()]/.256
    rng=np.random.default_rng(12);s=rng.normal(size=(len(k),2))+1j*rng.normal(size=(len(k),2))
    # Include repeated samples and shuffled acquisition order.
    i=rng.permutation(len(k));i=np.r_[i,i[:8]];k=k[i];s=s[i]
    out=reconstruct(s,k,q);ref=reconstruct(s,k,replace(q,recon='direct'))
    np.testing.assert_allclose(out['complex_image'],ref['complex_image'],atol=1e-12);assert out['on_grid']

def test_offgrid_direct_point():
    q=Protocol(resolution=16,recon='direct');rng=np.random.default_rng(2);k=rng.uniform(-20,20,(53,2));pos=np.array([3,-2])*.256/16;s=np.exp(2j*np.pi*k@pos)
    im=reconstruct(s,k,q);assert np.unravel_index(im['magnitude'].argmax(),(16,16))==(6,11)

def test_nufft_matches_direct_if_installed():
    pytest.importorskip('torchkbnufft');q=Protocol(resolution=16,recon='direct');rng=np.random.default_rng(5);k=rng.uniform(-25,25,(256,2));s=(rng.normal(size=(256,1))+1j*rng.normal(size=(256,1))).astype(np.complex64)
    a=reconstruct(s,k,q)['complex_image'];b=reconstruct(s,k,replace(q,recon='nufft'))['complex_image'];assert np.linalg.norm(a-b)/np.linalg.norm(a)<.01

def test_dicom_monochrome_if_installed():
    pd=pytest.importorskip('pydicom');import io,base64
    from pydicom.dataset import FileDataset,FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian,MRImageStorage,generate_uid
    from dicom_view import decode_dicom
    m=FileMetaDataset();m.TransferSyntaxUID=ExplicitVRLittleEndian;m.MediaStorageSOPClassUID=MRImageStorage;m.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(None,{},file_meta=m,preamble=b'\0'*128);ds.SOPClassUID=MRImageStorage;ds.SOPInstanceUID=m.MediaStorageSOPInstanceUID;ds.SeriesInstanceUID=generate_uid();ds.PatientName='Test^Private';ds.Rows=2;ds.Columns=3;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2';ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0;ds.RescaleSlope=2;ds.RescaleIntercept=-3;ds.PixelData=np.arange(6,dtype='<u2').tobytes();buf=io.BytesIO();ds.save_as(buf,enforce_file_format=True)
    d=decode_dicom(buf.getvalue());np.testing.assert_equal(np.frombuffer(base64.b64decode(d['pixels']),'<f4'),np.arange(6)*2-3);assert 'Private' not in str(d)
