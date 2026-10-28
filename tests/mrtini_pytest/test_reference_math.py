"""Sanity and fault-detection checks for the independent references themselves.
These tests run without a simulator checkout; they are NOT simulator validation.
"""
import ast
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from reference_math import (assert_so3,assert_unitary,channel_matrix,dft_matrix,
    fourier_k,gradient_area_zoh,demod_reference,solve_zoh_ivp,z_rotation,synthesis)

@pytest.mark.parametrize('n',[7,8,9,16])
def test_reference_fourier_roundtrip_and_parseval(n):
    z=(np.arange(n)-n//2)*.001+.00013;k=fourier_k(z)
    x=np.random.default_rng(21).normal(size=n)+1j*np.random.default_rng(22).normal(size=n)
    c=x@dft_matrix(z,k)
    np.testing.assert_allclose(synthesis(c,z,k),x,atol=3e-14)
    np.testing.assert_allclose(np.sum(abs(c)**2),np.mean(abs(x)**2),atol=3e-14)

@pytest.mark.parametrize('axis',[0,1,2])
def test_reference_ivp_against_closed_form(axis):
    rf=[.4+0j,.4+0j] if axis==0 else ([.4j,.4j] if axis==1 else [0j,0j])
    R=solve_zoh_ivp(t=[0,.2],rf=rf,gz=[0,0],z=0,b0=.4 if axis==2 else 0,gamma=2.)
    expected=Rotation.from_rotvec(np.eye(3)[axis]*.16).as_matrix()
    np.testing.assert_allclose(R,expected,atol=2e-12)
    assert_so3(R,atol=2e-12);assert_unitary(channel_matrix(R),atol=2e-12)


def test_reference_detects_wrong_matrix_and_complex_normalization():
    with pytest.raises(AssertionError): assert_so3(np.diag([1.,1.,-1.]))
    with pytest.raises(AssertionError): assert_so3(np.diag([1.,1.,.9]))
    with pytest.raises(AssertionError): assert_unitary(2*channel_matrix(np.eye(3)))


def test_reference_noncommuting_demodulation_and_wrong_side_detection():
    t=np.array([0.,.2,.7,1.]);g=np.array([1.,3.,-2.,999.]);z=np.array([-.3,.1,.4]);center=.4
    pre,post=gradient_area_zoh(t,g,center)
    assert pre==pytest.approx(.8) and post==pytest.approx(.3)
    mix=Rotation.from_rotvec([.5,.2,-.1]).as_matrix()
    full=z_rotation(2*post*z)@mix@z_rotation(2*pre*z)
    out=demod_reference(full,z,t,g,2,center)
    np.testing.assert_allclose(out,np.broadcast_to(mix,out.shape),atol=2e-15)
    wrong=z_rotation(-2*pre*z)@full@z_rotation(-2*post*z)
    assert np.max(abs(wrong-mix))>.01


def test_reference_detects_fourier_sign_and_odd_shift_errors():
    n=9;z=(np.arange(n)-4)*.001;k=fourier_k(z)
    x=np.exp(2j*np.pi*k[6]*z);expected=x@dft_matrix(z,k)
    wrong_sign=x@dft_matrix(z,k).conj()
    wrong_shift=x@np.roll(dft_matrix(z,k),1,axis=0)
    assert np.max(abs(expected-wrong_sign))>.5
    assert np.max(abs(expected-wrong_shift))>.5


def test_no_production_diagnostic_or_golden_checker_calls():
    forbidden={'check_unitarity_Rch','check_demod','check_rotations','test_bin_identity',
               'test_rch6_matches_gold','gold6_from_Rxyz','rch6_from_q_old','S_minus','S_fplus',
               'sanity_check_quantization','fit_linear_k_from_mplus','get_Rmix_from_R'}
    for path in Path(__file__).parent.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in forbidden,f'{path.name}:{node.lineno} calls {node.func.attr}'


def test_reference_module_never_imports_production_code():
    path=Path(__file__).with_name('reference_math.py')
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n,ast.Import): roots=[x.name for x in n.names]
        elif isinstance(n,ast.ImportFrom): roots=[n.module or '']
        else: continue
        assert not any(x.startswith(('src','MRzero','mrtini')) for x in roots)
