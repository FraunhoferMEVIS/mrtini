import math
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from reference_math import (assert_so3,demod_reference,gradient_area_zoh,solve_zoh_ivp,z_rotation)

@pytest.mark.parametrize('omega', [[0,0,0],[1e-9,0,0],[1,0,0],[0,2,0],[0,0,-3],[0.2,-0.3,0.7]])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_so3_exponential_matches_independent_scipy(bloch, omega, dtype):
    w = torch.tensor(omega, dtype=dtype)
    got = bloch.so3_exp(w, 0.7)
    expected = Rotation.from_rotvec(np.array(omega)*0.7).as_matrix()
    np.testing.assert_allclose(got.numpy(), expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(got.T @ got, torch.eye(3,dtype=dtype), atol=2e-6, rtol=2e-6)
    assert torch.det(got).item() == pytest.approx(1, abs=2e-6)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_batched_spinor_and_composition_match_matrix_products(bloch, dtype):
    w = torch.tensor([[0.,0.,0.],[1e-9,0,0],[.3,-.7,1.2],[2.,1.,-.5]], dtype=dtype)
    a,b = bloch.spinor_step_from_omega(w, .2)
    expected = torch.tensor(Rotation.from_rotvec(w.numpy()*.2).as_matrix(), dtype=dtype)
    torch.testing.assert_close(bloch.spinor_to_Rxyz(a,b), expected, atol=2e-6, rtol=2e-6)
    a2,b2 = bloch.spinor_step_from_omega(w.flip(0), .4)
    ac,bc = bloch.spinor_compose(a2,b2,a,b)
    torch.testing.assert_close(bloch.spinor_to_Rxyz(ac,bc), torch.tensor(Rotation.from_rotvec(w.flip(0).numpy()*.4).as_matrix(), dtype=dtype) @ expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(a.abs()**2+b.abs()**2, torch.ones(4,dtype=dtype))
    torch.testing.assert_close(bloch.spinor_to_Rxyz(3*a,3*b), expected, atol=2e-6, rtol=2e-6)

@pytest.mark.parametrize('t, expected', [(0.,(1.,2.,3.)),(.5,(2.,3.,4.)),(1.,(3.,4.,5.))])
def test_linear_control(bloch,t,expected):
    ctrl = bloch.InterpSharedLinear([0,1], [1+2j,3+4j], [3,5], device='cpu')
    np.testing.assert_allclose([x.item() for x in ctrl(torch.tensor(t))],expected)

def test_linear_control_rejects_duplicate_times(bloch):
    with pytest.raises(AssertionError, match='strictly increasing'):
        bloch.InterpSharedLinear([0,0], [0j,0j], [0,0],device='cpu')

@pytest.mark.parametrize('t,idx', [(-1,0),(0,0),(.49,0),(.5,1),(1.,2),(10.,2)])
def test_zoh_control_clamps_and_selects_samples(bloch,t,idx):
    c = bloch.InterpSharedZOH(0,.5,[1+2j,3+4j,5+6j],torch.tensor([10.,20.,30.]),[7,8,9],device='cpu')
    got = [v.item() for v in c(torch.tensor(t))]
    assert got == [1+2*idx,2+2*idx,10+10*idx,7+idx]

@pytest.mark.parametrize('relax',[False,True])
def test_bloch_rhs_cross_product_and_relaxation(bloch,relax):
    def control(t): return (torch.tensor(.2),torch.tensor(-.1),torch.tensor(0.),torch.tensor(.3))
    rhs = bloch.BlochRHS(control,[-.5,.5],[.1,-.2],[.8,1.2], gamma_rad=2.,
                         T1=2. if relax else None,T2=.5 if relax else None,device='cpu')
    M = torch.arange(36,dtype=torch.float32).reshape(2,3,2,3)/20
    B = torch.tensor([[[.16,-.08,-.05],[.16,-.08,.25]],[[.24,-.12,-.35],[.24,-.12,-.05]]])[:,None]
    expected = 2*torch.cross(B.expand_as(M),M,dim=-1)  # same active-rotation convention as solve_slice tests
    if relax:
        expected[...,:2] -= M[...,:2]/.5
        expected[...,2] -= (M[...,2]-1)/2
    torch.testing.assert_close(rhs(torch.tensor(0.),M),expected,atol=2e-6,rtol=2e-6)


def solve_args(n=9):
    return dict(rf_complex=np.zeros(n,complex),rf_freq=np.zeros(n),gz=np.zeros(n),
                t_grid=np.linspace(0,.001,n),z_m=[-.001,0,.001],b0_T=[0.],
                chem_shift_Hz=[0.],device='cpu',dtype_policy=torch.float64)

@pytest.mark.parametrize('backend',['so3','spinor'])
@pytest.mark.parametrize('axis',[0,1,2])
def test_solve_constant_field_against_analytic_rotation(bloch,backend,axis):
    a=solve_args(); gamma=2*math.pi*42.577478e6; field=math.pi/2/(gamma*.001)
    if axis==0: a['rf_complex'][:]=field
    elif axis==1: a['rf_complex'][:]=1j*field
    else: a['b0_T']=[field]
    out,debug=bloch.solve_slice(**a,backend=backend,return_debug=True)
    expected=torch.tensor(Rotation.from_rotvec(np.eye(3)[axis]*math.pi/2).as_matrix(),dtype=torch.float64)
    got=out.Rz_solve[0].Rz_B33Nz[0].permute(2,0,1)
    torch.testing.assert_close(got, expected.expand(3,-1,-1),atol=2e-6,rtol=2e-6)
    assert out.Rz_solve[0].name=='water'
    assert out.Rz_solve[0].to_dict()['chem_shift_Hz']==0
    assert set(debug)=={'t','z','rf','gz','Rz','Rz_mix'}

@pytest.mark.parametrize('backend',['so3','spinor'])
def test_solve_species_bins_and_spinor_renormalization(bloch,backend):
    a=solve_args(40); a.update(chem_shift_Hz=[0.,-412.],b0_T=[0.,1e-7],b1_scale=[.8,1.2])
    a['rf_complex'][:]=1e-6
    out=bloch.solve_slice(**a,backend=backend)
    assert [s.name for s in out.Rz_solve]==['water','fat']
    for s in out.Rz_solve:
        assert s.Rz_B33Nz.shape==(2,3,3,3)
        R=s.Rz_B33Nz.permute(0,3,1,2)
        torch.testing.assert_close(R.mT@R,torch.eye(3,dtype=R.dtype).expand_as(R),atol=3e-6,rtol=3e-6)
    assert not torch.allclose(out.Rz_solve[0].Rz_B33Nz,out.Rz_solve[1].Rz_B33Nz)

@pytest.mark.parametrize('change,match', [({'b1_scale':[1.,2.]},'b1_scale'),({'t_grid':[0.]},'at least 2'),({'backend':'invalid'},'Unknown backend')])
def test_solve_input_errors(bloch,change,match):
    a=solve_args();a.update(change)
    with pytest.raises(ValueError,match=match): bloch.solve_slice(**a)

@pytest.mark.parametrize('shift,name',[(0,'water'),(-412,'fat'),(100,'unknown')])
def test_species_names(bloch,shift,name): assert bloch.get_sp_name(shift)==name


@pytest.mark.parametrize('center',[0.,.15,.4,.83,1.])
def test_demodulation_against_independent_zoh_frame_change(bloch,center):
    # A noncommuting RF rotation makes wrong multiplication order observable.
    # Asymmetric, nonconstant Gz distinguishes ZOH from trapezoidal integration.
    t=np.array([0.,.1,.4,.75,1.]);gz=np.array([.2,-.7,.4,1.1,13.])
    z=np.array([-.3,0.,.17]);gamma=3.
    mix=Rotation.from_rotvec([[.3,-.2,.1],[-.2,.5,.1]]).as_matrix()
    pre,post=gradient_area_zoh(t,gz,center)
    full=z_rotation(gamma*post*z)[None]@mix[:,None]@z_rotation(gamma*pre*z)[None]
    # Production layout is (species, bank, Cartesian out, Cartesian in, z).
    inp=torch.tensor(full[None],dtype=torch.float64).permute(0,1,3,4,2)
    got=bloch.demod_Rxyz_to_Rmix_center(Rz_SB33Nz=inp,z_m=torch.tensor(z),
        t=torch.tensor(t),gz=torch.tensor(gz),gamma_rad=gamma,rf_center_time=center)
    assert isinstance(got,torch.Tensor), 'Demodulation contract returns one Cartesian tensor'
    assert got.shape==inp.shape, 'Species/bank/z dimensions must be preserved'
    expected=np.broadcast_to(mix[:,None],full.shape)[None]
    cart=got.permute(0,1,4,2,3).detach().cpu().numpy()
    np.testing.assert_allclose(cart,expected,atol=2e-12,rtol=2e-12)
    assert_so3(cart,atol=3e-12)
    # Independently restore the physical transfer matrix as a second check.
    restored=z_rotation(gamma*post*z)@cart@z_rotation(gamma*pre*z)
    np.testing.assert_allclose(restored,full[None],atol=3e-12,rtol=3e-12)

@pytest.mark.parametrize('backend',['so3','spinor'])
@pytest.mark.parametrize('center',[.00037,.0006])
def test_shaped_slice_solver_against_independent_adaptive_bloch_ode(bloch,backend,center):
    t=np.linspace(0,.0012,7)
    rf=np.array([0.,1.,2.,1.4,.6,.2,0.])*2e-6*np.exp(1j*np.array([0.,.1,.5,.8,.4,-.2,0.]))
    gz=np.array([.003,.006,.004,-.002,.001,0.,.009])
    z=np.array([-.0013,0.,.0011]);b0=np.array([0.,1.1e-7]);b1=np.array([.85,1.15]);species=[0.,-412.]
    args=dict(rf_complex=rf,rf_freq=np.zeros_like(t),gz=gz,t_grid=t,z_m=z,
              b0_T=b0,b1_scale=b1,chem_shift_Hz=species,device='cpu',dtype_policy=torch.float64,
              backend=backend,rf_center_time=center)
    out=bloch.solve_slice(**args)
    for si,chemical in enumerate(species):
        expected=np.empty((len(b0),len(z),3,3))
        for bi in range(len(b0)):
            for zi,position in enumerate(z):
                expected[bi,zi]=solve_zoh_ivp(t=t,rf=rf,gz=gz,z=position,
                    b0=b0[bi],b1=b1[bi],chemical_hz=chemical)
        expected=demod_reference(expected,z,t,gz,2*np.pi*42.577478e6,center)
        got=out.Rz_solve[si].Rz_B33Nz.permute(0,3,1,2).detach().cpu().numpy()
        np.testing.assert_allclose(got,expected,atol=2e-5,rtol=2e-5)
        assert_so3(got,atol=2e-5)
