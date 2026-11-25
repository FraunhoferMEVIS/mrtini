import math
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from reference_math import (channel_six,channel_matrix,assert_so3,assert_unitary,
    dft_matrix,fourier_k,fourier_orders,synthesis,matrix_from_six)

NAMES=('pp','pz','mp','mz','zp','zz')

@pytest.mark.parametrize('x,w,expected', [([3,1,2],[1,8,1],1),([1,3],[0,0],2),([np.nan],[1],1),([1,9,np.inf],[1,-1,8],1)])
def test_weighted_median(rf,x,w,expected): assert rf.weighted_median(x,w)==expected

@pytest.mark.parametrize('mode,expected',[('wmean',2.5),('wmedian',3.)])
def test_b1_reference(rf,mode,expected):
    assert rf.pd_weighted_b1_ref([1,3,np.nan],[1,3,4],mode)==pytest.approx(expected)

@pytest.mark.parametrize('values,expected',[([np.nan],1.),([1,3],2.)])
def test_b1_reference_fallback(rf,values,expected):
    assert rf.pd_weighted_b1_ref(values,np.zeros(len(values)),'wmean')==expected


def test_map_clamp_finiteness_and_pd_mass(rf):
    ax=rf.make_axes([10.,-10.],[1.2,.8],device='cpu')
    im=rf.build_imap([-99,0,99,np.nan],[.1,1,5,1],ax,device='cpu')
    torch.testing.assert_close(im.tx[:3],torch.tensor([0.,.5,1.],dtype=rf.DTYPE))
    torch.testing.assert_close(im.ty[:3],torch.tensor([0.,.5,1.],dtype=rf.DTYPE))
    assert im.valid.tolist()==[True,True,True,False]
    W=rf.pair_weights_from_imap_pd(im,ax,[1.,2.,3.,100.],'cpu')
    torch.testing.assert_close(W,torch.tensor([[1.5,.5],[.5,3.5]],dtype=rf.DTYPE))
    assert W.sum().item()==pytest.approx(6.)
    assert not rf.pair_weights_from_imap_pd(im,ax,[0.,-1.,np.nan,1.],'cpu').any()
    extrap=rf.build_imap([-20.],[1.],ax,device='cpu',clamp=False)
    assert extrap.tx.item()<0

@pytest.mark.parametrize('angles',[[0,0,0],[math.pi,0,0],[0,math.pi,0],[0,0,math.pi],[.3,-.7,1.1]])
def test_quaternion_rch6_against_independent_basis_transform(rf,angles):
    rotation=Rotation.from_rotvec(angles)
    R=rotation.as_matrix()[None]
    # SciPy quaternion order is x,y,z,w; MRTini input contract is w,x,y,z.
    xyzw=rotation.as_quat();qref=np.concatenate([xyzw[3:],xyzw[:3]])[None]
    q=rf.R_to_q(torch.tensor(R,dtype=rf.DTYPE)).detach().cpu().numpy()
    # q and -q describe the same rotation. Compare projectors, not arbitrary signs.
    np.testing.assert_allclose(q[..., :,None]*q[...,None,:],qref[..., :,None]*qref[...,None,:],atol=2e-6)
    got=rf.rch6_from_q(torch.tensor(qref,dtype=rf.DTYPE))
    expected=channel_six(R)
    for name,g in zip(NAMES,got):
        np.testing.assert_allclose(g.detach().cpu(),expected[name],atol=2e-6,rtol=2e-6)
    actual=matrix_from_six({name:g.detach().cpu().numpy() for name,g in zip(NAMES,got)})
    assert_unitary(actual)


@pytest.mark.parametrize('sign',[1.,-1.])
def test_slerp_shortest_arc_and_midpoint(rf,sign):
    q0=torch.tensor([[1.,0,0,0]],dtype=rf.DTYPE)
    q1=sign*torch.tensor([[math.sqrt(.5),0,0,math.sqrt(.5)]],dtype=rf.DTYPE)
    qm=rf.q_slerp(q0,q1,torch.tensor(.5))
    expected=torch.tensor([[math.cos(math.pi/8),0,0,math.sin(math.pi/8)]],dtype=rf.DTYPE)
    torch.testing.assert_close(qm,expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(rf.q_slerp(q0,q0,torch.tensor([.4])),q0)


def test_coeff_selection_phase_group_and_energy(rf):
    c=rf.Coeff6(**{k:torch.tensor([1+2j,3-4j],dtype=rf.CMPLX_DTYPE) for k in NAMES})
    sub=c.select_k(torch.tensor([1,0]))
    for k in NAMES: torch.testing.assert_close(getattr(sub,k),getattr(c,k).flip(0))
    a=c.with_rf_phase(.3).with_rf_phase(-.7); b=c.with_rf_phase(-.4)
    for k in NAMES: torch.testing.assert_close(getattr(a,k),getattr(b,k))
    torch.testing.assert_close(rf.energy_from_coeff6(c),torch.tensor([30.,150.],dtype=rf.DTYPE))
    torch.testing.assert_close(rf.energy_from_coeff6(a),rf.energy_from_coeff6(c))
    phi=.3; phased=c.with_rf_phase(phi)
    for k,n in {'pp':0,'pz':-1,'mp':2,'mz':1,'zp':1,'zz':0}.items():
        torch.testing.assert_close(getattr(phased,k),getattr(c,k)*np.exp(1j*n*phi))

@pytest.mark.parametrize('n',[8,9])
@pytest.mark.parametrize('bake',[False,True])
@pytest.mark.parametrize('norm',['K','Lz','dz'])
def test_fourier_projector_against_independent_complex_dft(rf,n,bake,norm):
    # Nonzero coordinate origin and asymmetric complex data expose phase errors.
    z=(np.arange(n)-n//2)*.001+.00013
    k=fourier_k(z)
    x=(np.arange(n)+1j*np.arange(n)[::-1])/n
    P=rf.make_projector_z_to_k(torch.tensor(z,dtype=rf.DTYPE),
        torch.tensor(k,dtype=rf.DTYPE),norm=norm,bake_ifftshift=bake)
    expected_matrix=dft_matrix(z,k,norm)
    np.testing.assert_allclose(P.detach().cpu(),expected_matrix,atol=3e-6,rtol=3e-6)
    got=(torch.tensor(x,dtype=rf.CMPLX_DTYPE)@P).detach().cpu().numpy()
    np.testing.assert_allclose(got,x@expected_matrix,atol=8e-6,rtol=8e-6)
    if norm in ('K','Lz'):
        np.testing.assert_allclose(synthesis(got,z,k),x,atol=3e-5,rtol=3e-5)
        assert np.sum(abs(got)**2)==pytest.approx(np.mean(abs(x)**2),rel=2e-5,abs=2e-6)



def test_projector_rejects_unknown_normalization(rf,zgrid):
    with pytest.raises(ValueError,match='norm'):
        rf.make_projector_z_to_k(zgrid,torch.tensor([0.]),norm='invalid')

@pytest.mark.parametrize('eps',[0.,.05,.3])
def test_energy_support_bounds_discarded_energy(rf,eps):
    grid=rf.KGrid(torch.arange(3),torch.tensor([-1,0,1]),1.,1)
    c=rf.Coeff6(**{k:torch.zeros(3,dtype=rf.CMPLX_DTYPE) for k in NAMES})
    empty=rf.select_support_by_energy(c,grid,eps); assert empty.k_idx.numel()==0
    c.pp=torch.tensor([1.,2.,3.],dtype=rf.CMPLX_DTYPE)
    s=rf.select_support_by_energy(c,grid,eps)
    assert s.p[s.k_idx.long()].sum().item()>=1-eps-1e-6
    assert torch.equal(s.deltas,grid.deltas[s.k_idx.long()])
    assert s.k_idx.tolist()==sorted(s.k_idx.tolist())
    if eps==0: assert s.k_idx.numel()==3

@pytest.mark.parametrize('layout',['BNz33','B33Nz'])
@pytest.mark.parametrize('order',['b0_major_b1_minor','b1_major_b0_minor'])
def test_bank_layouts_and_validation(rf,bank,zgrid,layout,order):
    R=bank.Rxyz_bank
    if order=='b1_major_b0_minor': R=R.transpose(0,1)
    flat=R.reshape(4,len(zgrid),3,3)
    if layout=='B33Nz': flat=flat.permute(0,2,3,1)
    b=rf.RFKernel6BankSO3.from_Rxyz_flat(flat,bank.ax,zgrid,device='cpu',in_layout=layout,flat_order=order)
    torch.testing.assert_close(b.Rxyz_bank,bank.Rxyz_bank)
    with pytest.raises(ValueError,match='attach_kgrid'): b.ensure_H6_bins()
    im=rf.build_imap([0.],[1.],b.ax,device='cpu')
    with pytest.raises(ValueError,match='attach_kgrid'): b.eval_voxels_so3(im,torch.tensor([0]))
    with pytest.raises(AssertionError): rf.RFKernel6BankSO3.from_Rxyz_flat(flat[:1],bank.ax,zgrid,device='cpu')

@pytest.mark.parametrize('chunk',[1,2,4096])
def test_voxel_bank_exact_corners_against_independent_channel_dft(rf,bank,chunk):
    im=rf.build_imap([-10.,10.,np.nan],[.8,1.2,1.],bank.ax,device='cpu')
    bank.ensure_H6_bins();hb=bank.H6_bank
    bank.ensure_H6_bins();assert bank.H6_bank is hb  # cache contract only
    idx=torch.arange(bank.kgrid.kz.numel())
    vox,mapping=bank.eval_voxels_so3(im,idx,chunk_vox=chunk)
    assert mapping=={i:i for i in range(len(idx))}
    z=bank.z_m.cpu().numpy();k=bank.kgrid.kz.cpu().numpy()
    reference=channel_six(Rotation.from_rotvec([[0.,0.,0.],[.9,0.,0.]]).as_matrix())
    projection=dft_matrix(z,k)
    for name in NAMES:
        expected=np.broadcast_to(reference[name][:,None],(2,len(z)))@projection
        got=getattr(vox,name).detach().cpu().numpy()
        np.testing.assert_allclose(got[:2],expected,atol=3e-6,rtol=3e-6)
        np.testing.assert_array_equal(got[2],0)
        bins=getattr(hb,name).detach().cpu().numpy()
        np.testing.assert_allclose(bins[0,0],expected[0],atol=3e-6,rtol=3e-6)
        np.testing.assert_allclose(bins[1,1],expected[1],atol=3e-6,rtol=3e-6)
    nominal=bank.compute_nominal_PD_weighted(im,[1.,3.,100.])
    for name in NAMES:
        expected=np.broadcast_to(reference[name][:,None],(2,len(z)))@projection
        np.testing.assert_allclose(getattr(nominal,name).detach().cpu(),.25*expected[0]+.75*expected[1],atol=3e-6,rtol=3e-6)
    bank.attach_kgrid(bank.kgrid);assert bank.H6_bank is None and bank.nominal is None



def test_interpolated_midpoint_against_independent_rotation_and_dft(rf,bank):
    im=rf.build_imap([0.],[1.],bank.ax,device='cpu')
    vox,_=bank.eval_voxels_so3(im,torch.arange(len(bank.kgrid.kz)))
    z=bank.z_m.cpu().numpy();k=bank.kgrid.kz.cpu().numpy()
    expected=channel_six(Rotation.from_rotvec([.45,0,0]).as_matrix())
    for name in NAMES:
        taps=np.full(len(z),expected[name])@dft_matrix(z,k)
        np.testing.assert_allclose(getattr(vox,name)[0].detach().cpu(),taps,atol=3e-6,rtol=3e-6)







@pytest.mark.parametrize('n',[8,9])
def test_kgrid_has_complete_nonduplicated_fourier_orders(rf,n):
    z=(np.arange(n)-n//2)*.001
    grid=rf.build_kgrid_projection(z_m=torch.tensor(z,dtype=rf.DTYPE),device='cpu')
    # This is a full-grid API contract. An intentionally oversampled tap grid
    # needs a separately specified contract rather than this FFT-grid test.
    k=grid.kz.detach().cpu().numpy();orders=grid.deltas.detach().cpu().numpy()
    assert len(k)==n, 'Full DFT grid must contain exactly N unique Fourier modes'
    np.testing.assert_array_equal(orders,fourier_orders(n))
    np.testing.assert_allclose(k,fourier_k(z),rtol=3e-6,atol=3e-5)
    assert float(grid.dk_tap)==pytest.approx(1/(n*.001),rel=3e-6)

@pytest.mark.parametrize('n',[8,9])
def test_spatially_varying_rf_taps_and_cross_order_convolution(rf,n):
    z=(np.arange(n)-n//2)*.001;k=fourier_k(z);orders=fourier_orders(n)
    # Rotate around x with a spatially varying angle: nonzero cross-order taps.
    angle=.7+.25*np.cos(2*np.pi*k[-1]*z)
    R=Rotation.from_rotvec(np.column_stack([angle,np.zeros((n,2))])).as_matrix()
    ax=rf.make_axes([-10.,10.],[.8,1.2],device='cpu')
    flat=np.broadcast_to(R,(4,n,3,3)).copy()
    bank=rf.RFKernel6BankSO3.from_Rxyz_flat(torch.tensor(flat,dtype=rf.DTYPE),ax,
        torch.tensor(z,dtype=rf.DTYPE),device='cpu')
    grid=rf.KGrid(kz=torch.tensor(k,dtype=rf.DTYPE),deltas=torch.tensor(orders),
        dk_tap=1/(n*.001),M=n//2)
    bank.attach_kgrid(grid);bank.ensure_H6_bins()
    channels=channel_six(R);P=dft_matrix(z,k)
    reconstructed={}
    for name in NAMES:
        got=getattr(bank.H6_bank,name)[0,0].detach().cpu().numpy()
        expected=channels[name]@P
        np.testing.assert_allclose(got,expected,atol=4e-6,rtol=4e-6)
        np.testing.assert_allclose(synthesis(got,z,k),channels[name],atol=3e-5,rtol=3e-5)
        reconstructed[name]=synthesis(got,z,k)
    assert_unitary(matrix_from_six(reconstructed),atol=4e-5)
    # Discrete periodic multiplication in z must equal circular convolution in k.
    mz=.8+.2*np.cos(2*np.pi*k[1]*z)
    mz_k=mz@P;h=bank.H6_bank.zp[0,0].detach().cpu().numpy()
    convolved=np.zeros(n,dtype=complex)
    for out in range(n):
        for a in range(n):
            for b in range(n):
                if (orders[a]+orders[b]-orders[out])%n==0: convolved[out]+=h[a]*mz_k[b]
    np.testing.assert_allclose(convolved,(channels['zp']*mz)@P,atol=4e-6,rtol=4e-6)
