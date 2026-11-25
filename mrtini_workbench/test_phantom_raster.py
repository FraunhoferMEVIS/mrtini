import numpy as np
import pytest
from types import SimpleNamespace as NS
from research import native_maps, _regular_axis

@pytest.mark.parametrize('n',[64,128,256])
@pytest.mark.parametrize('endpoints',[True,False])
def test_flattened_brainweb_float32(n,endpoints):
    # Physical BrainWeb FOV; no coordinate rounding in the simulation.
    x=np.linspace(-.181/2,.181/2,n,endpoint=endpoints,dtype=np.float32)
    y=np.linspace(-.217/2,.217/2,n,endpoint=endpoints,dtype=np.float32)
    ix,iy=np.meshgrid(np.arange(n),np.arange(n)); ix=ix.ravel();iy=iy.ravel()
    keep=((ix-n/2)**2+(iy-n/2)**2<(n*.45)**2)&(ix!=n//2+1)
    # Retain extrema so full original axis length can be verified.
    keep[[0,n*n-1]]=True
    ix=ix[keep];iy=iy[keep];N=len(ix)
    pos=np.column_stack((x[ix],y[iy],np.full(N,.015,np.float32)))
    # Same spatial position occupied by both species, with shuffled packing.
    ids=np.arange(0,N,3); pos=np.concatenate((pos,pos[ids]));sp=np.r_[np.zeros(N,int),np.ones(len(ids),int)]
    pd=np.r_[np.full(N,.8),np.full(len(ids),.2)].astype(np.float32);t1=np.where(sp==0,1.56,.37)
    perm=np.random.default_rng(8).permutation(len(pd));d=NS(voxel_pos=pos[perm],PD=pd[perm,None],T1=t1[perm,None],T2=np.ones(len(pd)),T2dash=np.ones(len(pd)),D=np.ones(len(pd)),B0=np.where(sp[perm],-440.,0.),B1=np.ones((1,len(pd)),complex))
    maps=native_maps(d,sp[perm]); assert maps['PD'].shape==(n,n)
    np.testing.assert_allclose(maps['x_m'],x,atol=2e-7,rtol=0);np.testing.assert_allclose(maps['y_m'],y,atol=2e-7,rtol=0)
    expected=np.zeros((n,n));expected[iy,ix]=.8;expected[iy[ids],ix[ids]]=1
    np.testing.assert_allclose(maps['PD'],expected,atol=5e-8)
    np.testing.assert_allclose(maps['fat_fraction'][iy[ids],ix[ids]],.2,atol=1e-7)
    for label,species in [('water',0),('fat',1)]:
        arr=maps[label+'_index']; mask=arr>=0; saved=arr[mask];assert np.all(sp[perm][saved]==species)
        yy,xx=np.where(mask);np.testing.assert_allclose(d.voxel_pos[saved,:2],np.column_stack((x[xx],y[yy])),atol=2e-7)
    assert np.isnan(maps['water_T1'][n//2,n//2+1]);assert maps['water_index'][n//2,n//2+1]==-1

def test_coordinate_noise_and_holes():
    x=np.arange(128)*.00142513-.09;keep=np.r_[np.arange(30),np.arange(40,128)];v=np.repeat(x[keep],3)+np.tile([-4e-8,0,4e-8],len(keep));axis,ix=_regular_axis(v)
    assert len(axis)==128;np.testing.assert_array_equal(ix,np.repeat(keep,3));np.testing.assert_allclose(axis,x,atol=2e-7,rtol=0)

def test_irregular_rejected():
    with pytest.raises(ValueError,match='regular'): _regular_axis([0,.001,.002,.0034,.004])

def test_single_coordinate():
    axis,ix=_regular_axis([.002,.00200002]);assert len(axis)==1;assert np.all(ix==0)
