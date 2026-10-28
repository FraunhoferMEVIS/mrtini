"""Independent references. This module must never import MRtini or its fixtures.

Contract: Cartesian column vectors, active R=exp(+[omega]x dt), and
m+=(Mx+iMy)/sqrt(2), m-=(Mx-iMy)/sqrt(2). RF coefficient names are source->target:
pp=C00, pz=C20, mp=C01, mz=C21, zp=C02, zz=C22.
Fourier analysis uses exp(-2*pi*i*k*z), synthesis exp(+2*pi*i*k*z).
The waveform solver's controls are sampled at interval starts (ZOH).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

GAMMA_HZ = 42.577478e6
GAMMA_RAD = 2*np.pi*GAMMA_HZ
CHANNEL_NAMES = ('pp','pz','mp','mz','zp','zz')
CHANNEL_ENTRIES = ((0,0),(2,0),(0,1),(2,1),(0,2),(2,2))
BASIS = np.array([[1,1j,0],[1,-1j,0],[0,0,np.sqrt(2)]],dtype=complex)/np.sqrt(2)

def channel_matrix(cartesian):
    return BASIS @ np.asarray(cartesian) @ BASIS.conj().T

def channel_six(cartesian):
    c=channel_matrix(cartesian)
    return {name:c[...,i,j] for name,(i,j) in zip(CHANNEL_NAMES,CHANNEL_ENTRIES)}

def dft_matrix(z,k,norm='K'):
    z=np.asarray(z,dtype=float);k=np.asarray(k,dtype=float)
    if norm in ('K','Lz'): scale=1/len(z)
    elif norm=='dz': scale=float(z[1]-z[0])
    else: raise ValueError(norm)
    return scale*np.exp(-2j*np.pi*np.outer(z,k))

def fourier_orders(n):
    # One representative of each discrete Fourier mode, including ONE Nyquist mode.
    return np.arange(-(n//2),(n-1)//2+1)

def fourier_k(z):
    z=np.asarray(z);return fourier_orders(len(z))/(len(z)*(z[1]-z[0]))

def synthesis(coefficients,z,k):
    return np.asarray(coefficients) @ np.exp(2j*np.pi*np.outer(k,z))

def assert_so3(R,atol=3e-6):
    R=np.asarray(R)
    assert np.isfinite(R).all(), 'Nonfinite rotation'
    np.testing.assert_allclose(np.swapaxes(R,-1,-2)@R,np.broadcast_to(np.eye(3),R.shape),atol=atol,rtol=0)
    np.testing.assert_allclose(np.linalg.det(R),np.ones(R.shape[:-2]),atol=atol,rtol=0)

def assert_unitary(C,atol=3e-6):
    C=np.asarray(C)
    assert np.isfinite(C).all(), 'Nonfinite channel matrix'
    np.testing.assert_allclose(np.swapaxes(C.conj(),-1,-2)@C,np.broadcast_to(np.eye(3),C.shape),atol=atol,rtol=0)

def z_rotation(phi):
    phi=np.asarray(phi);out=np.zeros(phi.shape+(3,3))
    out[...,0,0]=np.cos(phi);out[...,0,1]=-np.sin(phi)
    out[...,1,0]=np.sin(phi);out[...,1,1]=np.cos(phi);out[...,2,2]=1
    return out

def gradient_area_zoh(t,gz,center):
    t=np.asarray(t,dtype=float);g=np.asarray(gz,dtype=float)
    if len(t)<2 or not np.all(np.diff(t)>0): raise ValueError('Increasing interval boundaries required')
    dt=np.diff(t);pre_dt=np.clip(float(center)-t[:-1],0,dt)
    pre=np.dot(g[:-1],pre_dt);post=np.dot(g[:-1],dt-pre_dt)
    return float(pre),float(post)

def demod_reference(R,z,t,gz,gamma,center):
    """R has (...,Nz,3,3) layout. Remove active positive pre/post gradient rotations."""
    pre,post=gradient_area_zoh(t,gz,center)
    return z_rotation(-gamma*post*np.asarray(z))@R@z_rotation(-gamma*pre*np.asarray(z))

def solve_zoh_ivp(*,t,rf,gz,z,b0=0.,b1=1.,chemical_hz=0.,gamma=GAMMA_RAD):
    """Independent adaptive ODE solve of each constant-control interval.

    rf is a laboratory transverse complex field envelope in tesla, with any
    RF carrier/FM already encoded in its complex phase. No separate detuning
    is added for that carrier; chemical shift and B0 enter the z field.
    """
    R=np.eye(3);t=np.asarray(t,dtype=float);rf=np.asarray(rf);gz=np.asarray(gz)
    for i,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
        field=np.array([rf[i].real*b1,rf[i].imag*b1,gz[i]*z+b0+2*np.pi*chemical_hz/gamma])
        omega=gamma*field
        def rhs(_t,y):
            # Cross each COLUMN of the transfer matrix with the angular velocity.
            return np.cross(omega,y.reshape(3,3).T).T.reshape(-1)
        sol=solve_ivp(rhs,(float(ta),float(tb)),R.reshape(-1),method='DOP853',rtol=2e-11,atol=2e-13)
        assert sol.success,sol.message
        R=sol.y[:,-1].reshape(3,3)
    return R


def matrix_from_six(c):
    """Complete the real-Cartesian operator's channel symmetry, in test code."""
    shape=np.asarray(c['pp']).shape
    out=np.empty(shape+(3,3),dtype=complex)
    out[...,0,0]=c['pp'];out[...,0,1]=c['mp'];out[...,0,2]=c['zp']
    out[...,1,0]=np.conj(c['mp']);out[...,1,1]=np.conj(c['pp']);out[...,1,2]=np.conj(c['zp'])
    out[...,2,0]=c['pz'];out[...,2,1]=c['mz'];out[...,2,2]=c['zz']
    return out
