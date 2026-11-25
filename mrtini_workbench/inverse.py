"""Small exact nonuniform Fourier inverse, coil-independent Tikhonov CG.
This is an educational/validation solver, not SENSE or field-aware reconstruction.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator,cg

def direct_cg(signal,k,n,fov,iterations=12,regularization=1e-3):
    s=np.asarray(signal);s=s[:,None] if s.ndim==1 else s;k=np.asarray(k,float)[:,:2]
    if not 1<=iterations<=100 or not np.isfinite(regularization) or regularization<=0:raise ValueError('CG requires 1–100 iterations and positive regularization')
    if len(k)*n*n>8_000_000:raise ValueError('Exact CG reference is limited to 8 million sample-pixel products. Reduce the matrix or use an adjoint NUFFT.')
    pos=(np.arange(n)-n//2)*fov/n;Y,X=np.meshgrid(pos,pos,indexing='ij');xy=np.stack([X.ravel(),Y.ravel()]);A=np.exp(2j*np.pi*k@xy);AH=A.conj().T;scale=len(k)
    H=LinearOperator((n*n,n*n),matvec=lambda x:AH@(A@x)/scale+regularization*x,dtype=np.complex128);images=[];residuals=[]
    for v in s.T:
        b=AH@v/scale
        x,info=cg(H,b,maxiter=iterations,rtol=1e-6,atol=0)
        if info<0 or not np.isfinite(x).all():raise ValueError('CG reconstruction failed')
        images.append(x.reshape(n,n));residuals.append(float(np.linalg.norm(A@x-v)/max(np.linalg.norm(v),1e-30)))
    images=np.asarray(images);return dict(complex_image=images,magnitude=np.sqrt((abs(images)**2).sum(0)),phase=np.angle(images[0]),method=f'Exact nonuniform Fourier Tikhonov CG (≤{iterations} iterations; λ={regularization:g})',residuals=residuals)
