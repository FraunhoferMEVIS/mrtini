"""Reconstruct saved complex acquisitions without rerunning their physics."""
from dataclasses import replace
import numpy as np
from physics import Protocol,reconstruct

def process(signal,k,protocol,metadata,options):
    p=Protocol(**protocol);method=options.get('method','auto');n=int(options.get('matrix',p.reconstruction_resolution or p.resolution));fov=float(options.get('fov_mm',0) or metadata.get('reconstruction_fov_mm') or 0)
    if not fov and (p.native_edit or p.backend=='preview'):fov=p.fov_mm
    if not fov:raise ValueError('Imported trajectory: explicitly verify cycles/m and enter reconstruction FOV')
    if p.backend=='native' and not p.native_edit and options.get('confirm_units') is not True:raise ValueError('Confirm imported k-space coordinates are cycles/m in xy order')
    if method not in ('auto','nufft','nufft_dcf','direct','cg'):raise ValueError('Select an explicit adjoint method')
    if n not in (64,128,256,512) or not 1<=fov<=1000:raise ValueError('Invalid reconstruction geometry')
    p=replace(p,recon=method if method!='cg' else 'direct',reconstruction_resolution=n,recon_fov_mm=fov);p.check()
    signal=np.asarray(signal);k=np.asarray(k);sigma=float(options.get('noise_sigma',0));window=float(options.get('apodization',0));seed=int(options.get('seed',0))
    if not np.isfinite(sigma) or sigma<0 or not 0<=window<=1:raise ValueError('Invalid noise or apodization')
    s=signal.copy();rng=np.random.default_rng(seed)
    if sigma:s+=sigma*(rng.normal(size=s.shape)+1j*rng.normal(size=s.shape))
    if window:
        radius=np.linalg.norm(k[:,:2],axis=1);mx=max(float(radius.max()),1e-20);s=s*((1-window)+window*(.5+.5*np.cos(np.pi*radius/mx)))[:,None] if s.ndim==2 else s*((1-window)+window*(.5+.5*np.cos(np.pi*radius/mx)))
    if options.get('psf'):s=np.ones((len(k),1),complex)*(((1-window)+window*(.5+.5*np.cos(np.pi*np.linalg.norm(k[:,:2],axis=1)/max(float(np.linalg.norm(k[:,:2],axis=1).max()),1e-20))))[:,None] if window else 1)
    if method=='cg' and not options.get('psf'):
        from inverse import direct_cg
        r=direct_cg(s,k,n,fov/1000,int(options.get('iterations',12)),float(options.get('regularization',.001)))
    else:
        r=reconstruct(s,k,p)
    imgs=r['complex_image'];coil=options.get('coil','rss')
    if coil!='rss':
        i=int(coil)
        if not 0<=i<len(imgs):raise ValueError('Invalid coil index')
        r['magnitude']=np.abs(imgs[i]);r['phase']=np.angle(imgs[i])
    r['normalization']='Coil-independent regularized inverse; not SENSE. Relative data residuals: '+', '.join(f'{x:.3g}' for x in r['residuals']) if 'residuals' in r else 'Adjoint divided by output matrix area; repeated samples summed; DCF modifies weights and scale. Phase of selected coil (coil 1 for RSS).'
    r['fov_mm']=fov;r['matrix']=n
    return r
