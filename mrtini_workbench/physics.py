"""Acquisition definitions, explicit idealized preview, and complex adjoint reconstruction."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import gaussian_filter

@dataclass(frozen=True)
class Protocol:
    sequence: str = 'flash'
    resolution: int = 128
    phantom_resolution: int = 0  # 0 follows acquisition matrix
    reconstruction_resolution: int = 0
    fov_mm: float = 256
    slice_mm: float = 5
    tr_ms: float = 35
    te_ms: float = 8
    flip_deg: float = 20
    refocus_deg: float = 180
    etl: int = 8
    esp_ms: float = 12
    dwell_us: float = 12
    rf_ms: float = 1.5
    tbw: float = 4
    b0_hz: float = 0
    b0_offset_hz: float = 0
    b1_scale: float = 1
    workspace_mb: int = 128
    kernel_mode: str = 'dynamic'
    cuda_graph: bool = False
    backend: str = 'preview'
    recon: str = 'auto'
    recon_fov_mm: float = 0.0  # zero = generated encoding FOV; imported auto uses legacy
    sequence_file: str = ''
    phantom_file: str = ''
    slice_index: int = 15
    native_edit: bool = False
    global_rf: bool = False
    max_states: int = 800
    min_state_mag: float = 1e-4
    custom_recipe: str = ""
    diagnostics: bool = True  # preserve API compatibility; UI defaults to fast
    def check(self):
        if self.sequence not in ('flash','rare','epi','custom') or self.resolution not in (64,128,256): raise ValueError('Unsupported sequence or matrix')
        if self.phantom_resolution not in (0,64,128,256,512) or self.reconstruction_resolution not in (0,64,128,256,512): raise ValueError('Unsupported independent matrix')
        if self.backend=='preview' and self.phantom_resolution not in (0,self.resolution): raise ValueError('Independent phantom resolution requires native mode')
        for key,lo,hi in [('fov_mm',100,400),('slice_mm',1,15),('tr_ms',5,10000),('te_ms',2,500),('flip_deg',1,180),('refocus_deg',30,180),('esp_ms',.5,100),('dwell_us',4,40),('rf_ms',.5,5),('tbw',2,8),('b0_hz',-200,200),('b0_offset_hz',-2000,2000),('b1_scale',.1,2),('workspace_mb',1,512)]:
            x=getattr(self,key)
            if not np.isfinite(x) or not lo<=x<=hi: raise ValueError(f'{key} must be between {lo} and {hi}')
        if self.etl not in (2,4,8,16,32): raise ValueError('Unsupported echo train length')
        if self.kernel_mode not in ('dynamic','specialized') or self.backend not in ('preview','native') or self.recon not in ('auto','nufft','nufft_dcf','direct','native'): raise ValueError('Invalid execution option')
        if not np.isfinite(self.recon_fov_mm) or (self.recon_fov_mm!=0 and not 1<=self.recon_fov_mm<=1000): raise ValueError('Reconstruction FOV must be 0 (default) or 1–1000 mm')
        if self.recon=='native' and self.backend!='native': raise ValueError('Legacy native reconstruction requires native mode')
        if self.backend=='native' and not self.native_edit and self.recon not in ('auto','native') and not self.recon_fov_mm: raise ValueError('Set an explicit reconstruction FOV for imported trajectories')
        if type(self.diagnostics) is not bool: raise ValueError('diagnostics must be a boolean')
        if type(self.global_rf) is not bool: raise ValueError('global_rf must be a boolean')
        if type(self.max_states) is not int or not 1<=self.max_states<=20000: raise ValueError('State budget must be an integer between 1 and 20000')
        if not 0<=self.slice_index<=30 or not 0<self.min_state_mag<=1: raise ValueError('Invalid slice or graph settings')
        if self.sequence=='custom':
            if self.backend!='native' or not self.native_edit or not self.custom_recipe: raise ValueError('Custom designs require a native generated sequence and a recipe')
            if len(self.custom_recipe)>100000: raise ValueError('Recipe too large')
            return self
        if self.backend=='native' and not self.native_edit: return self
        if self.te_ms>=self.tr_ms: raise ValueError('TE must be shorter than TR')
        if self.sequence=='rare' and self.te_ms!=self.esp_ms: raise ValueError('RARE uses first echo TE = echo spacing in this linear-order preset')
        if self.sequence=='rare' and self.etl*self.esp_ms>=self.tr_ms: raise ValueError('TR must exceed the echo train duration')
        if self.sequence=='epi' and self.te_ms+(min(self.etl,self.resolution)//2)*self.esp_ms>=self.tr_ms: raise ValueError('TR must extend beyond the last EPI readout')
        if self.sequence=='epi' and self.te_ms/2-(min(self.etl,self.resolution)//2)*self.esp_ms<=0: raise ValueError('TE too short: first EPI line must follow the 180° pulse; increase TE or reduce echo spacing/matrix')
        return self

def phantom(p):
    n=p.resolution;x=(np.arange(n)-n//2)/ (n/2); yy,xx=np.meshgrid(x,x,indexing='ij')
    def ell(cx,cy,a,b,ang=0):
        t=np.deg2rad(ang); X=(xx-cx)*np.cos(t)+(yy-cy)*np.sin(t);Y=-(xx-cx)*np.sin(t)+(yy-cy)*np.cos(t)
        return gaussian_filter(((X/a)**2+(Y/b)**2<1).astype(float),.65*n/128)
    brain=ell(0,0,.78,.90); wm=ell(0,.015,.62,.76); csf=np.maximum(ell(-.13,-.035,.105,.30,-16),ell(.13,-.035,.105,.30,16)); csf=np.minimum(csf,wm)
    # Volume-fraction mixed synthetic tissue maps, not BrainWeb or patient data.
    skull=np.clip(ell(0,0,.84,.95)-brain,0,1); white=np.clip(wm-csf,0,1); gray=np.clip(brain-wm,0,1)
    fractions=np.stack([gray,white,csf,skull]); pdv=np.array([.86,.72,1,.12])[:,None,None]; weighted=fractions*pdv; pd=weighted.sum(0); den=np.maximum(pd,1e-10)
    t1=(weighted*np.array([1.35,.85,3.6,.4])[:,None,None]).sum(0)/den;t2=(weighted*np.array([.085,.065,.35,.025])[:,None,None]).sum(0)/den
    t1=np.where(pd>1e-7,t1,1); t2=np.where(pd>1e-7,t2,.08); t2d=np.full_like(t2,.12); b0=p.b0_hz*(.65*xx+.35*yy+.25*np.sin(4*xx)*np.cos(3*yy));
    return dict(PD=pd,T1=t1,T2=t2,T2dash=t2d,B0=b0)

def line_plan(p):
    n=p.resolution; rows=[]
    for y in range(n):
        if p.sequence=='flash': shot=y;echo=1;t=p.te_ms; polarity=1
        elif p.sequence=='rare': shot=y//p.etl;echo=y%p.etl+1;t=echo*p.esp_ms;polarity=1
        else:
            # Interleaved multishot EPI; centre readout in each shot occurs at TE.
            nshots=n//p.etl;shot=y%nshots;echo=y//nshots+1;t=p.te_ms+(echo-1-p.etl//2)*p.esp_ms;polarity=(-1)**(echo-1)
        rows.append(dict(ky=y-n//2,shot=shot,echo=echo,time_ms=t,polarity=polarity))
    return sorted(rows,key=lambda r:(r['shot'],r['echo']))

def preview(p, maps, progress):
    """Idealized line-centre approximation, no slice profile, diffusion or stimulated pathways.
    Forward positive phase. RF spoiling/steady state assumed for FLASH, primary echo only for RARE.
    EPI includes spatially varying B0 phase away from spin echo; no intra-readout evolution.
    """
    n=p.resolution; pd,t1,t2,t2d,b0=(maps[k] for k in ('PD','T1','T2','T2dash','B0')); tr=p.tr_ms/1000; alpha=np.deg2rad(p.flip_deg); beta=np.deg2rad(p.refocus_deg)
    E=np.exp(-tr/t1); m0=pd*(1-E)/(1-E*np.cos(alpha))*np.sin(alpha) if p.sequence=='flash' else pd*(1-E)*np.sin(alpha)
    k=[];s=[]; lines=line_plan(p); fov=p.fov_mm/1000
    for j,r in enumerate(lines):
        t=r['time_ms']/1000
        if p.sequence=='flash': m=m0*np.exp(-t/t2-t/t2d+2j*np.pi*b0*t)
        elif p.sequence=='rare': m=m0*np.exp(-t/t2)*(np.sin(beta/2)**2)**r['echo']
        else:
            dt=t-p.te_ms/1000; m=m0*np.sin(beta/2)**2*np.exp(-t/t2-np.abs(dt)/t2d+2j*np.pi*b0*dt)
        ky=r['ky']; kx=np.arange(n)-n//2
        if r['polarity']<0:kx=kx[::-1]
        spec=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(m)))*n*n
        s.append(spec[ky+n//2,kx+n//2]); k.append(np.stack([kx/fov,np.full(n,ky/fov)],1))
        if j%16==0 or j==n-1:progress('execute',(j+1)/n,f'Preview line {j+1}/{n}')
    return np.concatenate(s)[:,None],np.concatenate(k)

def reconstruct(signal,k,p):
    """Positive-phase simulator convention, k in cycles/m; array [ky,kx]."""
    s=np.asarray(signal); k=np.asarray(k,dtype=float)[:,:2]; n=p.reconstruction_resolution or p.resolution; fov=(p.recon_fov_mm or p.fov_mm)/1000
    if s.ndim==1:s=s[:,None]
    if s.ndim!=2 or len(s)!=len(k) or not len(k):raise ValueError('ADC signal/trajectory shape mismatch')
    if not np.iscomplexobj(s) or not np.isfinite(s).all() or not np.isfinite(k).all():raise ValueError('Expected finite complex ADC and finite trajectory')
    scaled=k*fov; ij=np.rint(scaled).astype(int)+n//2
    ongrid=np.max(np.abs(scaled-np.rint(scaled)))<2e-3 and (ij>=0).all() and (ij<n).all()
    density_weights=None;grid=np.zeros((s.shape[1],n,n),complex); counts=np.zeros((n,n),int)
    if ongrid:
        np.add.at(counts,(ij[:,1],ij[:,0]),1)
        for c in range(s.shape[1]):np.add.at(grid[c],(ij[:,1],ij[:,0]),s[:,c])
    if p.recon=='auto' and ongrid:
        # True adjoint: sum repeated samples; no implicit density compensation.
        imgs=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(grid,axes=(-2,-1)),axes=(-2,-1)),axes=(-2,-1))/ (n*n);method='Cartesian FFT adjoint'
    elif p.recon in ('nufft','nufft_dcf') or (p.recon=='auto' and not ongrid):
        try:
            import torch, torchkbnufft as tkbn
        except ImportError as e:raise RuntimeError('Non-Cartesian/NUFFT reconstruction requires torchkbnufft; install it or select Direct adjoint.') from e
        dev='cuda' if torch.cuda.is_available() else 'cpu'; op=tkbn.KbNufftAdjoint(im_size=(n,n)).to(dev)
        # torchkbnufft adjoint is exp(+i k r); reverse sign for MRTini forward +i, reorder y,x.
        om=torch.as_tensor(-2*np.pi*k[:,[1,0]].T*fov/n,dtype=torch.float32,device=dev)
        inp=torch.as_tensor(s.T[None],dtype=torch.complex64,device=dev)
        with torch.no_grad():
            if p.recon in ('auto','nufft_dcf'):
                weights=tkbn.calc_density_compensation_function(om,im_size=(n,n))
                if not torch.isfinite(weights).all() or (weights.real<0).any() or weights.abs().max()==0: raise ValueError('Density compensation produced invalid weights')
                density_weights=weights.real.detach().cpu().numpy().reshape(-1)
                inp=inp*weights
            imgs=op(inp,om)[0].cpu().numpy()/(n*n)
        method='Kaiser–Bessel NUFFT adjoint ('+('iterative DCF' if p.recon in ('auto','nufft_dcf') else 'no DCF')+')'
    elif p.recon=='direct':
        if len(k)*n*n*s.shape[1]>200_000_000: raise ValueError('Direct Fourier reference exceeds 200 million sample-pixel products; reduce matrix or select NUFFT')
        pos=(np.arange(n)-n//2)*fov/n; Y,X=np.meshgrid(pos,pos,indexing='ij'); xy=np.stack([X.ravel(),Y.ravel()]); accum=np.zeros((s.shape[1],n*n),complex)
        for i in range(0,len(k),128):accum+=s[i:i+128].T@np.exp(-2j*np.pi*k[i:i+128]@xy)
        imgs=accum.reshape(-1,n,n)/(n*n);method='Direct Fourier adjoint'
    else: raise ValueError('Unsupported explicit reconstruction method')
    # Display k-space raster only; off-grid trajectory is rasterized for visualization, not reconstruction.
    if not ongrid:
        valid=(ij>=0).all(1)&(ij<n).all(1)
        for c in range(s.shape[1]):np.add.at(grid[c],(ij[valid,1],ij[valid,0]),s[valid,c])
    return dict(magnitude=np.sqrt((np.abs(imgs)**2).sum(0)),phase=np.angle(imgs[0]),kspace=np.log1p(np.sqrt((np.abs(grid)**2).sum(0))),complex_image=imgs,method=method,on_grid=bool(ongrid),density_weights=density_weights)
