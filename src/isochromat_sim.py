"""NOMAD image-level isochromat runner; PyTorch and the original project required.

Put beside the original script (imports: sequence and phantom.VoxelGridPhantom).
Imported spans: rf_amp in Hz, gz in Hz/m, rf_phase in radians, t in seconds.
Convention: dM/dt = +gamma B x M; Mxy=Mx+iMy; receiver exp(+i adc_phase).
RF uses direct held-raster indexing and inverse-gradient centre referencing.
As in the supplied comparator, RF relaxation and diffusion are NOT simulated.
Species shifts are NOT added here; data.B0 is the effective static offset in Hz.
The inherited +rf_freq convention and cumulative B0 event timing are preserved.
This is a matched-model comparator, not the full waveform/relaxation reference.

Examples:
  python isochromat_sim_nomad.py --self-test
  python isochromat_sim_nomad.py --nx 128 --ny 128 --nz 30 --slice 14 \
      --seqs FLASH4mes --spins 32000 --voxel-mm 1 1 5 --no-b0 --no-b1
Repeat the same command to resume. --no-resume starts again; --help lists paths.
--voxel-mm specifies voxel dimensions: total FOV is nx*dx by ny*dy by dz.
Voxel batches traverse the complete sequence; their complex signals are summed.
Checkpoints commit at RF-repetition boundaries, not halfway through an RF solve.
"""
from __future__ import annotations
import argparse, hashlib, json, os, signal as os_signal, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
_STOP_REQUESTED = False


def atomic_torch_save(obj, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp=tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(obj,stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npz(path, **values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,**values); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def _hash_value(h, value):
    if isinstance(value,torch.Tensor):
        x=value.detach().cpu().contiguous().numpy()
        h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    elif isinstance(value,dict):
        for k in sorted(value): h.update(str(k).encode()); _hash_value(h,value[k])
    elif isinstance(value,(list,tuple)):
        for x in value: _hash_value(h,x)
    elif isinstance(value,np.ndarray):
        h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    else: h.update(repr(value).encode())


def rf_spans(seq):
    reps=list(seq); hybrid=seq.mrtini_hybrid
    mapping=getattr(seq,'mrtini_spans_by_rep',None)
    if mapping is None:
        ids=[i for i,s in enumerate(hybrid) if s['kind']=='bloch']
        if len(ids)!=len(reps):
            raise ValueError('Need exactly one RF span per repetition, or mrtini_spans_by_rep.')
        return [hybrid[i] for i in ids]
    if len(mapping)!=len(reps): raise ValueError('RF/repetition mapping length differs.')
    out=[]
    for r,indices in enumerate(mapping):
        ids=[int(i) for i in indices if hybrid[int(i)]['kind']=='bloch']
        if len(ids)!=1: raise ValueError(f'Repetition {r}: expected one RF span, found {len(ids)}.')
        out.append(hybrid[ids[0]])
    return out


def simulation_fingerprint(seq,data,spans,options):
    h=hashlib.sha256()
    _hash_value(h,{'source':file_hash(__file__),'options':options})
    for name in ('PD','B0','B1','T1','T2','T2dash','voxel_pos','coil_sens','nyquist'):
        h.update(name.encode()); _hash_value(h,getattr(data,name))
    for rep,span in zip(seq,spans):
        for name in ('event_time','gradm','adc_usage','adc_phase'):
            h.update(name.encode()); _hash_value(h,getattr(rep,name))
        _hash_value(h,str(rep.pulse.usage))
        _hash_value(h,{k:span.get(k) for k in ('t','rf_amp','gz','rf_phase','rf_freq','rf_center_time')})
    return h.hexdigest()


def _rotate_z(spins,angle):
    # Same positive-angle matrix as the source, without a V*S*3*3 allocation.
    c,s=torch.cos(angle),torch.sin(angle); out=spins.clone()
    out[...,0]=c*spins[...,0]-s*spins[...,1]
    out[...,1]=s*spins[...,0]+c*spins[...,1]
    return out


def relax(spins,T1,T2,dt):
    out=torch.empty_like(spins); r1=torch.exp(-dt/T1).reshape(-1,1)
    out[...,2]=spins[...,2]*r1+(1-r1)
    out[...,:2]=spins[...,:2]*torch.exp(-dt/T2).reshape(-1,1,1)
    return out


def dephase(spins,omega,T2dash,dt):
    return _rotate_z(spins,dt/T2dash.reshape(-1,1)*omega.reshape(1,-1))


def intravoxel_precess(spins,gradm,spin_pos):
    return _rotate_z(spins,spin_pos@gradm)


def B0_precess(spins,B0,dt):
    return _rotate_z(spins,(2*torch.pi*B0*dt).reshape(-1,1))


def grad_precess(spins,gradm,voxel_pos):
    return _rotate_z(spins,(2*torch.pi*(voxel_pos@gradm)).reshape(-1,1))


def measure(spins,coil_sensitivity):
    return (spins[...,0].sum(1)+1j*spins[...,1].sum(1))@coil_sensitivity


@torch.no_grad()
def demod_Rxyz_to_Rmix_center(*, Rz_SB33Nz, z_m, t, gz, gamma_rad, rf_center_time=None):
    dtype, device = Rz_SB33Nz.dtype, Rz_SB33Nz.device
    t=t.to(device=device,dtype=dtype); gz=gz.to(device=device,dtype=dtype); z=z_m.to(device=device,dtype=dtype)

    gamma_hz = gamma_rad / (2*torch.pi)
    t_ref = torch.as_tensor(
    .5*(t[0]+t[-1]) if rf_center_time is None else rf_center_time,
    device=device, dtype=dtype)
    if t.ndim != 1 or t.numel() < 2 or gz.shape != t.shape:
        raise ValueError('Expected matching 1D RF boundary times and held gradient samples.')
    dt = t[1:] - t[:-1]
    if not bool((dt > 0).all()) or not bool((t_ref >= t[0]) & (t_ref <= t[-1])):
        raise ValueError('RF times must increase and RF centre must lie inside the span.')
    pre_dt = torch.minimum((t_ref-t[:-1]).clamp_min(0), dt)
    area_pre = (gz[:-1]*pre_dt).sum()
    area_post = (gz[:-1]*dt).sum() - area_pre
    k_pre  = gamma_hz * area_pre
    k_post = gamma_hz * area_post

    phi_pre  = 2*torch.pi * k_pre  * z
    phi_post = 2*torch.pi * k_post * z

    def Qz(phi):
        c=torch.cos(phi); s=torch.sin(phi)
        
        Q=torch.zeros((*phi.shape,3,3), device=device, dtype=dtype)
        Q[...,0,0]=c;  Q[...,0,1]=-s
        Q[...,1,0]=s;  Q[...,1,1]= c
        Q[...,2,2]=1
        return Q

    # Inverse of positive B x M gradient precession: Qz(-phi).
    Qpre_inv  = Qz(-phi_pre)
    Qpost_inv = Qz(-phi_post)

    # Here the leading axes are voxels and spins within the current chunk.
    return Qpost_inv @ Rz_SB33Nz @ Qpre_inv

@torch.no_grad()
def apply_bloch_pulse_rep_chunk(
    spins, rw, data, rep,
    spin_offset_m,
    *,
    gamma_hz=42.577478e6,
    gz_units="T_per_m",
    include_relaxation=False,
    spin_chunk=4000,   # try 512, 1024, 2048
):
    if rw is None:
        raise ValueError('A waveform span is required for every RF repetition.')

    if include_relaxation:
        raise NotImplementedError('This matched-model RF solver omits relaxation during RF.')
    dev, dtype = spins.device, spins.dtype
    gamma_rad = 2 * torch.pi * gamma_hz
    GAMMA_HZ = gamma_hz

    if spin_offset_m is None:
        spin_offset_m = torch.zeros(spins.shape[1], 3, device=dev, dtype=dtype)
    else:
        spin_offset_m = spin_offset_m.to(device=dev, dtype=dtype)

    t = torch.as_tensor(rw["t"], device=dev, dtype=dtype)

    rf = torch.as_tensor(rw["rf_amp"], device=dev, dtype=torch.cfloat) / GAMMA_HZ
    gz = torch.as_tensor(rw["gz"], device=dev, dtype=dtype)
    if gz_units == 'T_per_m':
        gz = gz / GAMMA_HZ  # Imported span stores Hz/m; convert once to T/m.
    elif gz_units != 'cycles_per_m_per_s':
        raise ValueError('Unknown gz_units')

    rf_phase = rw.get("rf_phase", None)
    if rf_phase is None:
        rf_phase = torch.zeros_like(t)
    else:
        rf_phase = torch.as_tensor(rf_phase, device=dev, dtype=dtype)

    rf_complex = rf.to(torch.cfloat) * torch.exp(1j * rf_phase)

    rf_freq = rw.get("rf_freq", None)
    if rf_freq is None:
        rf_freq = torch.zeros_like(t)
    else:
        rf_freq = torch.as_tensor(rf_freq, device=dev, dtype=dtype)

    V, S = spins.shape[:2]

    # B1 map: (V, 1)
    B1map = data.B1
    if B1map.ndim > 1:
        B1map = B1map.sum(0)
    B1map = B1map.to(device=dev)
    if not torch.is_complex(B1map):
        B1map = B1map.to(dtype).to(torch.cfloat)
    B1map = B1map.reshape(V, 1)

    # B0 map: (V, 1)
    B0_hz = data.B0.to(device=dev, dtype=dtype).reshape(V, 1)

    # voxel z position: (V, 1)
    voxel_z = data.voxel_pos[:, 2].to(device=dev, dtype=dtype).reshape(V, 1)

    eye = torch.eye(3, device=dev, dtype=dtype).view(1, 1, 3, 3)

    for s0 in range(0, S, spin_chunk):
        s1 = min(s0 + spin_chunk, S)
        Sc = s1 - s0

        # chunked absolute z-position: (V, Sc)
        z_abs = voxel_z + spin_offset_m[s0:s1, 2].reshape(1, Sc)

        # chunked accumulated rotation: (V, Sc, 3, 3)
        Rtot = eye.expand(V, Sc, 3, 3).clone()

        for n in range(t.numel() - 1):
            dt = float((t[n + 1] - t[n]).item())
            if dt <= 0:
                continue

            B1eff = B1map * rf_complex[n]

            # Keep wx/wy as (V,1). Let broadcasting handle spin dimension.
            omega_x = gamma_rad * B1eff.real
            omega_y = gamma_rad * B1eff.imag

            if gz_units == "cycles_per_m_per_s":
                omega_z = 2 * torch.pi * (B0_hz + rf_freq[n] + gz[n] * z_abs)

            elif gz_units == "T_per_m":
                omega_z = 2 * torch.pi * (B0_hz + rf_freq[n]) + gamma_rad * gz[n] * z_abs

            else:
                raise ValueError(f"unknown gz_units={gz_units}")

            Rtot = left_compose_so3_components(
                Rtot, omega_x, omega_y, omega_z, dt
            )

        R = demod_Rxyz_to_Rmix_center(
            Rz_SB33Nz=Rtot,
            z_m=z_abs,
            t=t,
            gz=gz if gz_units == 'T_per_m' else gz / GAMMA_HZ,
            gamma_rad=gamma_rad,
            rf_center_time=rw.get('rf_center_time'),
        )

        spins[:, s0:s1, :] = torch.einsum(
            "vsij,vsj->vsi",
            R,
            spins[:, s0:s1, :]
        )

        del z_abs, Rtot, R

    return spins

@torch.no_grad()
def left_compose_so3_components(R, wx, wy, wz, dt, eps=1e-12):
    """
    Computes R_new = exp([w]x dt) @ R
    without forming the step rotation matrix.

    R:  (..., 3, 3)
    wx, wy, wz: broadcastable to R.shape[:-2]
    """

    w2 = wx*wx + wy*wy + wz*wz
    w = torch.sqrt(w2)
    theta = w * dt

    w_safe = w.clamp_min(eps)
    a = torch.sin(theta) / w_safe
    b = (1.0 - torch.cos(theta)) / (w_safe*w_safe)

    small = theta.abs() < 1e-4
    a = torch.where(small, dt - w2*(dt**3)/6.0, a)
    b = torch.where(small, 0.5*(dt**2) - w2*(dt**4)/24.0, b)

    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    c00 = wy*r20 - wz*r10
    c01 = wy*r21 - wz*r11
    c02 = wy*r22 - wz*r12

    c10 = wz*r00 - wx*r20
    c11 = wz*r01 - wx*r21
    c12 = wz*r02 - wx*r22

    c20 = wx*r10 - wy*r00
    c21 = wx*r11 - wy*r01
    c22 = wx*r12 - wy*r02

    d0 = wx*r00 + wy*r10 + wz*r20
    d1 = wx*r01 + wy*r11 + wz*r21
    d2 = wx*r02 + wy*r12 + wz*r22

    Rout = torch.empty_like(R)

    Rout[..., 0, 0] = r00 + a*c00 + b*(wx*d0 - w2*r00)
    Rout[..., 0, 1] = r01 + a*c01 + b*(wx*d1 - w2*r01)
    Rout[..., 0, 2] = r02 + a*c02 + b*(wx*d2 - w2*r02)

    Rout[..., 1, 0] = r10 + a*c10 + b*(wy*d0 - w2*r10)
    Rout[..., 1, 1] = r11 + a*c11 + b*(wy*d1 - w2*r11)
    Rout[..., 1, 2] = r12 + a*c12 + b*(wy*d2 - w2*r12)

    Rout[..., 2, 0] = r20 + a*c20 + b*(wz*d0 - w2*r20)
    Rout[..., 2, 1] = r21 + a*c21 + b*(wz*d1 - w2*r21)
    Rout[..., 2, 2] = r22 + a*c22 + b*(wz*d2 - w2*r22)

    return Rout

def _voxel_subset(data,lo,hi):
    d=SimpleNamespace(device=data.PD.device,nyquist=data.nyquist)
    for name in ('PD','B0','T1','T2','T2dash','voxel_pos'):
        setattr(d,name,getattr(data,name)[lo:hi])
    d.B1=data.B1[...,lo:hi]; d.coil_sens=data.coil_sens[:,lo:hi]
    return d


def requested_profile_events(rep,repetition,spec):
    if spec is None: return torch.nonzero(rep.adc_usage>0).flatten().tolist()
    if callable(spec): return list(spec(rep,repetition))
    if isinstance(spec,dict): return list(spec.get(repetition,()))
    return list(spec)


@torch.no_grad()
def isochromat_sim(seq,data,spin_count,perfect_spoiling=False,print_progress=True,
                   spin_dist='rand',r2_seed=None,profile_callback=None,profile_events=None,
                   checkpoint_path=None,resume=True,checkpoint_every=1,seed=2026,
                   spin_chunk=4000,voxel_batch=256,run_metadata=None):
    """Return complex [ADC samples,coils]; resume includes the same spin realization.

    Profile callbacks receive only the current voxel batch, identified by
    voxel_indices. Concatenate batches by these indices if profiles are needed.
    None selects every acquired event; keep callbacks disabled for full images.
    """
    if min(spin_count,spin_chunk,voxel_batch,checkpoint_every)<1:
        raise ValueError('Counts and checkpoint interval must be positive.')
    if spin_dist not in ('rand','r2','z'): raise ValueError('Unknown spin_dist.')
    if spin_dist!='r2' and r2_seed is not None: raise ValueError('r2_seed requires spin_dist=r2.')
    reps=list(seq); spans=rf_spans(seq)
    if not reps: raise ValueError('Empty sequence.')
    dev=data.PD.device; dtype=torch.float32
    V=int(data.PD.numel()); C=int(data.coil_sens.shape[0])
    if not V: raise ValueError('Phantom has no active voxels.')
    if data.PD.ndim!=1 or data.voxel_pos.shape!=(V,3) or data.coil_sens.shape[1]!=V:
        raise ValueError('Expected built data: PD[V], voxel_pos[V,3], coil_sens[C,V].')
    voxel_size=0.5/torch.as_tensor(data.nyquist,device=dev,dtype=dtype)
    if voxel_size.shape!=(3,) or not bool((torch.isfinite(voxel_size)&(voxel_size>0)).all()):
        raise ValueError('Need three finite positive voxel dimensions from data.nyquist.')
    counts=[int((r.adc_usage>0).sum()) for r in reps]
    offsets=np.concatenate(([0],np.cumsum(counts)))
    if offsets[-1]==0: raise ValueError('Sequence has no acquired samples.')
    options=dict(spin_count=int(spin_count),spin_dist=spin_dist,r2_seed=r2_seed,
                 seed=int(seed),perfect_spoiling=bool(perfect_spoiling),
                 spin_chunk=int(spin_chunk),voxel_batch=int(voxel_batch),metadata=run_metadata)
    fingerprint=simulation_fingerprint(seq,data,spans,options)
    checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None
    restored=resume and checkpoint_path is not None and checkpoint_path.exists()
    if restored:
        try: state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        except TypeError: state=torch.load(checkpoint_path,map_location='cpu')
        if state.get('fingerprint')!=fingerprint:
            raise ValueError('Checkpoint inputs/code differ. Use a new checkpoint directory or --no-resume.')
        lo=int(state['next_voxel']); next_rep=int(state['next_rep'])
        spin_pos=state['spin_pos'].to(dev); omega=state['omega'].to(dev)
        total=state['signal'].to(dev); spins=state['spins']
        if spins is not None: spins=spins.to(dev)
        if print_progress:
            print(f'Resume: {lo}/{V} voxels completed; current batch has {next_rep}/{len(reps)} RF reps completed.',flush=True)
    else:
        generator=torch.Generator(device=dev).manual_seed(seed)
        if spin_dist=='rand':
            u=torch.rand((spin_count,3),generator=generator,device=dev,dtype=dtype)
        elif spin_dist=='r2':
            first=torch.rand(3,generator=generator,device=dev) if r2_seed is None else torch.as_tensor(r2_seed,device=dev,dtype=dtype)
            if first.shape!=(3,) or not bool(((first>=0)&(first<=1)).all()):
                raise ValueError('r2_seed must contain three values in [0,1].')
            g=1.22074408460575947536
            a=1/torch.tensor([g,g*g,g*g*g],device=dev,dtype=dtype)
            u=(first+torch.arange(spin_count,device=dev,dtype=dtype)[:,None]*a)%1
        else:
            u=torch.full((spin_count,3),.5,device=dev,dtype=dtype)
            u[:,2]=(torch.arange(spin_count,device=dev,dtype=dtype)+.5)/spin_count
        spin_pos=2*torch.pi*(u-.5)*voxel_size
        off_res=torch.linspace(-.5,.5,spin_count,device=dev,dtype=dtype)
        omega=torch.tan(torch.pi*.999*off_res)
        omega=omega[torch.randperm(spin_count,generator=generator,device=dev)]
        total=torch.zeros((int(offsets[-1]),C),device=dev,dtype=torch.complex64)
        lo,next_rep,spins=0,0,None
    if total.shape!=(int(offsets[-1]),C): raise ValueError('Checkpoint signal shape differs.')
    spin_offset_m=spin_pos/(2*torch.pi)

    def save(next_voxel,rep_index,current_spins):
        if checkpoint_path is not None:
            atomic_torch_save(dict(schema=1,fingerprint=fingerprint,next_voxel=next_voxel,
                next_rep=rep_index,signal=total.detach().cpu(),
                spins=None if current_spins is None else current_spins.detach().cpu(),
                spin_pos=spin_pos.detach().cpu(),omega=omega.detach().cpu(),
                options={**options,'r2_seed':None if r2_seed is None else list(map(float,r2_seed))}),checkpoint_path)

    while lo<V:
        hi=min(lo+voxel_batch,V); d=_voxel_subset(data,lo,hi)
        if spins is None:
            spins=torch.zeros((hi-lo,spin_count,3),device=dev,dtype=dtype); spins[...,2]=1
        elif spins.shape!=(hi-lo,spin_count,3): raise ValueError('Checkpoint magnetization shape differs.')
        sensitivity=d.coil_sens.t().to(torch.complex64)*d.PD.to(dtype)[:,None]/spin_count
        for r in range(next_rep,len(reps)):
            rep=reps[r]
            if print_progress: print(f'Voxels {lo+1}-{hi}/{V}; RF rep {r+1}/{len(reps)}',flush=True)
            usage=getattr(rep.pulse.usage,'name',str(rep.pulse.usage)).upper()
            if perfect_spoiling and usage in ('EXCIT','EXCITATION','PULSEUSAGE.EXCIT'):
                spins[...,:2]=0
            spins=apply_bloch_pulse_rep_chunk(spins,spans[r],d,rep,
                    spin_offset_m,include_relaxation=False,spin_chunk=spin_chunk)
            start=spins; times=rep.event_time.cumsum(0).to(dev); moments=rep.gradm.cumsum(0).to(dev)
            selected=set(requested_profile_events(rep,r,profile_events)) if profile_callback is not None else set()
            local=[]
            for e in range(rep.event_count):
                spins=relax(start,d.T1,d.T2,times[e])
                spins=dephase(spins,omega,d.T2dash,times[e])
                spins=intravoxel_precess(spins,moments[e],spin_pos)
                spins=B0_precess(spins,d.B0,times[e])
                spins=grad_precess(spins,moments[e],d.voxel_pos)
                if e in selected:
                    profile_callback(repetition=r,event=e,stage='event',
                        voxel_indices=np.arange(lo,hi),
                        z_m=(d.voxel_pos[:,2,None]+spin_offset_m[None,:,2]).cpu().numpy().copy(),
                        M=spins.cpu().numpy().copy())
                if rep.adc_usage[e]>0:
                    local.append(measure(spins,sensitivity)*torch.exp(1j*rep.adc_phase[e]))
            if local: total[int(offsets[r]):int(offsets[r+1])]+=torch.stack(local)
            if (r+1)%checkpoint_every==0 or r+1==len(reps) or _STOP_REQUESTED:
                if r+1==len(reps): save(hi,0,None)
                else: save(lo,r+1,spins)
            if _STOP_REQUESTED: raise SystemExit('Stop requested; last completed RF repetition saved.')
        lo,next_rep,spins=hi,0,None
    return total


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nx',type=int,default=64); p.add_argument('--ny',type=int,default=64)
    p.add_argument('--nz',type=int,default=30); p.add_argument('--slice',type=int,default=14)
    p.add_argument('--seqs',nargs='+',default=['FLASH4mes','2D_spiral'])
    p.add_argument('--spins',nargs='+',type=int,default=[12000,15000,20000])
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--sequence-dir',type=Path,default=ROOT/'sequences')
    p.add_argument('--out-dir',type=Path,default=ROOT/'data'/'isochromat_sim_results')
    p.add_argument('--phantom',default='subject04_3T-highres-fat_custom_dense.npz')
    p.add_argument('--voxel-mm',nargs=3,type=float,default=[1.,1.,5.],metavar=('DX','DY','DZ'))
    p.add_argument('--no-b0',action='store_true'); p.add_argument('--no-b1',action='store_true')
    p.add_argument('--perfect-spoiling',action='store_true')
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints')
    p.add_argument('--resume',dest='resume',action='store_true',default=True)
    p.add_argument('--no-resume',dest='resume',action='store_false')
    p.add_argument('--checkpoint-every',type=int,default=1)
    p.add_argument('--spin-chunk',type=int,default=4000)
    p.add_argument('--voxel-batch',type=int,default=256)
    p.add_argument('--spin-dist',choices=['rand','r2','z'],default='rand')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--self-test',action='store_true')
    return p.parse_args(argv)


def _request_stop(signum,frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED=True
    print(f'Signal {signum}: will checkpoint after the current RF repetition.',flush=True)


def main(argv=None):
    args=parse_args(argv)
    if args.self_test: return self_test()
    from phantom.VoxelGridPhantom import VoxelGridPhantom
    from sequence import Sequence
    if min(args.nx,args.ny,args.nz,*args.spins,args.spin_chunk,args.voxel_batch,args.checkpoint_every)<1:
        raise ValueError('All dimensions/counts must be positive.')
    if not 0<=args.slice<args.nz: raise ValueError('Slice index is outside the interpolated volume.')
    if not all(np.isfinite(x) and x>0 for x in args.voxel_mm): raise ValueError('Invalid voxel dimensions.')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable.')
    for sig in (os_signal.SIGTERM,os_signal.SIGINT): os_signal.signal(sig,_request_stop)
    args.out_dir.mkdir(parents=True,exist_ok=True); args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    phantom_path=(args.data_dir/args.phantom).resolve(); phantom_sha=file_hash(phantom_path)
    for seq_name in args.seqs:
        seq_path=(args.sequence_dir/f'{seq_name}_sequence_raw.json').resolve()
        sequence_sha=file_hash(seq_path)
        for spin_count in args.spins:
            seq=Sequence.import_file_for_mrtini_sim(str(seq_path))
            phantom=VoxelGridPhantom.brainweb(str(phantom_path))
            phantom=phantom.interpolate(args.nx,args.ny,args.nz).slices([args.slice])
            dimensions=np.asarray(args.voxel_mm)*1e-3
            phantom.size=torch.as_tensor(dimensions*np.array([args.nx,args.ny,1]),
                                        dtype=phantom.size.dtype,device=phantom.size.device)
            data=phantom.build(voxel_shape='box')
            if args.no_b0: data.B0=torch.zeros_like(data.B0)
            if args.no_b1:
                data.B1=torch.zeros_like(data.B1)
                if data.B1.ndim==1: data.B1.fill_(1)
                else: data.B1[0].fill_(1)  # Unit net transmit field, also with several TX channels.
            if args.device.startswith('cuda'):
                with torch.cuda.device(torch.device(args.device)):
                    data=data.cuda(); seq=seq.cuda()
            else:
                if hasattr(data,'cpu'): data=data.cpu()
                if hasattr(seq,'cpu'): seq=seq.cpu()
            actual=(.5/torch.as_tensor(data.nyquist)).detach().cpu().numpy()
            if not np.allclose(actual,dimensions,rtol=1e-5,atol=1e-9):
                raise ValueError(f'Built voxel dimensions {actual} differ from requested {dimensions}.')
            metadata=dict(sequence_file=str(seq_path),sequence_sha256=sequence_sha,
                phantom_file=str(phantom_path),phantom_sha256=phantom_sha,
                nx=args.nx,ny=args.ny,nz=args.nz,slice=args.slice,voxel_mm=args.voxel_mm,
                no_b0=args.no_b0,no_b1=args.no_b1,perfect_spoiling=args.perfect_spoiling,
                spins=spin_count,spin_dist=args.spin_dist,seed=args.seed,
                spin_chunk=args.spin_chunk,voxel_batch=args.voxel_batch,device=args.device,
                torch_version=str(torch.__version__),source_sha256=file_hash(__file__),
                voxel_shape='box',rf_relaxation=False,diffusion=False,
                chemical_shift='No additional shift; data.B0 used as supplied')
            key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:12]
            stem=f'{seq_name}_nx{args.nx}_ny{args.ny}_nz{args.nz}_slice{args.slice}_spins{spin_count}_{key}'
            print(f'Running {stem}\nVoxel mm: {actual*1e3}; FOV mm: {phantom.size.detach().cpu().numpy()*1e3}',flush=True)
            print('Physics: supplied RF/event conventions; no RF relaxation or diffusion; no added species shift.',flush=True)
            signal=isochromat_sim(seq,data,spin_count,perfect_spoiling=args.perfect_spoiling,
                spin_dist=args.spin_dist,seed=args.seed,spin_chunk=args.spin_chunk,voxel_batch=args.voxel_batch,
                checkpoint_path=args.checkpoint_dir/f'{stem}.pt',resume=args.resume,
                checkpoint_every=args.checkpoint_every,run_metadata=metadata)
            out=args.out_dir/f'bloch_sim_signal_{stem}.npz'
            atomic_npz(out,signal=signal.cpu().numpy(),metadata_json=json.dumps(metadata,sort_keys=True),
                voxel_size_m=actual,phantom_size_m=phantom.size.detach().cpu().numpy(),
                adc_counts=np.array([int((r.adc_usage>0).sum()) for r in seq]))
            print(f'Saved: {out}',flush=True)
            del signal,data,seq,phantom
            if torch.cuda.is_available(): torch.cuda.empty_cache()


def self_test():
    """Small CPU test: RF sign, singleton chunk shapes, voxel batching and interrupted resume."""
    V,S=3,7; dtype=torch.float32
    data=SimpleNamespace(PD=torch.ones(V),B0=torch.tensor([0.,3.,-2.]),B1=torch.ones(1,V),
        T1=torch.full((V,),1.93),T2=torch.full((V,),.099),T2dash=torch.full((V,),.2),
        voxel_pos=torch.tensor([[0.,0.,0.],[.001,0.,0.],[0.,.001,0.]]),
        coil_sens=torch.ones(1,V,dtype=torch.complex64),nyquist=torch.tensor([500.,500.,100.]))
    class TestSequence(list): pass
    seq=TestSequence(); seq.mrtini_hybrid=[]; seq.mrtini_spans_by_rep=[]
    for r in range(3):
        alpha=np.pi/2 if r==0 else np.pi/3; T=.0002
        span=dict(kind='bloch',t=torch.tensor([0.,T]),rf_amp=torch.tensor([alpha/(2*np.pi*T)]*2),
                  gz=torch.zeros(2),rf_phase=torch.zeros(2),rf_freq=torch.zeros(2))
        seq.mrtini_hybrid.append(span); seq.mrtini_spans_by_rep.append([r])
        seq.append(SimpleNamespace(event_count=2,event_time=torch.tensor([.001,.002]),
            gradm=torch.tensor([[17.,0.,0.],[-4.,3.,0.]]),adc_usage=torch.ones(2,dtype=torch.int32),
            adc_phase=torch.tensor([0.,.1]),pulse=SimpleNamespace(usage='EXCIT',angle=alpha,phase=0.)))
    spins=torch.zeros(V,S,3); spins[...,2]=1
    d=_voxel_subset(data,0,V); d.B0=torch.zeros_like(d.B0)
    m=apply_bloch_pulse_rep_chunk(spins,seq.mrtini_hybrid[0],d,seq[0],torch.zeros(S,3),spin_chunk=1)
    torch.testing.assert_close(m,torch.tensor([0.,-1.,0.]).expand_as(m),atol=2e-6,rtol=0)
    # Pure gradient must disappear from the centre-referenced RF operator.
    z=torch.linspace(-.003,.003,3)[:,None]
    angle=2*torch.pi*1000*.0002*z
    R=torch.zeros(V,1,3,3); c,s=torch.cos(angle),torch.sin(angle)
    R[...,0,0]=c; R[...,0,1]=-s; R[...,1,0]=s; R[...,1,1]=c; R[...,2,2]=1
    mix=demod_Rxyz_to_Rmix_center(Rz_SB33Nz=R,z_m=z,t=torch.tensor([0.,.0002]),
        gz=torch.full((2,),1000/42.577478e6),gamma_rad=2*torch.pi*42.577478e6)
    torch.testing.assert_close(mix,torch.eye(3).expand_as(mix),atol=2e-6,rtol=0)
    kw=dict(spin_count=S,seed=7,print_progress=False)
    all_at_once=isochromat_sim(seq,data,voxel_batch=3,spin_chunk=7,**kw)
    batched=isochromat_sim(seq,data,voxel_batch=2,spin_chunk=1,**kw)
    torch.testing.assert_close(all_at_once,batched,atol=3e-6,rtol=3e-6)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'resume.pt'
        def interrupt(**values):
            if values['repetition']==1: raise InterruptedError('Deliberate test interruption')
        try: isochromat_sim(seq,data,voxel_batch=2,spin_chunk=1,checkpoint_path=path,profile_callback=interrupt,**kw)
        except InterruptedError: pass
        assert path.exists(),'First repetition did not create a checkpoint.'
        resumed=isochromat_sim(seq,data,voxel_batch=2,spin_chunk=1,checkpoint_path=path,**kw)
        torch.testing.assert_close(resumed,batched,atol=0,rtol=0)
        changed=SimpleNamespace(**vars(data)); changed.B0=data.B0+1
        try: isochromat_sim(seq,changed,voxel_batch=2,spin_chunk=1,checkpoint_path=path,**kw)
        except ValueError: pass
        else: raise AssertionError('A checkpoint with different fields was accepted.')
    print('SELF-TEST PASSED: RF sign, centre-reference shape, batching, interrupted resume and input guard.')


if __name__=='__main__': main()
