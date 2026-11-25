"""Read-only research diagnostics. Raw data retained; no simulation approximation."""
import numpy as np
from pathlib import Path
from analysis_views import cpu, physical_masks, counts

MAP_UNITS={'PD':'a.u.','T1':'s','T2':'s','T2dash':'s','D':'10⁻³ mm²/s','B0':'Hz','B1':'relative magnitude','water_PD':'a.u.','fat_PD':'a.u.','fat_fraction':'PD fraction'}

def _regular_axis(coordinates):
    """Recover a regular physical lattice from flattened float coordinates.

    Cluster numerical duplicates before estimating a pitch, then fit the pitch
    over all occupied coordinate levels instead of accumulating the smallest
    (potentially rounded) difference. Missing interior levels remain empty.
    """
    v=np.asarray(coordinates,dtype=np.float64).reshape(-1)
    if not v.size or not np.isfinite(v).all():raise ValueError('Invalid voxel coordinates')
    # Metres: 0.2 micrometre, or an allowance for float32 roundoff at large offsets.
    tol=max(2e-7,4*np.finfo(np.float32).eps*float(np.max(np.abs(v))))
    u=np.unique(v); cuts=np.r_[0,np.flatnonzero(np.diff(u)>tol)+1,len(u)]
    levels=np.array([u[a:b].mean() for a,b in zip(cuts[:-1],cuts[1:])])
    if len(levels)==1:
        if np.max(np.abs(v-levels[0]))>tol:raise ValueError('Voxel coordinate cluster too wide')
        return levels,np.zeros(len(v),dtype=np.int64)
    gaps=np.diff(levels);small=float(gaps.min())
    # Use the smallest occupied-level gap only to assign provisional integer
    # indices. Robust median and global fit remove systematic pitch bias.
    pitch=float(np.median(gaps[gaps<=small+max(2*tol,small*.005)]))
    j=np.rint((levels-levels[0])/pitch).astype(np.int64)
    if np.any(np.diff(j)<=0) or j[-1]>=2048:raise ValueError('Voxel positions are not a regular 2D grid')
    centered=j-j.mean();pitch=float(centered@(levels-levels.mean())/(centered@centered))
    origin=float(np.mean(levels-pitch*j));axis=origin+np.arange(j[-1]+1)*pitch
    ix=np.rint((v-origin)/pitch).astype(np.int64)
    if np.any(ix<0) or np.any(ix>=len(axis)):raise ValueError('Voxel coordinates outside fitted lattice')
    if np.max(np.abs(axis[ix]-v))>max(tol,pitch*1e-3):raise ValueError('Voxel positions are not on a regular grid')
    return axis,ix

def native_maps(data, physical):
    """Rasterize actual flattened active voxel positions, not reconstruction indices.
    Separate species maps; PD-weighted averages for overlapping entries within species.
    Missing PD is zero; missing tissue parameters are NaN. Axes are in metres.
    """
    pos=cpu(data.voxel_pos); pd=cpu(data.PD).reshape(-1); sp=cpu(physical).reshape(-1)
    if not len(pd) or pos.shape!=(len(pd),3) or sp.size!=len(pd):raise ValueError(f'Unsupported voxel positions: {pos.shape}, {len(pd)} PD values, {sp.size} species labels')
    if not np.isfinite(pos).all():raise ValueError('Invalid voxel coordinates')
    if not np.isin(sp,[0,1]).all():raise ValueError('Unknown physical species')
    x,ix=_regular_axis(pos[:,0]);y,iy=_regular_axis(pos[:,1]);shape=(len(y),len(x));flat=iy*len(x)+ix
    z=pos[:,2].astype(float);ztol=max(2e-7,4*np.finfo(np.float32).eps*np.max(np.abs(z)))
    if np.ptp(z)>2*ztol:raise ValueError('Map inspector supports a single physical slice')
    values={k:cpu(getattr(data,k)).reshape(-1) for k in ('T1','T2','T2dash','D','B0') if hasattr(data,k)}
    values['B1']=np.abs(cpu(data.B1).sum(0)).reshape(-1)
    out={'x_m':x,'y_m':y}; total=np.zeros(np.prod(shape),float); np.add.at(total,flat,np.abs(pd))
    for name,s in [('water',0),('fat',1)]:
        mask=sp==s; weights=np.zeros_like(total);np.add.at(weights,flat[mask],np.abs(pd[mask]));out[name+'_PD']=weights.reshape(shape)
        ids=np.full(len(total),-1,dtype=np.int64);ids[flat[mask]]=np.flatnonzero(mask);out[name+'_index']=ids.reshape(shape)
        for k,v in values.items():
            if len(v)!=len(pd):continue
            num=np.zeros_like(total);np.add.at(num,flat[mask],(np.abs(pd)*v)[mask]); out[name+'_'+k]=np.divide(num,weights,out=np.full_like(num,np.nan),where=weights>0).reshape(shape)
    out['PD']=total.reshape(shape);out['fat_fraction']=np.divide(out['fat_PD'],out['PD'],out=np.full(shape,np.nan),where=out['PD']>0)
    return out

def preview_maps(maps,p):
    n=p.resolution;axis=(np.arange(n)-n//2)*p.fov_mm/1000/n
    return {'x_m':axis,'y_m':axis,**{k:np.asarray(v) for k,v in maps.items()}}

def overview(g):
    m=physical_masks(g.node_mask,g.case_species); layers=[]
    for i,(a,b) in enumerate(zip(g.offsets[:-1],g.offsets[1:])):
        a,b=int(a),int(b);lo,hi=int(g.rows[a]),int(g.rows[b]); em=physical_masks(g.mask[lo:hi],g.case_species)
        layers.append(dict(layer=i,**counts(m[a:b]),total=b-a,edges=hi-lo,edge_counts=counts(em),emitting=int(np.count_nonzero(np.max(g.emitted[a:b],axis=1)>0)),latent=int(np.count_nonzero(np.max(g.latent[a:b],axis=1)>0))))
    return layers

def snapshot_graph(g,path):
    # State coordinates and membership, no GPU buffers. Edges kept only in latest graph.
    np.savez(path,offsets=g.offsets,kind=g.kind,kt_end=g.kt_end,kt_start=g.kt_start,mask=physical_masks(g.node_mask,g.case_species))

def compare_nodes(a,b):
    """Exact coordinate identity (no arbitrary tolerance/false pathway matches). Duplicate
    keys counted as multiset. Not transition/path-history equivalence."""
    from collections import Counter
    def keys(d):
        layer=np.repeat(np.arange(len(d['offsets'])-1),np.diff(d['offsets']));return Counter((int(l),int(k),*map(float,s),*map(float,e),int(m)) for l,k,s,e,m in zip(layer,d['kind'],d['kt_start'],d['kt_end'],d['mask']))
    x,y=keys(a),keys(b);return dict(shared=sum((x&y).values()),reference_only=sum((x-y).values()),current_only=sum((y-x).values()),basis='Exact layer, state kind, start/end k-t coordinates and species membership; not full transition identity.')

def magnitude(img):
    x=np.asarray(img).squeeze();return np.abs(x) if x.ndim==2 else np.sqrt(np.sum(np.abs(x)**2,axis=0))

def comparison(a,b,pa,pb,ma,mb):
    """a reference; b current. No regridding or implicit global phase alignment."""
    same_backend=pa['backend']==pb['backend'];same_geometry=same_backend and all(pa.get(k)==pb.get(k) for k in ('resolution','fov_mm','slice_mm','slice_index','phantom_file')) and pa.get('native_edit')==pb.get('native_edit')
    if ma.get('reconstruction_convention')!=mb.get('reconstruction_convention') or ma.get('reconstruction_fov_mm',pa.get('recon_fov_mm') or pa.get('fov_mm'))!=mb.get('reconstruction_fov_mm',pb.get('recon_fov_mm') or pb.get('fov_mm')):same_geometry=False
    if ma.get('reconstruction_method')!=mb.get('reconstruction_method') or pa.get('recon')!=pb.get('recon'):same_geometry=False
    if (pa.get('phantom_resolution') or pa.get('resolution'))!=(pb.get('phantom_resolution') or pb.get('resolution')):same_geometry=False
    # Imported trajectories do not establish image FOV from nominal controls.
    if pa['backend']=='native' and not pa.get('native_edit'):same_geometry=same_geometry and ma.get('sequence_file')==mb.get('sequence_file')
    x,y=magnitude(a['complex_image']),magnitude(b['complex_image']);same_geometry=same_geometry and x.shape==y.shape and ma.get('phantom_identity')==mb.get('phantom_identity')
    def rel(x,y):
        den=np.linalg.norm(x.ravel());return float(np.linalg.norm((y-x).ravel())/den) if den>0 else None
    same_adc=same_backend and a['signal'].shape==b['signal'].shape and a['kspace'].shape==b['kspace'].shape and np.allclose(a['kspace'],b['kspace'],atol=1e-6,rtol=0)
    return dict(image_reference=x,image_current=y,difference=y-x if same_geometry else None,image_l2=rel(x,y) if same_geometry else None,adc_l2=rel(a['signal'],b['signal']) if same_adc else None,same_geometry=bool(same_geometry),same_adc=bool(same_adc),limit=max(float(np.max(x)),float(np.max(y)),1e-20))

def rf_diagnostics(engine,rep,voxel=None,species=0):
    """Actual stored Fourier taps of a selected RF bank column, and signed spatial
    synthesis in the same +k convention as the executor (phase-referenced)."""
    from analysis_views import REL
    if rep<0 or rep>=len(engine.steps):raise ValueError('RF repetition out of range')
    ids=cpu(engine.rf.physical).reshape(-1);eligible=np.flatnonzero(ids==species)
    if not len(eligible):raise ValueError('Species not present')
    if voxel is None:voxel=int(eligible[np.argmin(np.linalg.norm(cpu(engine.pos)[eligible],axis=1))])
    if not 0<=voxel<len(ids) or ids[voxel]!=species:raise ValueError('Voxel does not belong to selected species')
    bank=engine.steps[rep].bank;key=engine.workbench_rep_keys[rep];pulse=engine.art.pulses[key];taps=pulse.water.nominal_taps
    tapids=np.array(list(bank.tap_map),int);orders=cpu(taps.deltas).reshape(-1)[tapids];dk=float(taps.dk)
    coef=cpu(bank.table[:,voxel]).reshape(6,bank.K); order=np.argsort(orders);orders=orders[order];coef=coef[:,order]
    period=1/abs(dk) if dk else .01; z=np.linspace(-period/2,period/2,401);prof=coef@np.exp(2j*np.pi*orders[:,None]*dk*z[None,:]);phase=float(cpu(engine.steps[rep].phase[voxel]));factors=np.exp(1j*np.array([0,-1,1,0,2,1])*phase);coef=coef*factors[:,None];prof=prof*factors[:,None]
    return dict(rep=rep,voxel=voxel,species=species,orders=orders.tolist(),dk=dk,period_mm=period*1000,z_mm=(z*1000).tolist(),equilibrium_mxy=(np.sqrt(2)*np.abs(prof[2])).tolist(),equilibrium_mz=prof[3].real.tolist(),refocusing=np.abs(prof[4]).tolist(),relations=list(REL),real=coef.real.tolist(),imag=coef.imag.tolist(),profile_real=prof.real.tolist(),profile_imag=prof.imag.tolist(),B0=float(cpu(engine.data.B0).reshape(-1)[voxel]),source='Retained RF Fourier coefficient synthesis (not raw Bloch profile); includes executor RF phase; omitted taps remain omitted.')

# RF history classes distinguish mechanisms without claiming unique paths in a merged PDG.
HISTORY_NAMES=('Equilibrium / recovery','Direct transverse','Single conjugation (spin-echo candidate)', 'Multiple conjugations (refocusing train)', 'Stored longitudinal','Recalled storage (stimulated-echo candidate)')
# Native relation convention: ++, +z, z+, zz, -+, -z; source -> destination.
HISTORY_MAP=((0,1,2,3,4,5),(4,4,4,4,4,4),(1,1,1,1,5,5),(0,1,2,3,4,5),(0,2,3,3,4,5),(4,4,4,4,4,4))
def history_rf(H,src,dst,rel,factor,children):
    """Real-linear RF propagation, including conjugation, with disjoint history bins."""
    import torch
    out=torch.zeros((children,6),device=H.device,dtype=H.dtype)
    table=torch.as_tensor(HISTORY_MAP,device=H.device,dtype=torch.long)
    v=H[src];v=torch.where((rel>=4)[:,None],v.conj(),v)*factor[:,None]
    idx=dst[:,None]*6+table[rel]
    out.view(-1).index_add_(0,idx.reshape(-1),v.reshape(-1))
    return out

def trace_ancestry(engine, voxel, case, rep, row, category, history, recoveries, adc, slab, max_layers=12):
    """Trace one locally strongest predecessor within a history class at each RF.
    This is a representative ancestry branch of a merged state, not its isolated
    complex contribution. Actual executor coordinates, masks and RF weights are used.
    Recovery is an independent source and terminates the branch when selected.
    """
    import torch
    from analysis_views import REL
    mapping=np.asarray(HISTORY_MAP)
    d=engine.data;scale=1/d.size if engine.seq.normalized_grads else torch.ones_like(d.size)
    kind_names=('F+','Z','Z0');steps=[];r=rep;child=int(row);cat=int(category);reason='Display window limit'
    # Physical ADC coordinate of this specific emitted state.
    selected=engine.steps[rep];idx=(selected.rows==row).nonzero(as_tuple=True)[0][0];at=selected.k0[idx]+selected.adc[adc]
    adct=cpu(at).tolist();deph=float(cpu(selected.deph[idx,adc]));selected_time=float(cpu(selected.adc[adc,3]))
    while r>=0 and len(steps)<max_layers:
        st=engine.steps[r];plan=st.plan;bank=st.bank;ts=plan.tensors(bank)
        # CSR slice only: do not copy all RF edges to host.
        lo,hi=int(plan.rows[child]),int(plan.rows[child+1]);src=ts['src'][lo:hi];relations=np.asarray(plan.rel[lo:hi],dtype=int)
        kind=int(st.kind[child]);end=st.end_kt[child];rr=engine.seq[r]
        traj=torch.cumsum(torch.cat((rr.gradm*scale[None,:],rr.event_time[:,None]),1),0)
        start=end-traj[-1] if kind==0 else end
        # For the selected final repetition stop the free-evolution trace at ADC.
        finish=at if r==rep else end
        # Preserve exact event endpoints (bounded rendering; selected ADC endpoint exact).
        tt=cpu(traj[:,3]);kk=cpu(traj[:,2]);duration=selected_time if r==rep else float(tt[-1]);valid=np.flatnonzero(tt<=duration+1e-8)
        if len(valid)>96:valid=valid[np.unique(np.r_[np.linspace(0,len(valid)-1,94,dtype=int),int(np.argmin(kk[valid])),int(np.argmax(kk[valid]))])]
        curve=[[0.,float(start[2])]]
        if kind==0:curve.extend([[float(tt[j]*1000),float(start[2])+float(kk[j])] for j in valid if tt[j]<duration-1e-8])
        curve.append([duration*1000,float(finish[2])]);record=dict(rep=r,local=child,kind=kind_names[kind],category=HISTORY_NAMES[cat],start_kt=cpu(start).tolist(),end_kt=cpu(finish).tolist(),curve=curve,duration_ms=duration*1000,relation=None)
        steps.append(record)
        if not len(relations):reason='No retained RF predecessor';break
        rel=torch.as_tensor(relations,device=engine.device);mask=torch.as_tensor(plan.mask[lo:hi],device=engine.device,dtype=torch.long)
        f=torch.exp(1j*torch.tensor([0,-1,1,0,2,1],device=engine.device)*st.phase[voxel])
        a=history[r][src];a=torch.where((rel>=4)[:,None],a.conj(),a)*bank.table[ts['slot'][lo:hi],voxel,None]*f[rel,None]*((mask>>case)&1)[:,None]
        eligible=torch.as_tensor(mapping[relations]==cat,device=engine.device);score=torch.where(eligible,a.abs(),torch.zeros_like(a.real));flat=int(score.argmax());edge,previous_cat=divmod(flat,6)
        if float(score[edge,previous_cat])<=0:reason='No nonzero RF predecessor in this class';break
        parent=int(src[edge]);record.update(relation=REL[relations[edge]],tap=int(plan.tap[lo+edge]),parent_local=parent,predecessor_magnitude=float(score[edge,previous_cat]),incoming_terms=int((score>0).sum()))
        if r==0:record['parent_kt']=[0.,0.,0.,0.];record['parent_kind']='Z0';reason='Initial equilibrium';break
        parent_step=engine.steps[r-1];record['parent_kt']=cpu(parent_step.end_kt[parent]).tolist();record['parent_kind']=kind_names[int(parent_step.kind[parent])]
        # The parent class may include a fresh recovery source. If it dominates
        # the pre-RF class over the propagated remainder, stop at that source.
        if previous_cat==0:
            rec=recoveries[r-1][parent];total=history[r][parent,0]
            if float(rec.abs())>0 and float(rec.abs())>=float((total-rec).abs()):reason='Longitudinal recovery source';break
        child=parent;cat=previous_cat;r-=1
    return dict(steps=list(reversed(steps)),origin=reason,adc_kt=adct,adc_dephasing=deph,slab_mm=slab*1000,z_weight_applied=not bool(engine.data.multi_slice),z_weight=float(np.sinc(adct[2]*slab)),trace_note='Locally strongest retained RF-history predecessor at each step; merged amplitudes are not isolated path amplitudes.')

def voxel_replay(engine,voxel,rep=0,*,slice_extent_m=None,adc_index=None,profile_samples=257,slab_thickness_m=None,trace_paths=True):
    """One actual voxel, same graph and coefficients; read-only diagnostic replay.
    Uses torch indexed accumulation rather than changing the production executor.
    Point magnetization at repetition ends, not voxel-averaged ADC signal.
    """
    import torch
    if not 0<=voxel<engine.V or not 0<=rep<len(engine.steps):raise ValueError('Voxel or repetition out of range')
    d=engine.data;device=engine.device;v=voxel;case=int(engine.ids[v]);pos=engine.pos[v];b0=engine.B0[v];D=engine.D[v];t2d=engine.T2d[v];t2=engine.T2[v]
    M=torch.ones(1,device=device,dtype=torch.complex64);H=torch.zeros((1,6),device=device,dtype=M.dtype);H[0,0]=1;time=0.;times=[0.];values=[torch.stack((M[0].real*0,M[0].real*0,M[0].real))];contrib=None;detail=None
    profiles=None;history=[];recoveries=[]
    if profile_samples not in (121,257,513,1025):raise ValueError('Unsupported spatial sample count')
    slab=float(slab_thickness_m if slab_thickness_m is not None else (slice_extent_m/2 if slice_extent_m else 0.))
    if not 0<=slab<=.1:raise ValueError('Invalid slab thickness')
    # Exact rectangular through-plane average of the retained Fourier states.
    # No quadrature or independent Bloch isochromats are needed for this average.
    slab_values=[values[0]]
    if slice_extent_m is not None:
        if not 0<slice_extent_m<=.1:raise ValueError('Invalid profile extent')
        dz=torch.linspace(-slice_extent_m/2,slice_extent_m/2,profile_samples,device=device)
        keep=set(np.linspace(0,len(engine.steps)-1,min(64,len(engine.steps)),dtype=int).tolist())|{rep}
        profiles={'z_mm':(cpu(dz)*1000).tolist(),'samples':profile_samples,'slab_thickness_mm':slab*1000,'nyquist_cycles_m':(profile_samples-1)/(2*slice_extent_m),'max_kz_cycles_m':0.,'time_ms':[0.],'rep':[-1],'stage':['Equilibrium'],'mxy':[[0.]*len(dz)],'mz':[[1.]*len(dz)]}
        def profile(M,kt,kind,tm,i,stage):
            profiles['max_kz_cycles_m']=max(profiles['max_kz_cycles_m'],float(kt[:,2].abs().max().item()))
            # Same retained-state synthesis as point replay, changing z only.
            p0=torch.exp(2j*torch.pi*(kt[:,3]*b0+kt[:,:3]@pos))*torch.exp(-kt[:,3].abs()/t2d)
            weight=M[:,None]*p0[:,None]*torch.exp(2j*torch.pi*kt[:,2:3]*dz[None,:])
            f=2**.5*(weight*(kind==0)[:,None]).sum(0);z=(weight*(kind!=0)[:,None]).sum(0)
            profiles['time_ms'].append(tm*1000);profiles['rep'].append(i);profiles['stage'].append(stage);profiles['mxy'].append(cpu(f.abs()).tolist());profiles['mz'].append(cpu(z.real).tolist())
    with torch.no_grad():
        for i,s in enumerate(engine.steps):
            plan=s.plan;bank=s.bank;ts=plan.tensors(bank);src=ts['src'];dst=ts['dst'];rel=torch.as_tensor(plan.rel,device=device,dtype=torch.long);mask=torch.as_tensor(plan.mask,device=device,dtype=torch.long)
            # Same relation/conjugation and phase convention as rf_reference.
            factor=torch.exp(1j*torch.tensor([0,-1,1,0,2,1],device=device)*s.phase[v]);val=M[src];val=torch.where(rel>=4,val.conj(),val)*bank.table[ts['slot'],v]*factor[rel]*((mask>>case)&1)
            X=torch.zeros(plan.children,device=device,dtype=torch.complex64);X.index_add_(0,dst,val)
            if i<=rep:
                if trace_paths:history.append(H)
                HX=history_rf(H,src,dst,rel,bank.table[ts['slot'],v]*factor[rel]*((mask>>case)&1),plan.children)
            if i==rep and s.adc.shape[0]:
                j=int(s.adc.shape[0]//2) if adc_index is None else int(adc_index)
                if not 0<=j<s.adc.shape[0]:raise ValueError(f'ADC index out of range for this layer (0–{s.adc.shape[0]-1})')
                Q=2**.5*X[s.rows]*torch.exp(2j*torch.pi*(s.k0[:,3]*b0+s.k0[:,:3]@pos));
                if s.allowed is not None:Q=Q*s.allowed[:,case]
                a=s.adc[j];event=torch.exp(2j*torch.pi*(a[3]*b0+a[:3]@pos))*torch.exp(-a[3]/t2)
                contrib=Q*torch.exp(-s.t[:,j].abs()/t2d+s.cb[:,j]*D)*s.deph[:,j]*event*s.rot[j,0]*engine.coil[v,0]
                W=2**.5*torch.exp(2j*torch.pi*(s.k0[:,3]*b0+s.k0[:,:3]@pos))*torch.exp(-s.t[:,j].abs()/t2d+s.cb[:,j]*D)*s.deph[:,j]*event*s.rot[j,0]*engine.coil[v,0]
                if s.allowed is not None:W=W*s.allowed[:,case]
                weighted=HX[s.rows]*W[:,None];hc=cpu(weighted.sum(0));tau=cpu(s.t[:,j]);refocused=np.abs(tau)<=1e-5;near=cpu(weighted[torch.as_tensor(refocused,device=device)].sum(0));err=float(abs(hc.sum()-cpu(contrib).sum()))
                # Sample the actual selected readout, including the inspected ADC.
                sample_ix=np.unique(np.r_[np.linspace(0,s.adc.shape[0]-1,min(512,s.adc.shape[0]),dtype=int),j]); traces=[]
                base=2**.5*torch.exp(2j*torch.pi*(s.k0[:,3]*b0+s.k0[:,:3]@pos))
                if s.allowed is not None:base=base*s.allowed[:,case]
                for block in np.array_split(sample_ix,max(1,int(np.ceil(len(sample_ix)/64)))):
                    ix=torch.as_tensor(block,device=device,dtype=torch.long);aa=s.adc[ix];ev=torch.exp(2j*torch.pi*(aa[:,3]*b0+aa[:,:3]@pos))*torch.exp(-aa[:,3]/t2)*s.rot[ix,0]*engine.coil[v,0]
                    weights=base[:,None]*torch.exp(-s.t[:,ix].abs()/t2d+s.cb[:,ix]*D)*s.deph[:,ix]*ev[None,:]
                    traces.append(cpu(weights.T@HX[s.rows]))
                traces=np.concatenate(traces,axis=0)
                readout=dict(adc_indices=(sample_ix+s.offset).tolist(),time_ms=(time*1000+cpu(s.adc[sample_ix,3])*1000).tolist(),real=traces.real.T.tolist(),imag=traces.imag.T.tolist())
                ids=cpu(s.rows).astype(int)+(int(engine.graph.offsets[i+1]) if hasattr(engine.graph,'offsets') else sum(len(layer) for layer in engine.graph[:i+1]));c=cpu(contrib);order=np.argsort(-np.abs(c))[:30];detail={'rep':i,'adc_index':s.offset+j,'readout':readout,'total_states':len(c),'history_classes':[{'name':name,'real':float(a.real),'imag':float(a.imag),'near_zero_time_real':float(b.real),'near_zero_time_imag':float(b.imag)} for name,a,b in zip(HISTORY_NAMES,hc,near)],'history_sum_error':err,'coherence_tolerance_ms':.01,'history_note':'Disjoint RF history classes; one conjugation or storage/recall are echo candidates, not guaranteed echo identities. Selected voxel, selected ADC, coil 1.', 'adc_in_rep':j,'adc_count_in_rep':int(s.adc.shape[0]),'sum_real':float(c.sum().real),'sum_imag':float(c.sum().imag),'states':[{'id':int(ids[k]),'real':float(c[k].real),'imag':float(c[k].imag),'coherence_time_ms':float(tau[k]*1000),'history_classes':[{'name':name,'real':float(value.real),'imag':float(value.imag)} for name,value in zip(HISTORY_NAMES,cpu(weighted[k]))]} for k in order]}
                # Inspect exact emitted-state coordinates, and offer a representative
                # branch for each nonzero class plus the strongest total states.
                weighted_cpu=cpu(weighted);allkt=cpu(s.k0+s.adc[j]);all_deph=cpu(s.deph[:,j])
                for q,k in enumerate(order):
                    state=detail['states'][q];kt=allkt[k].tolist();state.update(adc_kt=kt,adc_dephasing=float(all_deph[k]),z_weight=float(np.sinc(kt[2]*slab)))
                detail['ancestry_options']=[]
                if trace_paths and len(c):
                    choices=[]
                    for k in order[:4]:
                        if abs(c[k])>0:choices.append((int(k),int(np.argmax(abs(weighted_cpu[k])))))
                    for cat in range(6):
                        k=int(np.argmax(abs(weighted_cpu[:,cat])))
                        if abs(weighted_cpu[k,cat])>0:choices.append((k,cat))
                    for k,cat in dict.fromkeys(choices):
                        trace=trace_ancestry(engine,v,case,i,int(s.rows[k]),cat,history,recoveries,j,slab)
                        for stage in trace['steps']:stage['time_ms']=times[stage['rep']]
                        q=weighted_cpu[k,cat];detail['ancestry_options'].append(dict(id=int(ids[k]),category=cat,real=float(q.real),imag=float(q.imag),state_real=float(c[k].real),state_imag=float(c[k].imag),trace=trace))
            if profiles is not None and i in keep:
                r=engine.seq[i]; scale=1/d.size if engine.seq.normalized_grads else torch.ones_like(d.size)
                endtraj=torch.cat(((r.gradm*scale[None,:]).sum(0),r.event_time.sum().reshape(1)))
                kt=torch.where((s.kind==0)[:,None],s.end_kt-endtraj,s.end_kt)
                profile(X*s.live,kt,s.kind,time,i,'After RF (centre convention)')
            rec=(s.kind==2)*(1-s.r1[v])*((s.bits>>case)&1);decay=s.live*torch.where(s.kind==0,s.r2[v],s.r1[v])*torch.exp(D*s.b);M=X*decay+rec
            if i<rep:
                if trace_paths:recoveries.append(rec)
                H=HX*decay[:,None];H[:,0]+=rec
            phase=torch.exp(2j*torch.pi*(s.end_kt[:,3]*b0+s.end_kt[:,:3]@pos))*torch.exp(-s.end_kt[:,3].abs()/t2d);f=(M*phase*(s.kind==0)).sum()*2**.5;z=(M*phase*(s.kind!=0)).sum();values.append(torch.stack((f.real,f.imag,z.real)));avg=M*phase*torch.sinc(s.end_kt[:,2]*slab);af=2**.5*(avg*(s.kind==0)).sum();az=(avg*(s.kind!=0)).sum();slab_values.append(torch.stack((af.real,af.imag,az.real)));time+=float(cpu(engine.seq[i].event_time.sum()));times.append(time*1000)
            if profiles is not None and i in keep:profile(M,s.end_kt,s.kind,time,i,'End of repetition')
        m=cpu(torch.stack(values));av=cpu(torch.stack(slab_values));
    return dict(slice_profiles=profiles,slab_average=dict(mx=av[:,0].tolist(),my=av[:,1].tolist(),mz=av[:,2].tolist(),thickness_mm=slab*1000),time_ms=times,mx=m[:,0].tolist(),my=m[:,1].tolist(),mz=m[:,2].tolist(),contributions=detail,voxel=v,position_mm=(cpu(pos)*1000).tolist(),source='Diagnostic replay of retained state graph; point magnetization at repetition ends, normalized to equilibrium. ADC contributions include PD, finite voxel dephasing and coil 1. Parallel accumulation may differ in FP32 rounding.')


def finite_rf_profile(engine,design,rep,voxel,extent_m):
    """Independent rotation-only Bloch diagnostic for generated held waveforms.
    Uses the selected voxel's B0/B1; not the original baker or a replacement for it.
    No relaxation or diffusion inside the RF span; initial equilibrium M=(0,0,1).
    """
    r=design.pulses[rep]['span']; t=np.asarray(r['t'],float); dt=np.diff(t)
    if np.any(dt<=0) or np.any(np.asarray(r['rf_freq'])):raise ValueError('Only zero-carrier generated finite RF spans supported')
    shim=cpu(engine.seq[rep].pulse.shim_array); b1=cpu(engine.data.B1)[:,voxel]
    b1eff=b1.sum() if len(shim)==1 else np.sum(b1*shim[:,0]*np.exp(-1j*shim[:,1]))
    pos=cpu(engine.pos[voxel]).astype(float); b0=float(cpu(engine.B0[voxel]));z=np.linspace(-extent_m/2,extent_m/2,401);m=np.zeros((len(z),3));m[:,2]=1
    for j,step in enumerate(dt):
        a=float(r['rf_amp'][j])*abs(b1eff);ph=float(r['rf_phase'][j])+np.angle(b1eff)
        w=np.empty_like(m);w[:,0]=a*np.cos(ph);w[:,1]=a*np.sin(ph);w[:,2]=b0+r['gx'][j]*pos[0]+r['gy'][j]*pos[1]+r['gz'][j]*(pos[2]+z)
        norm=np.linalg.norm(w,axis=1);axis=w/np.maximum(norm[:,None],1e-30);theta=2*np.pi*norm*step;co=np.cos(theta)[:,None];si=np.sin(theta)[:,None]
        m=m*co+np.cross(axis,m)*si+axis*(axis*m).sum(1)[:,None]*(1-co)
    return {'z_mm':(z*1000).tolist(),'mxy':np.linalg.norm(m[:,:2],axis=1).tolist(),'mz':m[:,2].tolist(),'B1_abs':float(abs(b1eff)),'source':'Independent finite-waveform Bloch rotation from equilibrium, without intra-RF relaxation/diffusion; not original RF baker output.'}


def adc_windows(rep, step):
    """Contiguous runs of sampled native events, not inferred echo identities.

    Return half-open ranges in the *local ADC index*, not event index. The
    prepared executor uses exactly the positive adc_usage events in this order.
    """
    events=np.flatnonzero(cpu(rep.adc_usage).reshape(-1)>0)
    if len(events)!=len(step.adc):
        raise ValueError('ADC event count does not match prepared executor')
    if not len(events):return []
    bounds=np.r_[0,np.flatnonzero(np.diff(events)>1)+1,len(events)]
    return [(int(a),int(b)) for a,b in zip(bounds[:-1],bounds[1:])]


def contribution_overview(engine,voxel):
    """One read-only replay, one actual midpoint sample per ADC window.

    Preserve RF histories across every layer, including layers without ADC.
    Compare against independently accumulated unclassified state amplitudes.
    No ancestry retention, profile synthesis or full-readout replay is needed.
    """
    import torch
    if not 0<=voxel<engine.V:raise ValueError('Voxel out of range')
    v=int(voxel);device=engine.device;case=int(engine.ids[v]);pos=engine.pos[v]
    b0=engine.B0[v];D=engine.D[v];t2d=engine.T2d[v];t2=engine.T2[v]
    phase_codes=torch.tensor([0,-1,1,0,2,1],device=device)
    H=torch.zeros((1,6),device=device,dtype=torch.complex64);H[0,0]=1
    M=torch.ones(1,device=device,dtype=torch.complex64)
    rows=[];time=0.;max_error=0.;max_reference=0.
    with torch.no_grad():
        for i,s in enumerate(engine.steps):
            plan=s.plan;ts=plan.tensors(s.bank);src=ts['src'];dst=ts['dst']
            rel=torch.as_tensor(plan.rel,device=device,dtype=torch.long)
            mask=torch.as_tensor(plan.mask,device=device,dtype=torch.long)
            factor=s.bank.table[ts['slot'],v]*torch.exp(1j*phase_codes[rel]*s.phase[v])*((mask>>case)&1)
            X=torch.zeros(plan.children,device=device,dtype=M.dtype)
            X.index_add_(0,dst,torch.where(rel>=4,M[src].conj(),M[src])*factor)
            HX=history_rf(H,src,dst,rel,factor,plan.children)
            windows=adc_windows(engine.seq[i],s)
            if windows:
                # Chunk windows to bound the temporary [emitting states, samples] array.
                base=2**.5*torch.exp(2j*torch.pi*(s.k0[:,3]*b0+s.k0[:,:3]@pos))
                if s.allowed is not None:base=base*s.allowed[:,case]
                for start in range(0,len(windows),32):
                    batch=windows[start:start+32];indices=[(a+b)//2 for a,b in batch]
                    ix=torch.as_tensor(indices,device=device,dtype=torch.long);aa=s.adc[ix]
                    ev=torch.exp(2j*torch.pi*(aa[:,3]*b0+aa[:,:3]@pos))*torch.exp(-aa[:,3]/t2)*s.rot[ix,0]*engine.coil[v,0]
                    W=base[:,None]*torch.exp(-s.t[:,ix].abs()/t2d+s.cb[:,ix]*D)*s.deph[:,ix]*ev[None,:]
                    hc=cpu(W.T@HX[s.rows]);ref=cpu((W*X[s.rows,None]).sum(0));tms=cpu(aa[:,3])*1000+time*1000
                    near=cpu(torch.einsum('sb,sc->bc',W*(s.t[:,ix].abs()<=1e-5),HX[s.rows]))
                    for n,((a,b),j) in enumerate(zip(batch,indices)):
                        h=hc[n];r=ref[n];err=float(abs(h.sum()-r));max_error=max(max_error,err);max_reference=max(max_reference,float(abs(r)))
                        rows.append(dict(window=len(rows),rep=i,adc_in_rep=j,adc_index=int(s.offset+j),adc_start=int(s.offset+a),adc_stop=int(s.offset+b),samples=b-a,time_ms=float(tms[n]),real=h.real.tolist(),imag=h.imag.tolist(),near_real=near[n].real.tolist(),near_imag=near[n].imag.tolist(),sum_real=float(r.real),sum_imag=float(r.imag),residual=err))
            rec=(s.kind==2)*(1-s.r1[v])*((s.bits>>case)&1)
            decay=s.live*torch.where(s.kind==0,s.r2[v],s.r1[v])*torch.exp(D*s.b)
            M=X*decay+rec;H=HX*decay[:,None];H[:,0]+=rec
            time+=float(cpu(engine.seq[i].event_time.sum()))
    return dict(voxel=v,position_mm=(cpu(pos)*1000).tolist(),coil=1,classes=list(HISTORY_NAMES),windows=rows,repetitions=len(engine.steps),adc_samples=sum(len(s.adc) for s in engine.steps),max_absolute_residual=max_error,max_reference_magnitude=max_reference,sampling='One midpoint ADC sample per contiguous sampled-event window; windows are not necessarily echoes or shots.',source='Selected physical voxel and coil 1; actual retained RF history. Magnitudes are not additive; the complex components are.')
