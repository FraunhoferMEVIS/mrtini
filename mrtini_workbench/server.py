from pathlib import Path
from dataclasses import asdict, fields
from concurrent.futures import ThreadPoolExecutor
import argparse, base64, io, time, uuid, threading, traceback, json, re
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from physics import Protocol, phantom, preview, reconstruct, line_plan
from native import NativeRunner
from analysis_views import graph_window, sequence_window, probe_on_graph
from native_sequences import design
from research import native_maps, preview_maps, overview, snapshot_graph, compare_nodes, comparison, rf_diagnostics, MAP_UNITS, voxel_replay, finite_rf_profile, contribution_overview

ROOT=Path(__file__).resolve().parent; app=FastAPI(title='MRTini Workbench');runner=NativeRunner();analysis_job=None;pool=ThreadPoolExecutor(max_workers=1);lock=threading.Lock();jobs={};active=None
class JobRequest(BaseModel):
    protocol: dict

def encode(x,phase=False,limit=None):
    a=np.asarray(x,float);lo=-np.pi if phase else 0;hi=np.pi if phase else (limit if limit is not None else float(np.percentile(a[a>0],99.7)) if (a>0).any() else 1)
    image=Image.fromarray(np.uint8(np.clip((a-lo)/max(hi-lo,1e-20),0,1)*255));out=io.BytesIO();image.save(out,format='PNG');return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()

def update(j,**kwargs):
    with lock:j.update(kwargs)

def run(j,p):
    global analysis_job
    analysis_job=None
    start=time.perf_counter();metrics={};meta={}
    def progress(stage,fraction,message):update(j,stage=stage,fraction=fraction,message=message,elapsed=time.perf_counter()-start,metrics=dict(metrics))
    def timed(stage,fn):
        progress(stage,None,stage.capitalize());t=time.perf_counter();v=fn();metrics[stage]=time.perf_counter()-t;return v
    try:
        update(j,status='running');maps=None
        if p.backend=='native':
            sig,k,meta,reco=runner.run(p,j['directory'],progress,metrics)
        else:
            maps=timed('phantom',lambda:phantom(p));sig,k=timed('execute',lambda:preview(p,maps,progress));metrics.update(active_voxels=int(np.count_nonzero(maps['PD']>1e-7)),repetitions=len(set(r['shot'] for r in line_plan(p))),prepared_reused=False,compile_cache='not applicable')
        if p.backend!='native':reco=timed('reconstruct',lambda:reconstruct(sig,k,p))
        metrics['total']=time.perf_counter()-start
        # Inspection artifacts are optional; graph construction for physics is not.
        directory=Path(j['directory']);warnings=[];graph_summary=None
        metrics['diagnostics_enabled']=p.diagnostics
        if p.diagnostics:
            progress('diagnostics',None,'Preparing inspection data')
            diag_start=time.perf_counter()
            try:
                maps=preview_maps(maps,p) if maps is not None else native_maps(runner.engine.data,runner.engine.rf.physical)
                np.savez_compressed(directory/'maps.npz',**maps)
            except Exception as e:
                maps=None;warnings.append('Phantom maps unavailable: '+str(e))
            if p.backend=='native':
                try:
                    graph_summary=overview(runner.engine.graph);snapshot_graph(runner.engine.graph,directory/'graph_nodes.npz')
                except Exception as e:warnings.append('Graph archive unavailable: '+str(e))
            metrics['diagnostics']=time.perf_counter()-diag_start
        else:
            maps=None
            warnings.append('Fast acquisition: inspection data skipped. Choose With diagnostics and run again to inspect maps and pathways.')
        progress('export',None,'Saving acquisition')
        export_start=time.perf_counter()
        design_export={} if reco.get('design_k') is None else {'expected_k_xy':reco['design_k'],'adc_times_s':reco['adc_times']}
        if reco.get('density_weights') is not None:
            dw=reco['density_weights'];design_export['density_weights']=dw;meta['density_compensation']={'method':'iterative NUFFT DCF','min':float(dw.min()),'max':float(dw.max()),'mean':float(dw.mean())}
        np.savez_compressed(Path(j['directory'])/'result.npz',signal=sig,kspace=k,complex_image=reco['complex_image'],**design_export,protocol_json=__import__('json').dumps(asdict(p)),metadata_json=__import__('json').dumps(meta),timings_json=__import__('json').dumps(metrics))
        metrics['export']=time.perf_counter()-export_start
        display_start=time.perf_counter()
        # Native phantom is not inferred from synthetic preview. Adapter may provide its own display image later.
        imgs={'magnitude':encode(reco['magnitude']),'phase':encode(reco['phase'],phase=True),'kspace':encode(reco['kspace'])}
        if maps is not None:imgs['phantom']=encode(np.nan_to_num(maps['PD']),limit=max(1,float(np.nanmax(maps['PD']))))
        a=np.linalg.norm(sig,axis=1);idx=np.linspace(0,len(a)-1,min(512,len(a)),dtype=int);kidx=np.linspace(0,len(k)-1,min(4000,len(k)),dtype=int)
        result=dict(job_id=j['id'],images=imgs,metrics=metrics,metadata=meta,method=reco['method'],adc_samples=len(sig),signal_trace=a[idx].tolist(),signal_index=idx.tolist(),trajectory=np.asarray(k)[kidx,:2].tolist(),protocol=asdict(p),preview=p.backend=='preview',download=f'/api/jobs/{j["id"]}/result')
        result.update(graph_overview=graph_summary,map_keys=[k for k in maps if k not in ('x_m','y_m') and not k.endswith('_index')] if maps is not None else [],warnings=warnings,created=time.time(),magnitude_limit=float(np.max(reco['magnitude'])))
        result['signal_l2']=float(np.linalg.norm(sig.ravel()));result['signal_peak']=float(np.max(a));metrics['display']=time.perf_counter()-display_start;metrics['backend_total']=time.perf_counter()-start
        (directory/'summary.tmp').write_text(json.dumps(result,allow_nan=False),encoding='utf-8');(directory/'summary.tmp').replace(directory/'summary.json')
        analysis_job=j['id'] if p.backend=='native' else None
        update(j,status='done',stage='done',fraction=1,message='Acquisition complete',metrics=dict(metrics),result=result,elapsed=time.perf_counter()-start)
    except Exception as e:
        traceback.print_exc();update(j,status='error',stage='error',message=str(e),elapsed=time.perf_counter()-start,metrics=dict(metrics))

@app.get('/api/status')
def status():return {'native_configured':runner.configured,'mode':'native available' if runner.configured else 'preview only','version':'5.4-fast-editor',**runner.catalog()}
@app.post('/api/jobs')
def submit(req:JobRequest):
    global active
    try:
        p=Protocol(**req.protocol).check()
        if p.backend=='native' and not runner.configured:raise ValueError('MRTini repository not found (README).')
    except (ValueError,TypeError) as e:raise HTTPException(422,str(e))
    with lock:
        if active in jobs and jobs[active]['status'] in ('queued','running'):raise HTTPException(409,'A simulation is already running')
        i=uuid.uuid4().hex;d=ROOT/'runs'/i;d.mkdir(parents=True);j=dict(id=i,status='queued',stage='queued',fraction=None,message='Queued',elapsed=0,metrics={},directory=str(d));jobs[i]=j;active=i
    pool.submit(run,j,p);return {'id':i}
@app.get('/api/jobs/{i}')
def state(i:str):
    with lock:
        if i not in jobs:raise HTTPException(404,'Unknown acquisition')
        return {k:v for k,v in jobs[i].items() if k!='directory'}
@app.get('/api/jobs/{i}/result')
def result(i:str):
    with lock:
        path=run_dir(i)/'result.npz'
        if not path.is_file():raise HTTPException(404,'No result')
    return FileResponse(path,filename=f'mrtini-{i[:8]}.npz')
def analysis_ready(i,require_diagnostics=True):
    if active in jobs and jobs[active]['status'] in ('queued','running'):raise HTTPException(409,'Acquisition is updating. Analysis will follow the latest result.')
    if i not in jobs or jobs[i]['status']!='done':raise HTTPException(404,'No completed acquisition')
    j=jobs[i]
    if require_diagnostics and not j['result'].get('protocol',{}).get('diagnostics',True):raise HTTPException(409,'Fast acquisition: choose With diagnostics and run again to enable analysis.')
    if not j['result']['preview'] and (analysis_job!=i or runner.engine is None):raise HTTPException(409,'Native graph belongs to another acquisition. Run this protocol again.')
    return j

@app.get('/api/jobs/{i}/graph')
def graph_view(i:str,start:int=0,window:int=6,mode:str='union',kind:str='all',limit:int=85,activity:str='all'):
    with lock:
        try:
            f=run_dir(i)/'graph_full.npz'
            if i!=analysis_job and f.is_file():
                from types import SimpleNamespace
                with np.load(f) as d:g=SimpleNamespace(**{k:d[k] for k in d.files})
                g.case_species=[int(x) for x in g.case_species]
            else:
                j=analysis_ready(i)
                if j['result']['preview']:raise HTTPException(422,'A real packed graph requires native MRTini execution.')
                g=runner.engine.graph
            return graph_window(g,start,window,mode,kind,limit,activity=activity)
        except ValueError as e:raise HTTPException(422,str(e))

@app.get('/api/jobs/{i}/strip')
def strip_view(i:str,start:int=0,window:int=4):
    with lock:
        j=analysis_ready(i,require_diagnostics=False)
        try:
            if j['result']['preview']:return sequence_window(design=design(Protocol(**j['result']['protocol'])),start=start,window=window,preview=True)
            return sequence_window(runner.seq,runner.design,start,window)
        except (ValueError,AttributeError,TypeError) as e:raise HTTPException(422,str(e))

@app.get('/api/jobs/{i}/probe')
def probe_view(i:str,tissue:str='gm'):
    with lock:
        j=analysis_ready(i)
        if j['result']['preview']:raise HTTPException(422,'Voxel replay requires the native packed graph.')
        try:
            import torch
            with torch.no_grad():return probe_on_graph(runner.engine,tissue,runner.meta['slice_thickness_m'],j['result']['protocol']['workspace_mb'])
        except Exception as e:raise HTTPException(422,str(e))


def run_dir(i):
    if not re.fullmatch(r'[0-9a-f]{32}',i):raise HTTPException(404,'Unknown acquisition')
    return ROOT/'runs'/i

def stored(i):
    p=run_dir(i)/'summary.json'
    if not p.is_file():raise HTTPException(404,'No saved acquisition')
    return json.loads(p.read_text(encoding='utf-8'))

@app.get('/api/history')
def history():
    entries=[]
    for f in (ROOT/'runs').glob('*/summary.json'):
        try:
            r=json.loads(f.read_text(encoding='utf-8'));entries.append({**{k:r[k] for k in ('job_id','protocol','metrics','metadata','created','preview','adc_samples')},'graph_archived':(f.parent/'graph_full.npz').is_file()})
        except (ValueError,KeyError,OSError):continue
    return sorted(entries,key=lambda x:x['created'],reverse=True)[:100]

@app.post('/api/jobs/{i}/archive-graph')
def archive_graph(i:str):
    with lock:
        j=analysis_ready(i)
        if j['result']['preview']:raise HTTPException(422,'Native graph required')
        g=runner.engine.graph;f=run_dir(i)/'graph_full.npz';tmp=f.with_suffix('.tmp.npz')
        # Explicit action: full edge arrays can be hundreds of MB for long acquisitions.
        np.savez_compressed(tmp,**{k:np.asarray(getattr(g,k)) for k in ('offsets','kind','kt_end','kt_start','node_mask','emitted','latent','rows','src','rel','tap','mask','case_species')});tmp.replace(f)
        return {'bytes':f.stat().st_size}

@app.delete('/api/jobs/{i}/saved')
def delete_saved(i:str):
    import shutil
    with lock:
        if active==i and jobs.get(i,{}).get('status') in ('queued','running'):raise HTTPException(409,'Acquisition is running')
        if not (run_dir(i)/'summary.json').is_file():raise HTTPException(404,'No saved acquisition')
        shutil.rmtree(run_dir(i));jobs.pop(i,None)
    return {'deleted':True}

@app.get('/api/jobs/{i}/saved')
def saved(i:str):return stored(i)

@app.get('/api/jobs/{i}/artifact/{kind}')
def artifact(i:str,kind:str):
    names={'maps':'maps.npz','graph-nodes':'graph_nodes.npz','graph-full':'graph_full.npz','design':'sequence_design.npz'}
    if kind not in names:raise HTTPException(404,'Unknown artifact')
    f=run_dir(i)/names[kind]
    if not f.is_file():raise HTTPException(404,'Artifact unavailable')
    return FileResponse(f,filename=f'mrtini-{i[:8]}-{names[kind]}')

@app.get('/api/jobs/{i}/maps')
def map_view(i:str,key:str='PD'):
    f=run_dir(i)/'maps.npz'
    if not f.is_file():raise HTTPException(404,'No phantom maps available')
    with np.load(f) as d:
        if key not in d or key.endswith('_index') or key in ('x_m','y_m'):raise HTTPException(422,'Map unavailable')
        a=d[key];finite=a[np.isfinite(a)];lo=float(np.min(finite)) if finite.size else 0.;hi=float(np.max(finite)) if finite.size else 1.
        # JSON null for masked background, not a fabricated zero parameter.
        return {'key':key,'values':np.where(np.isfinite(a),a,None).tolist(),'x_mm':(d['x_m']*1000).tolist(),'y_mm':(d['y_m']*1000).tolist(),'min':lo,'max':hi,'unit':MAP_UNITS.get(key,MAP_UNITS.get(key.removeprefix('water_').removeprefix('fat_'),'a.u.'))}

@app.get('/api/jobs/{i}/voxel')
def voxel(i:str,x:int,y:int):
    f=run_dir(i)/'maps.npz'
    if not f.is_file():raise HTTPException(404,'No phantom maps available')
    with np.load(f) as d:
        if not (0<=x<len(d['x_m']) and 0<=y<len(d['y_m'])):raise HTTPException(422,'Voxel index out of range')
        vals={k:float(d[k][y,x]) if np.isfinite(d[k][y,x]) else None for k in d.files if k not in ('x_m','y_m')}
        return {'x_mm':float(d['x_m'][x]*1000),'y_mm':float(d['y_m'][y]*1000),'values':vals}

@app.get('/api/jobs/{i}/analysis-default')
def analysis_default(i:str):
    f=run_dir(i)/'maps.npz'
    if not f.is_file():raise HTTPException(404,'No phantom maps available')
    with np.load(f) as d:
        choices=[]
        xx,yy=np.meshgrid(d['x_m'],d['y_m']);dist=xx**2+yy**2
        for species,name in enumerate(('water','fat')):
            if name+'_index' not in d:continue
            pd=d[name+'_PD'];valid=(d[name+'_index']>=0)&np.isfinite(pd)&(pd>0)
            if valid.any():
                # Prefer central high-PD tissue, not a nearly empty partial-volume entry.
                strong=valid&(pd>=.1*np.max(pd[valid]));y,x=np.unravel_index(np.argmin(np.where(strong,dist,np.inf)),pd.shape);choices.append((species,int(x),int(y)))
        if not choices:raise HTTPException(422,'No active native voxel available')
    species,x,y=choices[0]
    with lock:
        analysis_ready(i)
        steps=runner.engine.steps
        # Skip dummy/preparation RF layers without ADC; choose central readout sample.
        rep=next((j for j,s in enumerate(steps) if len(s.adc)),0)
        adc_count=len(steps[rep].adc)
        adc=adc_count//2 if adc_count else None
    return dict(x=x,y=y,species=species,rep=rep,adc=adc,repetitions=len(steps),adc_count=adc_count)

@app.get('/api/jobs/{i}/trajectory')
def trajectory(i:str,coil:int=0):
    f=run_dir(i)/'result.npz'
    if not f.is_file():raise HTTPException(404,'No acquisition')
    with np.load(f) as d:
        k=d['kspace'];sig=d['signal'];sig=sig[:,None] if sig.ndim==1 else sig
        if not 0<=coil<sig.shape[1]:raise HTTPException(422,'Coil out of range')
        # Limit transport only; retain original index for linking and export.
        ix=np.linspace(0,len(k)-1,min(65536,len(k)),dtype=int);s=sig[ix,coil]
        times=d['adc_times_s'][ix].tolist() if 'adc_times_s' in d else None
        return dict(k=k[ix,:2].tolist(),index=ix.tolist(),real=s.real.tolist(),imag=s.imag.tolist(),time_s=times,total=len(k),coils=sig.shape[1],coil=coil,unit='cycles/m for verified generated or preview sequence; native coordinates as exported for imported sequence')

@app.get('/api/jobs/{i}/compare/{ref}')
def compare(i:str,ref:str):
    a,b=stored(ref),stored(i)
    with np.load(run_dir(ref)/'result.npz') as x,np.load(run_dir(i)/'result.npz') as y:
        c=comparison(x,y,a['protocol'],b['protocol'],a['metadata'],b['metadata']);diff=c.pop('difference');lim=c.pop('limit');r=c.pop('image_reference');q=c.pop('image_current');c.update(reference=encode(r,limit=lim),current=encode(q,limit=lim),limit=lim)
        if diff is not None:
            m=max(float(np.max(np.abs(diff))),1e-20);c.update(difference=diff.tolist(),difference_limit=m);refmag=r; threshold=max(float(np.max(refmag))*1e-6,1e-20);rel=np.divide(100*diff,refmag,out=np.full_like(diff,np.nan,dtype=float),where=refmag>threshold);finite=rel[np.isfinite(rel)];c['relative_difference']=np.where(np.isfinite(rel),rel,None).tolist();c['relative_limit']=max(float(np.max(np.abs(finite))) if finite.size else 0.,1e-20);c['relative_mask_threshold']=threshold
    ga,gb=run_dir(ref)/'graph_nodes.npz',run_dir(i)/'graph_nodes.npz'
    if ga.is_file() and gb.is_file():
        # Avoid expensive full-history matching for huge graphs; counts remain available.
        with np.load(ga) as x,np.load(gb) as y:
            if x['kind'].size+y['kind'].size<=1000000:c['node_comparison']=compare_nodes(x,y)
            else:c['node_comparison']={'basis':'Exact matching disabled above one million combined states; use layer counts.'}
    return c

@app.get('/api/jobs/{i}/rf')
def rf(i:str,rep:int=0,species:int=0,voxel:int|None=None):
    with lock:
        j=analysis_ready(i)
        if j['result']['preview']:raise HTTPException(422,'RF coefficients require native MRTini')
        try:
            out=rf_diagnostics(runner.engine,rep,voxel,species)
            if runner.design is not None:
                from analysis_views import rf_context
                out['rf_context']=rf_context(runner.design,rep)
                try:out['finite_profile']=finite_rf_profile(runner.engine,runner.design,rep,out['voxel'],3*runner.meta['slice_thickness_m'])
                except (ValueError,AttributeError,IndexError,TypeError,KeyError) as e:out['finite_profile_note']=str(e)
            out['slice_thickness_mm']=runner.meta['slice_thickness_m']*1000
            return out
        except (ValueError,AttributeError,IndexError,KeyError) as e:raise HTTPException(422,str(e))

@app.get('/api/jobs/{i}/contribution-overview')
def overview_contributions(i:str,voxel:int):
    with lock:
        j=analysis_ready(i)
        if j['result']['preview']:raise HTTPException(422,'RF-history overview requires native MRTini')
        try:return contribution_overview(runner.engine,voxel)
        except (ValueError,AttributeError,RuntimeError,IndexError,KeyError) as e:raise HTTPException(422,str(e))

@app.get('/api/jobs/{i}/voxel-replay')
def replay(i:str,voxel:int,rep:int=0,slice_profile:bool=False,adc_index:int|None=None,profile_samples:int=257,trace_paths:bool=False):
    with lock:
        j=analysis_ready(i)
        if j['result']['preview']:raise HTTPException(422,'Native voxel replay requires MRTini')
        try:return voxel_replay(runner.engine,voxel,rep,slice_extent_m=2*runner.meta['slice_thickness_m'] if slice_profile else None,adc_index=adc_index,profile_samples=profile_samples,slab_thickness_m=runner.meta['slice_thickness_m'],trace_paths=trace_paths)
        except (ValueError,AttributeError,RuntimeError,IndexError,KeyError) as e:raise HTTPException(422,str(e))

@app.post('/api/draft-strip')
def draft_strip(req:JobRequest):
    try:
        p=Protocol(**req.protocol).check()
        if p.backend=='native' and not p.native_edit:raise ValueError('Imported sequence: use completed acquisition timeline')
        return sequence_window(design=design(p),start=0,window=2,preview=p.backend=='preview')
    except (ValueError,TypeError) as e:raise HTTPException(422,str(e))

@app.get('/api/jobs/{i}/figure')
def scientific_figure(i:str,kind:str='trajectory',fmt:str='svg',rep:int=0,species:int=0,voxel:int|None=None,component:str='mxy',mode:str='equilibrium',relation:str='++',coefficient_component:str='mag',profile_samples:int=257):
    from figure_export import render_figure
    if fmt not in ('svg','pdf') or component not in ('mxy','mz'):raise HTTPException(422,'Invalid figure format or component')
    try:
        if kind=='trajectory':
            with np.load(run_dir(i)/'result.npz') as f:data=f['kspace'].copy()
        elif kind=='rf':data=rf(i,rep,species,voxel)
        elif kind=='slice':
            if voxel is None:raise ValueError('Select a voxel')
            data=replay(i,voxel,rep,True,profile_samples=profile_samples)
        else:raise ValueError('Unknown figure')
        payload=render_figure(kind,data,fmt,component,mode,relation,coefficient_component)
        return Response(payload,media_type='image/svg+xml' if fmt=='svg' else 'application/pdf',headers={'Content-Disposition':f'attachment; filename="mrtini-{kind}-{i[:8]}.{fmt}"'})
    except ImportError as e:raise HTTPException(422,'Install matplotlib to export figures: python -m pip install matplotlib') from e
    except (ValueError,FileNotFoundError,KeyError) as e:raise HTTPException(422,str(e))

class ScannerRequest(BaseModel):
    protocol: dict
    limits: dict = {}

@app.post('/api/design-check')
def design_check(req:ScannerRequest):
    from timed_design import scanner_report
    try:
        p=Protocol(**req.protocol).check();d=design(p);r=scanner_report(d,req.limits)
        return dict(**r, duration_ms=d.final*1000,adc_samples=len(d.adc_times),rf_events=len(d.pulses),trajectory=d.expected_k[::max(1,len(d.expected_k)//4000)].tolist(),strip=sequence_window(design=d,start=0,window=4,shot_end=p.tr_ms/1000 if p.sequence=='custom' else None))
    except (ValueError,TypeError,KeyError,OverflowError) as e:raise HTTPException(422,str(e))

@app.post('/api/jobs/{i}/reconstruct')
def reprocess(i:str,options:dict):
    from postprocess import process
    a=stored(i)
    try:
        with np.load(run_dir(i)/'result.npz') as f:r=process(f['signal'],f['kspace'],a['protocol'],a['metadata'],options)
        return dict(magnitude=encode(r['magnitude']),phase=encode(r['phase'],phase=True),method=r['method'],normalization=r['normalization'],matrix=r['matrix'],fov_mm=r['fov_mm'],peak=float(r['magnitude'].max()),density_compensation=None if r.get('density_weights') is None else dict(min=float(r['density_weights'].min()),max=float(r['density_weights'].max()),mean=float(r['density_weights'].mean())))
    except (ValueError,RuntimeError,TypeError,KeyError) as e:raise HTTPException(422,str(e))

@app.get('/api/jobs/{i}/dicom-export')
def dicom_export(i:str):
    from dicom_view import encode_derived
    a=stored(i)
    try:
        with np.load(run_dir(i)/'result.npz') as f:
            from research import magnitude
            payload=encode_derived(magnitude(f['complex_image']),a['protocol'],a['metadata'])
        return Response(payload,media_type='application/dicom',headers={'Content-Disposition':f'attachment; filename="mrtini-{i[:8]}.dcm"'})
    except (ValueError,ImportError) as e:raise HTTPException(422,str(e))

@app.post('/api/design')
def preview_design(req:JobRequest):
    try:
        p=Protocol(**req.protocol).check();d=design(p)
        return {'duration_ms':d.final*1000,'pulses':len(d.pulses),'samples':len(d.adc_times),'trajectory':d.expected_k[::max(1,len(d.expected_k)//8192)].tolist(),'times_ms':(d.adc_times[::max(1,len(d.expected_k)//8192)]*1000).tolist(),'strip':sequence_window(design=d,start=0,window=4,shot_end=p.tr_ms/1000 if p.sequence=='custom' else None),'rf_duration_ms':d.actual_rf_ms}
    except (ValueError,TypeError,KeyError,OverflowError) as e:raise HTTPException(422,str(e))

@app.post('/api/dicom')
async def dicom(request:Request):
    from dicom_view import decode_dicom,MAX_BYTES
    payload=bytearray()
    async for chunk in request.stream():
        if len(payload)+len(chunk)>MAX_BYTES:raise HTTPException(413,'DICOM exceeds 64 MiB limit')
        payload.extend(chunk)
    try:
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(decode_dicom,bytes(payload))
    except Exception as e:raise HTTPException(422,'DICOM could not be opened: '+str(e)) from e

app.mount('/', StaticFiles(directory=ROOT/'web',html=True),name='web')
if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--adapter');cli.add_argument('--mrtini-root');cli.add_argument('--port',type=int,default=8765);args=cli.parse_args();runner=NativeRunner(args.mrtini_root,args.adapter);__import__('os').chdir(runner.root)
    import uvicorn;uvicorn.run(app,host='127.0.0.1',port=args.port)
