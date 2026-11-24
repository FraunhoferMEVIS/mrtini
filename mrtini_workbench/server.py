from pathlib import Path
from dataclasses import asdict, fields
from concurrent.futures import ThreadPoolExecutor
import argparse, base64, io, time, uuid, threading, traceback
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from physics import Protocol, phantom, preview, reconstruct, line_plan
from native import NativeRunner

ROOT=Path(__file__).resolve().parent; app=FastAPI(title='MRtini Workbench');runner=NativeRunner();pool=ThreadPoolExecutor(max_workers=1);lock=threading.Lock();jobs={};active=None
class JobRequest(BaseModel):
    protocol: dict

def encode(x,phase=False,limit=None):
    a=np.asarray(x,float);lo=-np.pi if phase else 0;hi=np.pi if phase else (limit if limit is not None else float(np.percentile(a[a>0],99.7)) if (a>0).any() else 1)
    image=Image.fromarray(np.uint8(np.clip((a-lo)/max(hi-lo,1e-20),0,1)*255));out=io.BytesIO();image.save(out,format='PNG');return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()

def update(j,**kwargs):
    with lock:j.update(kwargs)

def run(j,p):
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
        design_export={} if reco.get('design_k') is None else {'expected_k_xy':reco['design_k'],'adc_times_s':reco['adc_times']}
        np.savez_compressed(Path(j['directory'])/'result.npz',signal=sig,kspace=k,complex_image=reco['complex_image'],**design_export,protocol_json=__import__('json').dumps(asdict(p)),metadata_json=__import__('json').dumps(meta),timings_json=__import__('json').dumps(metrics))
        # Native phantom is not inferred from synthetic preview. Adapter may provide its own display image later.
        imgs={'magnitude':encode(reco['magnitude']),'phase':encode(reco['phase'],phase=True),'kspace':encode(reco['kspace'])}
        if maps is not None:imgs['phantom']=encode(maps['PD'],limit=1);imgs['t1']=encode(maps['T1']*(maps['PD']>1e-7),limit=3.6);imgs['t2']=encode(maps['T2']*(maps['PD']>1e-7),limit=.35)
        a=np.linalg.norm(sig,axis=1);idx=np.linspace(0,len(a)-1,min(512,len(a)),dtype=int);kidx=np.linspace(0,len(k)-1,min(4000,len(k)),dtype=int)
        result=dict(images=imgs,metrics=metrics,metadata=meta,method=reco['method'],adc_samples=len(sig),signal_trace=a[idx].tolist(),signal_index=idx.tolist(),trajectory=np.asarray(k)[kidx,:2].tolist(),protocol=asdict(p),preview=p.backend=='preview',download=f'/api/jobs/{j["id"]}/result')
        update(j,status='done',stage='done',fraction=1,message='Acquisition complete',metrics=dict(metrics),result=result,elapsed=time.perf_counter()-start)
    except Exception as e:
        traceback.print_exc();update(j,status='error',stage='error',message=str(e),elapsed=time.perf_counter()-start,metrics=dict(metrics))

@app.get('/api/status')
def status():return {'native_configured':runner.configured,'mode':'native available' if runner.configured else 'preview only','version':'2.0',**runner.catalog()}
@app.post('/api/jobs')
def submit(req:JobRequest):
    global active
    try:
        p=Protocol(**req.protocol).check()
        if p.backend=='native' and not runner.configured:raise ValueError('MRtini repository not found (README).')
    except (ValueError,TypeError) as e:raise HTTPException(422,str(e))
    with lock:
        if active and jobs[active]['status'] in ('queued','running'):raise HTTPException(409,'A simulation is already running')
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
        if i not in jobs or jobs[i]['status']!='done':raise HTTPException(404,'No result')
        path=Path(jobs[i]['directory'])/'result.npz'
    return FileResponse(path,filename=f'mrtini-{i[:8]}.npz')
app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--adapter');cli.add_argument('--mrtini-root');cli.add_argument('--port',type=int,default=8765);args=cli.parse_args();runner=NativeRunner(args.mrtini_root,args.adapter);__import__('os').chdir(runner.root)
    import uvicorn;uvicorn.run(app,host='127.0.0.1',port=args.port)
