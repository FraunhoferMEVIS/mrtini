# SPDX-License-Identifier: AGPL-3.0-or-later
import os,sys,time,json,contextlib
from types import SimpleNamespace as NS
import numpy as np
import torch
import mrtini_prepass_rs as rs
from src.mrtini.optimized import PackedGraph

@contextlib.contextmanager
def quiet():
    fd=os.dup(1)
    with open(os.devnull,'w') as f:
        os.dup2(f.fileno(),1)
        try:yield
        finally:os.dup2(fd,1);os.close(fd)

def fixture(reps=24,events=16,K=33):
    z=(np.arange(K)-K//2)*.010/K;U=np.array([[1,1j,0],[1,-1j,0],[0,0,np.sqrt(2)]])/np.sqrt(2)
    def bank(fat=False):
        a=(2*np.pi/3)*np.exp(-((z-(.0007 if fat else 0))/.0024)**4)
        c,s=np.cos(a),np.sin(a);R=np.zeros((K,3,3));R[:,0,0]=1;R[:,1,1]=R[:,2,2]=c;R[:,1,2]=-s;R[:,2,1]=s
        Q=U[None]@R@U.conj().T[None]
        coeff={n:Q[:,i,j] for n,i,j in [('pp',0,0),('pz',0,2),('mp',0,1),('mz',1,2),('zp',2,0),('zz',2,2)]}
        coeff={n:np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v)))/K for n,v in coeff.items()}
        taps=NS(**{n:torch.from_numpy(v.astype('complex64')) for n,v in coeff.items()},deltas=torch.arange(-(K//2),K//2+1,dtype=torch.int32),dk=100.)
        ids=torch.arange(K,dtype=torch.int32)
        return taps,{r:ids for r in ['++','+z','-+','-z','z+','zz']}
    w,ws=bank();f,fs=bank(True)
    seq=[]
    for i in range(reps):
        g=torch.zeros(events,3);g[0,0]=-250;g[1:,0]=500/(events-1);g[0,1]=float((i%8)-4)*12.5;g[-1,2]=100.
        seq.append(NS(pulse=NS(angle=2.,phase=float((i%2)*np.pi/2)),event_count=events,event_time=torch.full((events,),.0005),gradm=g,adc_usage=torch.ones(events,dtype=torch.int32)))
    return seq,torch.zeros(reps,dtype=torch.int32),w,ws,f,fs

def packed_views(p):
    n=p['nodes'];S=p['cases']
    return dict(layers=np.frombuffer(p['layers'],'<u8'),kind=np.frombuffer(p['kind'],'u1'),kt=np.frombuffer(p['kt_end'],'<f8').reshape(n,4),mag=np.frombuffer(p['nominal'],'<f4').reshape(n,S,2),mask=np.frombuffer(p['node_mask'],'u1'))

def compare(g,p,s=0):
    q=packed_views(p);errors=[];missing=0
    for ri,row in enumerate(g):
        a,b=map(int,q['layers'][ri:ri+2]);fast={ (int(q['kind'][i]),tuple(np.round(q['kt'][i],6))):q['mag'][i,s,0]+1j*q['mag'][i,s,1] for i in range(a,b) if q['mask'][i]&(1<<s)}
        for d in row:
            key=({'+' :0,'z':1,'z0':2}[d.dist_type],tuple(np.round(d.prepass_kt_vec,6)))
            if key not in fast:missing+=1;continue
            errors.append(abs(fast[key]-complex(d.prepass_mag)))
    return dict(missing=missing,max_abs_nominal_error=float(max(errors,default=0)))
