"""Explicit timed research events. SI internally; gradient amplitudes Hz/m.
No scanner deployment or safety qualification. RF-centre PDG convention.
"""
import numpy as np
from composer import number
from native_sequences import Design, RASTER


def expand(events, offset=0., depth=0):
    if depth>4:raise ValueError('Nested repeat depth exceeds four')
    if not isinstance(events,list):raise ValueError('Events must be a list')
    out=[]
    for e in events:
        if not isinstance(e,dict):raise ValueError('Invalid event')
        start=number(e,'start_ms',0,0,10000)+offset
        if e.get('channel')=='repeat':
            count=number(e,'count',1,1,512);period=number(e,'period_ms',1,.001,10000)
            if int(count)!=count:raise ValueError('Repeat count must be integer')
            for i in range(int(count)):
                out.extend(expand(e.get('events'),start+i*period,depth+1))
                if len(out)>4096:raise ValueError('Too many expanded events')
        else:out.append(dict(e,start_ms=start))
        if len(out)>4096:raise ValueError('Too many expanded events')
    return out


def compile_timed(p, recipe):
    events=expand(recipe.get('events'));tr=p.tr_ms/1000;shots=number(recipe,'shots',1,1,512)
    if shots!=int(shots) or shots*len(events)>16000:raise ValueError('Invalid shot count or event budget')
    shots=int(shots);phase_step=number(recipe,'phase_step_deg',0,-360,360);quad=recipe.get('quadratic_phase',False)
    local=[]
    for e in events:
        channel=e.get('channel');a=number(e,'start_ms',0,0,10000)*.001
        if channel=='adc':
            n=number(e,'samples',64,1,8192);dwell=number(e,'dwell_us',12,1,10000)*1e-6
            if int(n)!=n:raise ValueError('ADC samples must be integer')
            b=a+n*dwell
        else:b=a+number(e,'duration_ms',1,.01,10000)*.001
        if channel not in ('rf','gx','gy','gz','adc'):raise ValueError('Unknown timed channel')
        if b>tr+1e-10:raise ValueError(f'{channel.upper()} extends beyond TR')
        local.append((a,b,e))
    for channel in ('rf','adc','gx','gy','gz'):
        intervals=sorted((a,b) for a,b,e in local if e['channel']==channel)
        if any(b>c+1e-10 for (a,b),(c,d) in zip(intervals,intervals[1:])):raise ValueError(f'Overlapping {channel.upper()} events')
    rf=sorted((a,b,e) for a,b,e in local if e['channel']=='rf');adcs=sorted((a,b,e) for a,b,e in local if e['channel']=='adc')
    if not rf or not adcs or rf[0][2].get('usage','excitation')!='excitation':raise ValueError('Use an initial excitation and at least one ADC')
    if any(a<d-1e-10 and b>c+1e-10 for a,b,_ in rf for c,d,_ in adcs):raise ValueError('ADC during RF is not supported by MRTini')
    if adcs[0][0]<rf[0][1]-1e-10:raise ValueError('ADC must follow initial excitation')
    if any(a<rf[0][0]-1e-10 for a,b,e in local if e['channel'] in ('gx','gy','gz')):raise ValueError('Pre-excitation gradients before the first RF span are not supported')
    if shots*sum(int(e.get('samples',64)) for _,_,e in adcs)>2_000_000:raise ValueError('ADC budget exceeded')
    pulses=[];grads=[];adc=[];expected=[];k=np.zeros(3);last=0.
    previous=[]
    for shot in range(shots):
        base=shot*tr;glist=[];phaseoff=np.deg2rad((phase_step*(shot*(shot+1)/2 if quad else shot))%360)
        for a,b,e in local:
            ch=e['channel']
            if ch in ('gx','gy','gz'):
                axis='xyz'.index(ch[1]);g=np.zeros(3);g[axis]=(number(e,'moment',0,-1e5,1e5)+shot*number(e,'shot_step',0,-1e5,1e5))/(b-a)
                glist.append((base+a,base+b,g));grads.append(glist[-1])
        entries=[];read=[]
        for a,b,e in rf:
            a+=base;b+=base;c=(a+b)/2;dur=b-a;cells=round(dur/RASTER)
            if cells<2 or abs(cells*RASTER-dur)>1e-9:raise ValueError('RF duration must align to 10 µs raster')
            flip=number(e,'flip_deg',90,0,360);phase=np.deg2rad(number(e,'phase_deg',0,-360,360))+phaseoff;tbw=number(e,'tbw',4,1,16);usage=e.get('usage','excitation')
            if usage not in ('excitation','refocusing'):raise ValueError('Only excitation/refocusing RF semantics supported in timed mode')
            t=a+np.arange(cells+1)*RASTER;u=(np.arange(cells)+.5)/cells-.5;env=np.sinc(tbw*u)*(.54+.46*np.cos(2*np.pi*u))
            amp=env*np.deg2rad(flip)/(2*np.pi*RASTER*env.sum());gg=np.zeros((cells+1,3))
            for lo,hi,g in glist:
                overlap=np.maximum(0,np.minimum(t[1:],hi)-np.maximum(t[:-1],lo));gg[:-1]+=overlap[:,None]/RASTER*g
                if lo<b-1e-10 and hi>a+1e-10 and any(abs((x-a)/RASTER-round((x-a)/RASTER))>1e-5 for x in (lo,hi) if a<x<b):raise ValueError('Gradient transitions inside RF must align to RF raster')
            span=dict(kind='bloch',t=t,t0=a,t1=b,rf_center=c,rf_amp=np.r_[amp,0.],rf_phase=np.full(cells+1,phase),rf_freq=np.zeros(cells+1),gx=gg[:,0],gy=gg[:,1],gz=gg[:,2],adc_active=np.zeros(cells+1),adc_phase=np.zeros(cells+1),adc_freq=np.zeros(cells+1),flip_deg=flip)
            entry=dict(centre=c,span=span,exc=usage=='excitation',phase=phase,reads=[]);entries.append(entry);pulses.append(entry)
        for a,b,e in adcs:
            for j in range(int(e.get('samples',64))):
                t=base+a+(j+.5)*number(e,'dwell_us',12,1,10000)*1e-6;read.append((t,np.deg2rad(number(e,'phase_deg',0,-360,360)),bool(e.get('track_rf_phase',False))))
        # Integrate physical gradient intervals between RF centres and ADC samples.
        integrals=previous+glist
        timeline=sorted([(r['centre'],0,r) for r in entries]+[(r[0],1,r) for r in read],key=lambda x:x[0]);entry=None
        for t,kind,r in timeline:
            k+=sum((max(0,min(t,hi)-max(last,lo))*g for lo,hi,g in integrals if hi>last and lo<t),np.zeros(3));last=t
            if kind==0:entry=r;k=np.zeros(3) if r['exc'] else -k
            else:
                if entry is None:raise ValueError('ADC before RF centre')
                entry['reads'].append((t,r[1]-(entry['phase'] if r[2] else 0)));adc.append(t);expected.append(k[:2].copy())
        previous=[x for x in glist if x[1]>last]
    return Design(pulses,grads,shots*tr,np.array(expected),np.array(adc),max((b-a)*1000 for a,b,_ in rf))


def scanner_report(d, limits):
    """Held waveform feasibility report. Not a scanner safety check."""
    gamma=42.57747892e6;edges=sorted({t for a,b,g in d.gradients for t in (a,b)})
    # sweep intervals instead of quadratic pair scan
    changes={}
    for a,b,g in d.gradients:
        changes[a]=changes.get(a,np.zeros(3))+g;changes[b]=changes.get(b,np.zeros(3))-g
    current=np.zeros(3);peak=0.;jump=0.
    for t in edges:
        delta=changes[t];current+=delta;peak=max(peak,float(np.max(np.abs(current)))*1000/gamma);jump=max(jump,float(np.max(np.abs(delta)))*1000/gamma)
    b1=max(float(np.max(np.abs(r['span']['rf_amp'])))*1e6/gamma for r in d.pulses)
    maxg=float(limits.get('gradient_mtm',40));maxrf=float(limits.get('rf_ut',20));field=float(limits.get('field_t',3))
    if not all(np.isfinite([maxg,maxrf,field])) or min(maxg,maxrf,field)<=0:raise ValueError('Scanner limits must be positive finite values')
    return dict(peak_gradient_mtm=peak,peak_b1_ut=b1,gradient_limit_mtm=maxg,rf_limit_ut=maxrf,gradient_exceeded=peak>maxg,rf_exceeded=b1>maxrf,largest_gradient_step_mtm=jump,physical_moment_cycles_m=d.area(0,d.final).tolist(),larmor_mhz=42.57747892*field,fat_separation_hz=42.57747892*field*3.4,note='Held gradients have discontinuous steps; slew, SAR, PNS and field-dependent tissue/RF calibration are not validated. Field setting is informational only.')
