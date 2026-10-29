"""Figures from actual MRTini and isochromat outputs; no simulated plot data."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

NAVY='#183153'; TEAL='#159A9C'; CORAL='#D95D5D'; VIOLET='#7768AE'; AMBER='#E9A23B'

def _axes(n=1, width=10.2, height=4.5):
    fig, axes=plt.subplots(1,n,figsize=(width,height),facecolor='white')
    axes=np.atleast_1d(axes)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
    return fig,axes

def _save(fig,out,name):
    fig.tight_layout();fig.savefig(out/name,dpi=360,facecolor='white');plt.close(fig)

def _coil1(signal,indices):
    return np.asarray(signal)[indices].reshape(len(indices),-1)[:,0]

def _nrmse(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:raise ValueError(f'Unmatched ADC arrays: {a.shape} and {b.shape}')
    den=np.linalg.norm(b)
    if den==0:raise ValueError('Zero reference ADC signal')
    return float(np.linalg.norm(a-b)/den)

def signal_figures(name,seq,reference,result,centres,out):
    pred=result['signal'];ix=centres if len(centres)>1 else np.arange(len(reference))
    ref=_coil1(reference,ix);got=_coil1(pred,ix);x=np.arange(1,len(ix)+1)
    xlabel='Echo (centre ADC sample)' if len(centres)>1 else 'ADC sample'
    fig,ax=_axes(3,width=13.0)
    ax[0].plot(x,abs(ref),color=NAVY,label='Isochromats')
    ax[0].plot(x,abs(got),'--',color=TEAL,label='MRTini')
    ax[0].set(xlabel=xlabel,ylabel='Magnitude, coil 1',title='A  Complex echo magnitude');ax[0].legend(frameon=False)
    phase=np.angle(got*ref.conj());phase[abs(ref)<.001*max(abs(ref).max(),1e-30)]=np.nan
    ax[1].plot(x,phase,color=CORAL);ax[1].axhline(0,color='#CBD5E1',lw=.7)
    ax[1].set(xlabel=xlabel,ylabel='Phase difference (rad)',title='B  MRTini minus isochromats')
    ax[2].plot(x,abs(got-ref),color=VIOLET)
    ax[2].set(xlabel=xlabel,ylabel='|Complex difference|, coil 1',title='C  Echo residual')
    _save(fig,out,f'fig_{name}_echo_train.png')

    a=np.asarray(reference).reshape(len(reference),-1)[:,0]
    b=np.asarray(pred).reshape(len(pred),-1)[:,0]
    n=np.arange(1,len(a)+1)
    fig,ax=_axes(2,width=11)
    ax[0].plot(n,a.real,color=NAVY,lw=1,label='Iso real')
    ax[0].plot(n,b.real,'--',color=TEAL,lw=1,label='MRTini real')
    ax[0].plot(n,a.imag,color=NAVY,alpha=.45,lw=1,label='Iso imag')
    ax[0].plot(n,b.imag,'--',color=TEAL,alpha=.55,lw=1,label='MRTini imag')
    ax[0].set(xlabel='ADC sample in native acquisition order',ylabel='Complex signal, coil 1',title='A  ADC trace')
    ax[0].legend(frameon=False,ncol=2,fontsize=8)
    ax[1].plot(n,abs(b-a),color=CORAL,lw=1)
    ax[1].set(xlabel='ADC sample in native acquisition order',ylabel='|MRTini - isochromats|',title='B  Absolute complex residual')
    _save(fig,out,f'fig_{name}_adc_residual.png')

def tap_figures(name,result,sweeps,out):
    pulses=result['pulses'];shown=pulses[:min(4,len(pulses))]
    fig,ax=_axes(2,width=11.5)
    for j,p in enumerate(shown):
        order=np.asarray(p['orders']);energy=np.asarray(p['energy']);keep=np.asarray(p['kept'])
        near=(abs(order)<=max(12,int(np.max(abs(order[keep]))) if len(keep) else 12))
        label=f'RF {j+1}'
        ax[0].semilogy(order[near],energy[near]+1e-15,lw=1.2,label=label)
        ax[0].scatter(order[keep],energy[keep]+1e-15,s=11,label=f'{label} retained')
        rank=np.sort(energy)[::-1];cum=np.cumsum(rank)/max(rank.sum(),1e-30)
        ax[1].semilogx(np.arange(1,len(cum)+1),cum,lw=1.2,label=label)
    ax[0].set(xlabel='Signed RF offset Δ',ylabel='Sum of six tap energies',title='A  Actual MRTini RF taps')
    ax[1].set(xlabel='Retained offsets K (ranked)',ylabel='Cumulative energy fraction',title='B  RF support ranking')
    ax[0].legend(frameon=False,fontsize=7,ncol=2);ax[1].legend(frameon=False,fontsize=7)
    _save(fig,out,f'fig_{name}_tap_spectrum.png')

    fig,ax=_axes(2)
    eps=np.asarray([r['epsilon'] for r in sweeps])
    counts=[r['stats']['retained_taps'] for r in sweeps]
    ax[0].semilogx(eps,counts,'o-',color=TEAL)
    ax[0].invert_xaxis();ax[0].set(xlabel='Discarded-energy tolerance ε',ylabel='Sum of retained offsets across unique RFs',title='A  Actual support size')
    ax[1].loglog(eps,[max(_nrmse(r['signal'],result['signal']),1e-15) for r in sweeps],'o-',color=CORAL)
    ax[1].invert_xaxis();ax[1].set(xlabel='Discarded-energy tolerance ε',ylabel='Complex ADC NRMSE vs tightest ε',title='B  RF support sensitivity')
    _save(fig,out,f'fig_{name}_tap_sparsification.png')

def operator_figures(name,result,out):
    pulses=result['pulses'][:4]
    fig,ax=_axes(2,width=11)
    for j,p in enumerate(pulses):
        z=1e3*np.asarray(p['z_m']);mxy=np.asarray(p['mxy']);mz=np.asarray(p['mz'])
        ax[0].plot(z,abs(mxy),lw=1.25,label=f'RF {j+1}')
        ax[1].plot(z,mz,lw=1.25,label=f'RF {j+1}')
    ax[0].set(xlabel='z (mm)',ylabel='|Mx+iMy| from Z input',title='A  RF operator transverse response')
    ax[1].set(xlabel='z (mm)',ylabel='Mz from Z input',title='B  RF operator longitudinal response')
    for a in ax:a.legend(frameon=False,fontsize=8)
    _save(fig,out,f'fig_{name}_rf_operator_profiles.png')

    fig,ax=_axes(len(pulses),width=max(5.5,3.5*len(pulses)),height=4.2)
    for j,p in enumerate(pulses):
        omitted=p['omitted_by_bin']
        im=ax[j].imshow(omitted,origin='lower',aspect='auto',cmap='magma',vmin=0)
        ax[j].set(xlabel='B1 bin',ylabel='B0 bin',title=f'RF {j+1}: omitted energy')
        fig.colorbar(im,ax=ax[j],fraction=.045,pad=.04)
    _save(fig,out,f'fig_{name}_bank_omitted_energy.png')

def graph_figures(name,result,centres,out):
    g=result['graph'];rep=np.arange(1,len(g['plus'])+1)
    fig,ax=_axes(2,width=11)
    ax[0].stackplot(rep,g['plus'],g['z'],colors=[TEAL,VIOLET],labels=['F+','Z'],alpha=.82)
    ax[0].set(xlabel='RF-to-RF repetition',ylabel='Retained states',title='A  Graph growth');ax[0].legend(frameon=False)
    ax[1].plot(rep,g['edges'],color=NAVY,lw=1.3)
    ax[1].set(xlabel='RF-to-RF repetition',ylabel='Ancestry edges',title='B  RF branch complexity')
    _save(fig,out,f'fig_{name}_graph_growth.png')

    stats=result['kz_diag']['rep_state_stats'];groups=('kz0','abs1','abs2_4','high')
    bins=np.zeros((4,len(stats)))
    for rep_i,item in enumerate(stats):
        for order,record in item.items():
            n=abs(int(order));j=0 if n==0 else 1 if n==1 else 2 if n<=4 else 3
            bins[j,rep_i]+=record['count']
    fig,ax=_axes(2,width=11.2)
    ax[0].stackplot(rep,*bins,labels=['0','|n|=1','2–4','>4'],colors=[NAVY,TEAL,VIOLET,AMBER])
    ax[0].set(xlabel='RF-to-RF repetition',ylabel='States',title='A  Slice-order populations');ax[0].legend(frameon=False,fontsize=8)
    signals=result['kz_diag']['measured_groups']
    idx=centres if len(centres)>1 else np.arange(len(result['signal']))
    for key,col in zip(groups,[NAVY,TEAL,VIOLET,AMBER]):
        sig=_coil1(signals[key],idx)
        ax[1].plot(np.arange(1,len(sig)+1),abs(sig),color=col,lw=1,label=key)
    ax[1].set(xlabel='Echo (centre ADC sample)' if len(centres)>1 else 'ADC sample',
              ylabel='|Complex contribution|, coil 1',title='B  Emission by cumulative RF slice order')
    ax[1].legend(frameon=False,fontsize=8)
    _save(fig,out,f'fig_{name}_slice_order_pathways.png')

    mixed=[sum(v.get('mixed_count',0) for v in item.values()) for item in stats]
    fig,ax=_axes(2,width=11)
    ax[0].plot(np.arange(1,len(mixed)+1),mixed,'o-',color=CORAL)
    ax[0].set(xlabel='RF-to-RF repetition',ylabel='Merged states with mixed RF slice-order ancestry',
              title='A  Diagnostic order-label ambiguity')
    total=_coil1(result['signal'],idx)
    component_sum=sum(_coil1(signals[key],idx) for key in groups)
    ax[1].plot(np.arange(1,len(idx)+1),abs(total),color=NAVY,label='Full ADC')
    ax[1].plot(np.arange(1,len(idx)+1),abs(component_sum),'--',color=TEAL,
               label='Sum of complex order groups')
    ax[1].set(xlabel='Echo' if len(centres)>1 else 'ADC sample',ylabel='Magnitude, coil 1',
              title='B  Coherent pathway accounting')
    ax[1].legend(frameon=False,fontsize=8)
    _save(fig,out,f'fig_{name}_pathway_accounting.png')

def sweep_figures(name,reference,sweeps,caps,grids,out):
    fig,ax=_axes(3,width=13)
    eps=[r['epsilon'] for r in sweeps]
    ax[0].loglog(eps,[max(_nrmse(r['signal'],reference),1e-15) for r in sweeps],'o-',color=CORAL)
    ax[0].invert_xaxis();ax[0].set(xlabel='RF tolerance ε',ylabel='Complex ADC NRMSE',title='A  Pruning accuracy')
    ax[1].semilogx(eps,[r['stats']['peak_states'] for r in sweeps],'o-',color=TEAL)
    ax[1].invert_xaxis();ax[1].set(xlabel='RF tolerance ε',ylabel='Peak graph states',title='B  State cost')
    for key,col in [('rf_bank_s',AMBER),('graph_s',NAVY),('replay_s',VIOLET)]:
        ax[2].semilogx(eps,[r['stats'][key] for r in sweeps],'o-',label=key,color=col)
    ax[2].invert_xaxis();ax[2].set(xlabel='RF tolerance ε',ylabel='Elapsed (s)',title='C  Measured stage times');ax[2].legend(frameon=False,fontsize=8)
    _save(fig,out,f'fig_{name}_pruning_cost.png')

    fig,ax=_axes(2,width=11)
    for a,runs,key,title in [(ax[0],caps,'max_states','State-cap sweep'),
                              (ax[1],grids,'rf_grid_oversampling','RF z-grid sweep')]:
        if runs:
            a.plot([r[key] for r in runs],[max(_nrmse(r['signal'],reference),1e-15) for r in runs],
                   'o-',color=TEAL)
        a.set(xlabel='Maximum graph states' if key=='max_states' else 'RF grid oversampling',
              ylabel='Complex ADC NRMSE',title=title)
    _save(fig,out,f'fig_{name}_state_and_grid_sensitivity.png')

def late_echo_sensitivity(name,reference,centres,sweeps,caps,out):
    if len(centres)<2:return
    ref=_coil1(reference,centres)
    def grid(runs):
        return np.stack([abs(_coil1(r['signal'],centres)-ref) /
                          max(np.max(abs(ref)),1e-30) for r in runs])
    fig,ax=_axes(2,width=11.5,height=4.4)
    for a,runs,labels,title in (
            (ax[0],sweeps,[f'{r["epsilon"]:g}' for r in sweeps],'A  RF tap tolerance'),
            (ax[1],caps,[str(r['max_states']) for r in caps],'B  Graph state cap')):
        img=a.imshow(grid(runs),aspect='auto',origin='lower',cmap='magma',vmin=0)
        a.set(yticks=np.arange(len(runs)),yticklabels=labels,
              xlabel='Echo index',ylabel='ε' if a is ax[0] else 'Nmax',title=title)
        fig.colorbar(img,ax=a,label='|Complex residual| / maximum |reference echo|',
                     fraction=.04,pad=.02)
    _save(fig,out,f'fig_{name}_late_echo_sensitivity.png')

def hard_rf_figure(name,reference,selective,hard,centres,out):
    ix=centres if len(centres)>1 else np.arange(len(reference))
    ref=_coil1(reference,ix)
    selected=_coil1(selective['signal'],ix)
    approx=_coil1(hard['signal'],ix)
    fig,ax=_axes(2,width=11)
    n=np.arange(1,len(ix)+1)
    ax[0].plot(n,abs(ref),color=NAVY,label='Iso, actual Gz')
    ax[0].plot(n,abs(selected),'--',color=TEAL,label='MRTini, slice selective')
    ax[0].plot(n,abs(approx),':',color=CORAL,label='MRTini, Gz=0 during RF')
    ax[0].set(xlabel='Echo' if len(centres)>1 else 'ADC sample',
              ylabel='|Complex signal|, coil 1',title='A  Same sequence, RF gradient control')
    ax[0].legend(frameon=False,fontsize=8)
    ax[1].plot(n,abs(selected-ref),color=TEAL,label='Slice selective residual')
    ax[1].plot(n,abs(approx-ref),color=CORAL,label='Gz=0 RF residual')
    ax[1].set(xlabel='Echo' if len(centres)>1 else 'ADC sample',
              ylabel='|Complex residual|, coil 1',title='B  Difference from explicit z')
    ax[1].legend(frameon=False,fontsize=8)
    _save(fig,out,f'fig_{name}_hard_rf_control.png')

def min_magnitude_figure(name,reference,runs,out):
    fig,ax=_axes(2,width=10.5)
    x=[r['min_state_mag'] for r in runs]
    ax[0].loglog(x,[max(_nrmse(r['signal'],reference),1e-15) for r in runs],
                 'o-',color=CORAL)
    ax[0].set(xlabel='Minimum state magnitude',ylabel='Complex ADC NRMSE',
              title='A  Amplitude-pruning sensitivity')
    ax[1].semilogx(x,[r['stats']['peak_states'] for r in runs],'o-',color=TEAL)
    ax[1].set(xlabel='Minimum state magnitude',ylabel='Peak retained graph states',
              title='B  Pruning cost')
    _save(fig,out,f'fig_{name}_state_magnitude_pruning.png')

def cross_sequence_figure(rows,out):
    if len(rows)<2:return
    fig,ax=_axes(2,width=10.7)
    for row in rows:
        ax[0].scatter(row['seconds'],row['complex_nrmse'],color=TEAL,s=48)
        ax[0].annotate(row['sequence'],(row['seconds'],row['complex_nrmse']),xytext=(5,4),textcoords='offset points',fontsize=8)
        ax[1].scatter(row['echoes'],row['peak_states'],color=VIOLET,s=48)
        ax[1].annotate(row['sequence'],(row['echoes'],row['peak_states']),xytext=(5,4),textcoords='offset points',fontsize=8)
    ax[0].set(xlabel='MRTini total time (s)',ylabel='Complex ADC NRMSE vs supplied isochromats',title='A  Across supplied sequences')
    ax[1].set(xlabel='ADC-bearing repetitions',ylabel='Peak graph states',title='B  Sequence complexity')
    _save(fig,out,'fig_cross_sequence_summary.png')
