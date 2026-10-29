"""Place beside validation.py/result_cache.py; run python full_publication.py.

Reads output/raw/*/*/plot_inputs.json. No simulation or MRTini imports.
Uses ../chapter_03/figure_theme.py and figure_views.sequence_strip.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import hashlib
import json
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
CHAPTER3 = HERE.parent / 'chapter_03'
plt.rcParams.update({
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.labelsize': 11,
})

def setup(chapter3):
    sys.path.insert(0, str(chapter3))
    global theme, sequence_strip
    import figure_theme as theme
    from figure_views import sequence_strip
    theme.set_style()


def ratio(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a-b)/den) if den else float('nan')


def read_record(filename):
    from result_cache import load_result
    meta = json.loads(filename.read_text(encoding='utf-8'))
    folder = filename.parent
    ref = load_result(folder/'reference.npz')
    acquisition = load_result(folder/'acquisition.npz')
    fields = ('signal', 'stats', 'epsilon', 'max_states', 'min_state_mag',
              'hard_rf', 'cache_options')
    runs = {}
    for key in ('primary', 'sweeps', 'caps', 'magnitudes', 'hard'):
        names = meta.get(key, [])
        for name in names if isinstance(names, list) else [names]:
            if name and name not in runs:
                full = load_result(folder/name)
                runs[name] = {k: full[k] for k in fields if k in full}
    counts = np.asarray(meta['adc_counts'], dtype=int)
    offsets = np.r_[0, np.cumsum(counts[:-1])]
    reps = np.flatnonzero(counts > 0)
    ids = (offsets+counts//2)[reps]
    group = -1
    groups = []
    for count in counts:
        if count == 0:
            group += 1
        else:
            groups.append(max(group, 0))
    if len(ref['signal']) != counts.sum():
        raise ValueError(f'{meta["name"]}: ADC counts and reference differ')
    for run in runs.values():
        if run['signal'].shape != ref['signal'].shape:
            raise ValueError(f'{meta["name"]}: unmatched signal arrays')
    return dict(meta=meta, folder=folder, acquisition=acquisition, ref=ref,
                runs=runs, counts=counts, offsets=offsets, reps=reps,
                ids=ids, groups=np.asarray(groups), strip_source=None)


def metrics(r, run):
    a, b = run['signal'], r['ref']['signal']
    ae, be = a[r['ids'], 0], b[r['ids'], 0]
    mask = (abs(be) > 1e-3*abs(be).max()) & (abs(ae) > 1e-3*abs(be).max())
    phi = np.angle(ae[mask]*be[mask].conj())
    phase = np.degrees(np.sqrt(np.average(phi**2, weights=abs(be[mask])**2))) if mask.any() else np.nan
    energy = np.sum(abs(a-b)**2, axis=1)
    cut = int(r['counts'][r['reps'][:8]].sum())
    return dict(complex_adc_nrmse=ratio(a, b), magnitude_adc_nrmse=ratio(abs(a), abs(b)),
                centre_nrmse=ratio(a[r['ids']], b[r['ids']]), phase_rms_deg=float(phase),
                first8_squared_error_fraction=float(energy[:cut].sum()/energy.sum()) if energy.sum() else 0.,
                after8_adc_nrmse=ratio(a[cut:], b[cut:]))


class Envelope:
    def __init__(self, t, values):
        self.t, self.values = np.asarray(t), np.asarray(values, dtype=complex)
        self.start, self.end, self.label = float(t[0]), float(t[-1]), ''

    def sample(self, t):
        i = np.clip(np.searchsorted(self.t, t, side='right')-1, 0, len(self.values)-1)
        return np.where((t >= self.start) & (t < self.end), self.values[i], 0j)


def source_strip(path, readouts, counts):
    """Use MRTini's actual GammaSTAR parser; adapt its waveforms for the renderer."""
    import mrstar
    parser = mrstar.load_gstar(str(path))
    def intervals(channel, limit=None):
        result=[];time=0.
        while limit is None or len(result)<limit:
            event=parser.encounter(channel,time+1e-12)
            if event is None:break
            result.append((float(event[0]),float(event[1])));time=float(event[1])
        return result
    samples=np.asarray(parser.events('adc',0.,parser.duration()),dtype=float)
    if len(samples)!=int(np.sum(counts)):
        raise ValueError('Source parser ADC count differs from the simulated acquisition')
    windows=[];cursors=[];offset=0
    for count in np.asarray(counts)[np.asarray(counts)>0][:readouts]:
        times=samples[offset:offset+count];offset+=count
        dwell=float(np.median(np.diff(times))) if count>1 else 0.
        windows.append((float(times[0]),float(times[-1]+dwell)))
        cursors.append(float(times[count//2]))
    if not windows:raise ValueError('No ADC windows in the parsed sequence')
    end=windows[-1][1]
    rw=parser.raw_waveforms(0.,end)
    time=np.asarray(rw['t'],dtype=float)
    g=np.column_stack([np.asarray(rw.get('g'+axis,np.zeros_like(time)),dtype=float)
                       for axis in 'xyz'])/42.577478e6
    rf=[]
    for lo,hi in intervals('rf'):
        if lo>end:break
        pulse=parser.raw_waveforms(lo,hi)
        t=np.asarray(pulse['t'],dtype=float)
        values=np.asarray(pulse['rf_amp'])*np.exp(1j*np.asarray(pulse.get('rf_phase',0.)))
        rf.append(Envelope(t,values))
    return SimpleNamespace(t=time,grad_Tpm=g,source=SimpleNamespace(
        rf_events=rf,hard_rf_events=[],adc_events=[SimpleNamespace(start=a,end=b) for a,b in windows]),
        markers={}),cursors


def draw_strip(ax, r, sequence_dir, readouts):
    context = json.loads((r['folder']/'context.json').read_text(encoding='utf-8'))
    original = context['sequence_file']
    basename = original.replace('\\', '/').split('/')[-1]
    candidates = [Path(original), sequence_dir/basename]
    source = next((p for p in candidates if p.is_file()), None)
    compiled=None
    if source:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != context['sequence_sha256']:
            raise ValueError(f'{source}: sequence differs from the saved simulation')
        try:
            compiled,cursors=source_strip(source,readouts,r['counts'])
        except ModuleNotFoundError as error:
            if error.name!='mrstar':raise
    if compiled is not None:
        sequence_strip(ax, compiled, show_adc=True, label_events=False)
        # for j, t in enumerate(cursors):
        #     ax.axvline(t*1e3, ls='--', color=theme.CORAL, lw=.65, alpha=.8)
        #     ax.text(t*1e3, 1.02, str(j+1), ha='center', fontsize=6, color=theme.CORAL)
        r['strip_source'] = str(source)
        # note = f'Source waveforms: first {len(cursors)} readouts'
    else:
        windows = [r['acquisition']['native'][i]['time_s'] for i in r['reps'][:readouts]]
        end = float(windows[-1][-1])*1e3
        ax.set(xlim=(0, end*1.025), ylim=(0, 1)); ax.axis('off')
        ax.plot([0,end],[.3,.3], color='#D8DEE8', lw=.7)
        for j, t in enumerate(windows):
            x = np.linspace(float(t[0]), float(t[-1]), 80)*1e3
            ax.plot(x,.3+.10*np.cos(np.linspace(-5*np.pi,5*np.pi,80)),color=theme.TEAL,lw=.8)
            centre = float(t[len(t)//2])*1e3
            ax.axvline(centre, ymin=.1, ymax=.8, color=theme.CORAL, ls='--', lw=.65)
            ax.text(centre,.88,str(j+1),ha='center',fontsize=6,color=theme.CORAL)
        ax.text(1,0,f'{end:.1f} ms',transform=ax.transAxes,ha='right',fontsize=6,color=theme.FG3)
        note = 'Saved ADC timing; RF/gradient waveforms require source JSON and mrstar'
        print(f'{r["meta"]["name"]}: {note}', flush=True)
    # ax.text(0,1.20,note,transform=ax.transAxes,fontsize=6.6,color=theme.FG3)


def primary(r):
    return r['runs'][r['meta']['primary']]


def title(r):
    return 'RARE' if 'rare' in r['meta']['name'].lower() else 'FLASH'


def settings_note(r):
    run = primary(r)
    return (f'{title(r)}: ε={run["epsilon"]:g}, cap={run["max_states"]}/family, '
            f'minimum magnitude={run["min_state_mag"]:g}')


def finish(fig, name, records, out, details):
    fig.savefig(out/name, dpi=360, facecolor='white'); plt.close(fig)
    metadata = dict(contexts=[str(r['folder']) for r in records],
                    references=[str(r['folder']/'reference.npz') for r in records],
                    primary_files=[str(r['folder']/r['meta']['primary']) for r in records],
                    sequence_strip_sources=[r['strip_source'] for r in records], **details)
    (out/Path(name).with_suffix('.json')).write_text(json.dumps(metadata,indent=2))
    print(f'Saved {out/name}',flush=True)


def boundaries(ax, r):
    for x in np.flatnonzero(np.diff(r['groups']))+1.5:
        ax.axvline(x,color='#D8DEE8',lw=.55,zorder=0)


def fidelity(records, out, sequence_dir, readouts, *, width=18., height=15.,
             height_ratios=(1.65,1.25,1.2)):
    for r in records:
        fig = theme.new_figure(width=width,height=height)
        gs = fig.add_gridspec(3,2,height_ratios=height_ratios,
                             left=.085,right=.975,bottom=.13,top=.93,hspace=.60,wspace=.28)
        strip = fig.add_subplot(gs[0,:]); draw_strip(strip,r,sequence_dir,readouts)
        # theme.panel(strip,'A','Sequence waveforms')
        fig.text(.085,.965,f'{title(r)} signal fidelity',color=theme.NAVY,weight='bold',fontsize=13)
        run = primary(r); met = metrics(r,run)
        a,b = run['signal'][:,0],r['ref']['signal'][:,0]
        hard_file = r['meta'].get('hard')
        hard = r['runs'][hard_file] if hard_file else None
        if hard is not None and not hard.get('hard_rf'):
            raise ValueError(f'{r["meta"]["name"]}: hard control is not marked hard_rf')
        if hard is None:
            warnings.warn(f'{r["meta"]["name"]}: no saved hard control; order-local PDG omitted',
                          stacklevel=2)
        hard_signal = hard['signal'][:,0] if hard is not None else None
        hard_metrics = metrics(r,hard) if hard is not None else None
        x = np.arange(1,len(r['ids'])+1); ae,be = a[r['ids']],b[r['ids']]
        ax = fig.add_subplot(gs[1,0]);theme.panel(ax,'B','Readout-centre magnitude')
        ax.plot(x,abs(be),color=theme.NAVY,lw=1.45)
        ax.plot(x,abs(ae),color=theme.TEAL,lw=1.15,ls='--')
        if hard_signal is not None:
            hard_centres = hard_signal[r['ids']]
            ax.plot(x,abs(hard_centres),color=theme.CORAL,lw=1.2,ls=':')
        ax.set(xlabel='Acquired readout',ylabel=r'$|S|$ (coil 1)',xlim=(1,len(x)));boundaries(ax,r);theme.clean(ax)
        ax.text(.98,.96,f'MRTini all ADC: {100*met["complex_adc_nrmse"]:.2f}%\nCentres: {100*met["centre_nrmse"]:.2f}%',
                transform=ax.transAxes,ha='right',va='top',fontsize=7,color=theme.TEAL)
        if hard_metrics is not None:
            ax.text(.98,.79,
                    f'Order-local PDG all ADC: {100*hard_metrics["complex_adc_nrmse"]:.2f}%\n'
                    f'Centres: {100*hard_metrics["centre_nrmse"]:.2f}%',
                    transform=ax.transAxes,ha='right',va='top',fontsize=7,color=theme.CORAL)
        ax = fig.add_subplot(gs[1,1]);theme.panel(ax,'C','Readout-centre phase')
        mask = (abs(be)>1e-3*abs(be).max()) & (abs(ae)>1e-3*abs(be).max())
        p = np.unwrap(np.angle(be)); q = p+np.angle(ae*be.conj())
        ax.plot(x,np.where(mask,np.degrees(p),np.nan),color=theme.NAVY,lw=1.3)
        ax.plot(x,np.where(mask,np.degrees(q),np.nan),color=theme.TEAL,lw=1.0,ls='--')
        if hard_signal is not None:
            hard_mask = (abs(be)>1e-3*abs(be).max()) & (abs(hard_centres)>1e-3*abs(be).max())
            hard_phase = p+np.angle(hard_centres*be.conj())
            ax.plot(x,np.where(hard_mask,np.degrees(hard_phase),np.nan),
                    color=theme.CORAL,lw=1.2,ls=':')
        ax.set(xlabel='Acquired readout',ylabel='Phase (degrees)',xlim=(1,len(x)));boundaries(ax,r);theme.clean(ax)
        chosen = sorted(set((1, 4, min(7, len(x)-1), len(x)-1)))
        windows = gs[2, :].subgridspec(1, len(chosen), wspace=.62)
        window_errors = []
        hard_window_errors = []
        for window_index, readout_index in enumerate(chosen):
            window_axis = fig.add_subplot(windows[0, window_index])
            theme.panel(window_axis, chr(68+window_index), f'Complex ADC · R{readout_index+1}')
            repetition = r['reps'][readout_index]
            start = r['offsets'][repetition]
            samples = slice(start, start+r['counts'][repetition])
            window_axis.plot(b[samples].real, b[samples].imag, color=theme.NAVY, lw=1.25, alpha=.8)
            window_axis.plot(a[samples].real, a[samples].imag, color=theme.TEAL, lw=1., ls='--')
            if hard_signal is not None:
                window_axis.plot(hard_signal[samples].real, hard_signal[samples].imag,
                                 color=theme.CORAL, lw=1.2, ls=':')
                hard_centre = hard_signal[r['ids'][readout_index]]
                window_axis.scatter(hard_centre.real, hard_centre.imag, s=21,
                                    marker='x', color=theme.CORAL, zorder=5)
                hard_window_errors.append(ratio(hard_signal[samples], b[samples]))
            reference_centre = b[r['ids'][readout_index]]
            model_centre = a[r['ids'][readout_index]]
            window_axis.scatter(reference_centre.real, reference_centre.imag, s=15, color=theme.NAVY, zorder=5)
            window_axis.scatter(model_centre.real, model_centre.imag, s=21, facecolors='none', edgecolors=theme.TEAL, zorder=5)
            window_axis.margins(x=.12, y=.16)
            window_axis.set(xlabel=r'Re $S$', ylabel=r'Im $S$')
            theme.clean(window_axis)
            window_axis.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            window_axis.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            window_errors.append(ratio(a[samples], b[samples]))
            error_note = f'MRTini: {100*window_errors[-1]:.2f}%'
            if hard_signal is not None:
                error_note += f'\nOrder-local: {100*hard_window_errors[-1]:.2f}%'
            window_axis.text(.02, -.48, error_note,
                             transform=window_axis.transAxes, ha='left', va='top', fontsize=6.3, color=theme.FG3)
        legend_handles = [Line2D([], [], color=theme.NAVY, label='Numerical isochromats'),
                          Line2D([], [], color=theme.TEAL, ls='--', label='MRTini')]
        if hard_signal is not None:
            legend_handles.append(Line2D([], [], color=theme.CORAL, ls=':',
                                         label='Order-local PDG'))
        fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(.98, .945),
                   ncol=len(legend_handles), frameon=False, fontsize=7.5)
        # fig.text(.085, .075, settings_note(r), fontsize=7, color=theme.FG3)
        boundary_note = ' Grey boundaries identify separate RARE trains.' if title(r)=='RARE' else ''
        # footer = ('All readouts remain included.'+boundary_note+' No gain or phase fit.\n'
        #           'Complex windows use independent component scales; filled/open markers identify acquired centres (coil 1).')
        # fig.text(.085, .035, footer, fontsize=6.7, color=theme.FG3)
        details = dict(selected_windows=[int(index+1) for index in chosen], phase_mask_fraction=.001,
                       metrics=met, window_adc_nrmse=window_errors,
                       order_local_pdg=(dict(file=str(r['folder']/hard_file), metrics=hard_metrics,
                                             window_adc_nrmse=hard_window_errors,
                                             cache_options=hard.get('cache_options'),
                                             label='Order-local PDG (hard RF)',
                                             color=theme.CORAL, linestyle=':')
                                        if hard is not None else None),
                       layout=dict(width=width, height=height, height_ratios=height_ratios))
        finish(fig, f'fig_full_signal_fidelity_{r["meta"]["name"]}.png', [r], out, details)


def sensitivity(records, out):
    fig = theme.new_figure(width=18.,height=10.3)
    gs = fig.add_gridspec(len(records),3,left=.085,right=.98,bottom=.09,top=.88,wspace=.43,hspace=.72)
    for row,r in enumerate(records):
        ref = r['ref']['signal']; designated = primary(r)
        for col,(family,key,label) in enumerate((('sweeps','epsilon',r'RF tolerance $\epsilon$'),
                                                 ('caps','max_states','Cap per state family'),
                                                 ('magnitudes','min_state_mag','Minimum state magnitude'))):
            runs = sorted([r['runs'][f] for f in r['meta'].get(family,[])],key=lambda x:x[key])
            ax = fig.add_subplot(gs[row,col]);theme.panel(ax,chr(65+row*3+col),f'{title(r)} · {label}')
            x = [run[key] for run in runs]
            y = [100*ratio(run['signal'],ref) for run in runs]
            change = [100*np.linalg.norm(run['signal']-designated['signal'])/np.linalg.norm(ref) for run in runs]
            ax.plot(x,y,'o-',color=theme.TEAL,lw=1.35,ms=3)
            ax.plot(x,change,'s--',color=theme.VIOLET,lw=1.,ms=2.7)
            ax.axvline(designated[key],color=theme.FG3,lw=.75,ls=':')
            if key != 'max_states':ax.set_xscale('log');ax.invert_xaxis()
            ax.set(xlabel=label,ylabel='Relative difference (%)',ylim=(0,max(y+change+[1])*1.14));theme.clean(ax)
            
            actual = [run['stats']['retained_taps' if col==0 else 'peak_states'] for run in runs]
            prefix = 'Total RF offsets' if col==0 else 'Peak stored states'
            ax.text(.5,.5,f'{prefix}: {min(actual)}–{max(actual)}',transform=ax.transAxes,fontsize=6.3,color=theme.FG3)
            bottom, top = ax.get_ylim()
            padding = 0.05 * (top - bottom)
            ax.set_ylim(min(bottom, -padding), top)
    fig.legend(handles=[Line2D([],[],color=theme.TEAL,marker='o',label='Full ADC error against isochromats'),
                        Line2D([],[],color=theme.VIOLET,marker='s',ls='--',label='Change from designated MRTini / Bloch norm')],
               loc='upper center',bbox_to_anchor=(.55,.98),ncol=2,frameon=False,fontsize=7)
    # fig.text(.085,.055,'One control varies within each panel; all other controls remain fixed. Vertical dotted line: saved designated setting.\n'
    #          'All native samples/coils are included. A small change within MRTini does not establish small error against the reference.',
    #          fontsize=6.7,color=theme.FG3)
    finish(fig,'fig_full_numerical_sensitivity.png',records,out,dict(within_model_normalization='Bloch full-signal norm'))


def residual_structure(records, out):
    matrices=[]
    for r in records:
        b = r['ref']['signal'][:,0];a = primary(r)['signal'][:,0]
        widths = r['counts'][r['reps']]
        if len(set(widths)) != 1:
            raise ValueError('Residual atlas needs equal ADC sample counts per window')
        matrices.append(100*abs(a-b).reshape(len(widths),int(widths[0]))/abs(b).max())
    vmax = max(float(x.max()) for x in matrices)
    cmap = LinearSegmentedColormap.from_list('residual',['#F7F8FA','#F2D3A2','#D95D5D','#6C365C'])
    cmap.set_bad('white');cmap.set_under('#F7F8FA')
    fig = theme.new_figure(width=18.,height=11.)
    gs = fig.add_gridspec(2,len(records),height_ratios=[1.7,1],left=.09,right=.91,bottom=.18,top=.89,wspace=.4,hspace=.53)
    image = None
    for col,(r,matrix) in enumerate(zip(records,matrices)):
        ax = fig.add_subplot(gs[0,col]);theme.panel(ax,chr(65+col),f'{title(r)} · full ADC residual')
        nread,nsample = matrix.shape
        image = ax.imshow(np.ma.masked_less_equal(matrix,0),origin='upper',aspect='auto',
            extent=(-nsample//2-.5,nsample-nsample//2-.5,nread+.5,.5),interpolation='nearest',
            cmap=cmap,norm=LogNorm(vmin=.001,vmax=max(vmax,.01)))
        for y in np.flatnonzero(np.diff(r['groups']))+1.5:ax.axhline(y,color='white',lw=.45,alpha=.7)
        ax.set(xlabel='Sample offset from selected ADC centre',ylabel='Acquired readout')
        ax.set_yticks([1,32,64,96,nread] if nread>=96 else [1,nread]);theme.clean(ax)
        ax = fig.add_subplot(gs[1,col]);theme.panel(ax,chr(67+col),'Where squared error accumulates')
        a,b = primary(r)['signal'],r['ref']['signal'];energy=np.sum(abs(a-b)**2,axis=1)
        per_read = np.array([energy[o:o+c].sum() for o,c in zip(r['offsets'],r['counts']) if c])
        cum=100*np.cumsum(per_read)/max(float(per_read.sum()),1e-30);x=np.arange(1,len(cum)+1)
        ax.plot(x,cum,color=theme.CORAL,lw=1.6);ax.axvspan(.5,8.5,color=theme.AMBER,alpha=.1)
        ax.scatter(8,cum[7],color=theme.CORAL,s=18,zorder=4)
        ax.text(.98,.08,f'First 8 readouts: {cum[7]:.1f}%\nof squared residual',transform=ax.transAxes,
                ha='right',va='bottom',fontsize=7,color=theme.FG3)
        ax.set(xlabel='Acquired readout',ylabel='Cumulative squared error (%)',ylim=(0,105),xlim=(1,len(x)));theme.clean(ax)
    cb = fig.colorbar(image,cax=fig.add_axes([.935,.47,.013,.35]))
    cb.set_label(r'$100|S_{\rm MRTini}-S_{\rm Bloch}|/\max|S_{\rm Bloch}|$',fontsize=7)
    cb.ax.tick_params(labelsize=8)
    fig.text(.09,.075,'   ·   '.join(settings_note(r) for r in records),fontsize=6.5,color=theme.FG3)
    # fig.text(.09,.035,'The atlas uses one common colour scale and each sequence’s reference peak; colour is logarithmic.\n'
    #          'Exact zeros are white; values below 0.001% use the lowest colour. Every readout and ADC sample remains included.',
    #          color=theme.FG3,fontsize=6.6)
    finish(fig,'fig_full_residual_structure.png',records,out,dict(colour_floor_percent=.001,normalization='sequence reference peak, coil 1'))


def export(records,out):
    rows=[];summary=[]
    for r in records:
        for filename,run in r['runs'].items():
            row=dict(sequence=r['meta']['name'],context=r['folder'].name,file=filename,
                     designated=filename==r['meta']['primary'],epsilon=run['epsilon'],
                     cap_per_family=run['max_states'],min_state_mag=run['min_state_mag'],
                     hard_rf=run.get('hard_rf',False),**run['stats'],**metrics(r,run))
            rows.append(row)
            if row['designated']:summary.append(row)
    for name,data in [('all_run_metrics.csv',rows),('designated_metrics.csv',summary)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    settings = []
    for r in records:
        settings.append(dict(sequence=r['meta']['name'],context=r['folder'].name,
                             phantom={k:np.asarray(v).tolist() for k,v in r['acquisition']['phantom'].items()},
                             primary_options=primary(r).get('cache_options'),
                             reference_seconds=r['ref'].get('seconds'),
                             rf_repetitions=len(r['counts']),readouts=len(r['ids']),adc_samples=int(r['counts'].sum())))
    (out/'effective_settings.json').write_text(json.dumps(settings,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=HERE/'output/raw')
    parser.add_argument('--out',type=Path,default=HERE/'output/publication')
    parser.add_argument('--chapter3',type=Path,default=CHAPTER3)
    parser.add_argument('--sequence-dir',type=Path,default=HERE.parent/'common/sequences')
    parser.add_argument('--name',action='append',help='Sequence name; repeat for multiple names')
    parser.add_argument('--context',help='Exact context hash; otherwise select the latest plot_inputs per sequence')
    parser.add_argument('--strip-readouts',type=int,default=8)
    args=parser.parse_args();setup(args.chapter3)
    found={}
    for filename in args.raw.rglob('plot_inputs.json'):
        meta=json.loads(filename.read_text(encoding='utf-8'));name=meta['name']
        if args.name and name not in args.name:continue
        if args.context and filename.parent.name != args.context:continue
        if name not in found or filename.stat().st_mtime_ns > found[name].stat().st_mtime_ns:found[name]=filename
    if not found:raise SystemExit('No matching plot_inputs.json; check --raw/--context')
    records=[]
    for key in sorted(found):
        print(f'{key}: context {found[key].parent.name}',flush=True)
        records.append(read_record(found[key]))
    args.out.mkdir(parents=True,exist_ok=True)
    export(records,args.out)
    fidelity(records,args.out,args.sequence_dir,args.strip_readouts)
    sensitivity(records,args.out)
    residual_structure(records,args.out)


if __name__=='__main__':
    main()
