"""Print-oriented vector figures from actual data. Matplotlib imported lazily."""
import io
import numpy as np


def render_figure(kind, data, fmt, component='mxy', mode='equilibrium', relation='++', coefficient_component='mag'):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if fmt not in ('svg','pdf'):raise ValueError('Use SVG or PDF')
    fig=Figure(figsize=(6.6,3.8),layout='constrained',facecolor='white');FigureCanvasAgg(fig)
    ax=fig.add_subplot(111);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.set_axisbelow(True)
    if kind=='trajectory':
        k=np.asarray(data);ax.scatter(k[:,0],k[:,1],s=1.5,c=np.arange(len(k)),cmap='viridis',linewidths=0);ax.set_aspect('equal');ax.set(xlabel='kx (exported native units)',ylabel='ky (exported native units)',title='Acquired k-space trajectory')
    elif kind=='rf':
        if mode=='coefficient':
            i=data['relations'].index(relation);z=data['z_mm'];r=np.array(data['profile_real'][i]);j=np.array(data['profile_imag'][i]);v={'mag':np.hypot(r,j),'phase':np.arctan2(j,r),'real':r,'imag':j}[coefficient_component];series=[(f'{relation} · {coefficient_component}',v)];title='Retained RF Fourier transition · one period'
        elif mode=='overlay':
            d=data.get('finite_profile')
            if not d:raise ValueError('Finite waveform unavailable')
            z=np.asarray(data['z_mm']);z0=np.asarray(d['z_mm']);series=[('Retained |Mxy|',data['equilibrium_mxy']),('Retained Mz',data['equilibrium_mz']),('Rotation-only |Mxy|',np.interp(z,z0,d['mxy'],left=np.nan,right=np.nan)),('Rotation-only Mz',np.interp(z,z0,d['mz'],left=np.nan,right=np.nan))];title='Retained Fourier vs independent rotation-only response'
        elif mode=='finite':
            d=data.get('finite_profile');
            if not d:raise ValueError('Finite waveform unavailable')
            z=d['z_mm'];series=[('|Mxy| / M0',d['mxy']),('Mz / M0',d['mz'])];title='Independent rotation-only finite RF response'
        elif mode=='refocus':
            z=data['z_mm'];series=[('|R−+|',data['refocusing']),('|R++|',np.hypot(data['profile_real'][0],data['profile_imag'][0]))];title='Retained transverse transfer'
        else:
            z=data['z_mm'];series=[('|Mxy| / M0',data['equilibrium_mxy']),('Mz / M0',data['equilibrium_mz'])];title='Retained equilibrium RF response · one period'
        for (name,v),c in zip(series,['#087e85','#7951a2','#d78a27','#7892a5']):ax.plot(z,v,label=name,color=c,lw=1.4)
        ax.set(xlabel='Through-plane position (mm)',ylabel='Normalized response',title=title);
        if mode!='coefficient':ax.set_ylim(min(-1 if mode!='refocus' else 0,min(min(v) for _,v in series))-.03,max(1,max(max(v) for _,v in series))+.03)
        ax.legend(frameon=False)
    elif kind=='slice':
        d=data['slice_profiles'];a=np.asarray(d[component]);z=np.asarray(d['z_mm']);bound=max(1,float(np.abs(a).max()));ax.clear();im=ax.imshow(a,aspect='auto',origin='upper',interpolation='nearest',extent=(z[0],z[-1],len(a)-.5,-.5),cmap='RdBu_r' if component=='mz' else 'viridis',vmin=-bound if component=='mz' else 0,vmax=bound);ix=np.unique(np.linspace(0,len(a)-1,min(6,len(a)),dtype=int));ax.set_yticks(ix,[f"{d['rep'][i]} · {d['stage'][i].split(' ')[0]}" for i in ix]);ax.set(xlabel='Through-plane offset (mm)',ylabel='Sampled RF stage',title='Retained PDG slice evolution');fig.colorbar(im,ax=ax,label='Mz / M0' if component=='mz' else '|Mxy| / M0')
    else:raise ValueError('Unsupported figure')
    out=io.BytesIO();fig.savefig(out,format=fmt,dpi=300);return out.getvalue()
