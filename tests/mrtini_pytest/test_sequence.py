from types import SimpleNamespace as NS
import pytest
import torch


def test_pulse_zero_clone_and_cpu(seqmod):
    p=seqmod.Pulse.zero();assert p.usage==seqmod.PulseUsage.UNDEF and p.selective
    assert p.device.type=='cpu' and not p.angle.any() and not p.phase.any()
    q=p.clone();q.angle[:]=1;assert not p.angle.any()
    c=p.cpu();torch.testing.assert_close(c.shim_array,p.shim_array)

@pytest.mark.parametrize('field,value,match',[
    ('event_time',torch.zeros(0),'zero elements'),('event_time',torch.zeros(1,3),'event_time'),
    ('gradm',torch.zeros(3,2),'gradm'),('adc_phase',torch.zeros(3,1),'adc_phase'),
    ('adc_usage',torch.zeros(2),'adc_usage')])
def test_repetition_validation(seqmod,field,value,match):
    kw=dict(pulse=seqmod.Pulse.zero(),event_time=torch.zeros(3),gradm=torch.zeros(3,3),adc_phase=torch.zeros(3),adc_usage=torch.zeros(3,dtype=torch.int32),raw_waveform={})
    kw[field]=value
    with pytest.raises(ValueError,match=match): seqmod.Repetition(**kw)


def test_clone_independence_and_contrasts(seqmod,repetition):
    r=repetition;q=r.clone();q.gradm[0,0]=99;assert r.gradm[0,0]==1
    assert r.get_contrasts()==[1,2];r.shift_contrasts(3);assert r.adc_usage.tolist()==[0,4,5]
    s=seqmod.Sequence([r],normalized_grads=False)
    assert s.device.type=='cpu' and s.get_duration()==pytest.approx(.06)
    assert s.get_contrasts()==[4,5]
    assert s.get_contrast_mask(4).tolist()==[True,False]
    s.shift_contrasts(1); assert s.get_contrasts()==[5,6]
    copy=s.clone();assert copy.normalized_grads is False
    copy[0].event_time[0]=0;assert s[0].event_time[0]>0
    assert s.new_rep(2) is s[-1]


def test_gradient_kspace_cumulative_and_adc_filter(seqmod,repetition):
    s=seqmod.Sequence([repetition])
    expected=torch.tensor([[1.,0.,0.,.01],[1.,2.,0.,.03],[1.,2.,3.,.06]])
    torch.testing.assert_close(s.get_full_kspace()[0],expected)
    torch.testing.assert_close(s.get_kspace(),expected[1:])

@pytest.mark.parametrize('usage,expected', [('REFOC',-1.),('EXCIT',0.),('UNDEF',1.),('FATSAT',1.)])
def test_kspace_pulse_usage(seqmod,usage,expected):
    a=seqmod.Repetition.zero(1);a.gradm[0,0]=1
    b=seqmod.Repetition.zero(1);b.pulse.usage=getattr(seqmod.PulseUsage,usage)
    assert seqmod.Sequence([a,b]).get_full_kspace()[1][0,0].item()==expected


def test_store_then_excitation_restores_stored_kspace(seqmod):
    reps=[seqmod.Repetition.zero(1) for _ in range(3)]
    reps[0].gradm[0,0]=2;reps[1].pulse.usage=seqmod.PulseUsage.STORE
    reps[1].gradm[0,0]=3;reps[2].pulse.usage=seqmod.PulseUsage.EXCIT
    assert seqmod.Sequence(reps).get_full_kspace()[2][0,0].item()==2

@pytest.mark.parametrize('oneshot',[False,True])
def test_chain_contrasts_and_source_independence(seqmod,repetition,oneshot):
    a=seqmod.Sequence([repetition]);b=a.clone()
    c=seqmod.chain(a,b,oneshot=oneshot)
    assert c.get_contrasts()==([1,2] if oneshot else [1,2,3,4])
    assert a.get_contrasts()==b.get_contrasts()==[1,2]
    c[0].gradm[0,0]=9;assert a[0].gradm[0,0]==1


def span(seqmod,**kw):
    args=dict(t0=.5,t1=2.5,rep_abs_times=[0,1,2,3],gm_x=[2,4,6],gm_y=[0,0,0],gm_z=[0,0,0],adc_start=1,adc_count=1,adc_phase=[.7],holes=[])
    args.update(kw);return seqmod.Sequence._build_epg_soa_from_rep_slice(**args)


def test_epg_span_clips_time_and_gradient_proportionally(seqmod):
    s=span(seqmod)
    torch.testing.assert_close(s['time'],torch.tensor([.5,1.,.5]))
    torch.testing.assert_close(s['gradm'][0],torch.tensor([1.,4.,3.]))
    assert s['adc_mask'].tolist()==[False,True,False]
    assert s['adc_phase'][1].item()==pytest.approx(.7)
    e=span(seqmod,t0=4.,t1=5.);assert e['time'].numel()==0 and e['gradm'].shape==(3,0)


def test_epg_holes_keep_time_remove_gradients_and_adc(seqmod):
    s=span(seqmod,t0=0,t1=3,holes=[(1.25,1.75),(2.,2.)])
    assert s['time'].sum().item()==pytest.approx(3)
    assert s['gradm'][0].sum().item()==pytest.approx(10)
    assert s['adc_mask'].tolist()==[False,True,False,True,False]
    assert s['gradm'][:,2].abs().sum()==0

@pytest.mark.parametrize('t0,windows,expected',[(2,[],0),(2,[(0,1),(1,2)],2),(3,[(0,1),(1,2)],0),(2,[(1,2),(0,1),(4,4)],2)])
def test_preceding_bloch_pad(seqmod,t0,windows,expected):
    assert seqmod.Sequence._preceding_bloch_pad(t0,windows)==pytest.approx(expected)

@pytest.mark.parametrize('pad',[0.,.2])
def test_prepend_relax_only(seqmod,pad):
    s=span(seqmod);s['_relax_pad_dt']=pad
    out=seqmod.Sequence._prepend_relax_only(s)
    assert '_relax_pad_dt' not in out
    assert out['time'].sum().item()==pytest.approx(2+pad)
    torch.testing.assert_close(out['gradm'].sum(1),s['gradm'].sum(1))
    if pad: assert not out['adc_mask'][0] and not out['gradm'][:,0].any()


def test_collapse_no_adc_conserves_moments(seqmod):
    s=span(seqmod,adc_start=None,adc_count=0)
    out=seqmod.Sequence.soa_collapse_keep_adcs(s)
    assert out['time'].tolist()==[2.]
    torch.testing.assert_close(out['gradm'],s['gradm'].sum(1,keepdim=True))
    assert out['adc_mask'].tolist()==[False]

@pytest.mark.parametrize('timeline',[False,True])
def test_kspace_plot_contains_measured_points(seqmod,repetition,monkeypatch,timeline):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt,'show',lambda:None)
    s=seqmod.Sequence([repetition,repetition.clone()])
    try:
        s.plot_kspace_trajectory(plot_timeline=timeline)
        axes=plt.gcf().axes
        assert len(axes)==(2 if timeline else 1)
        assert len(axes[0].lines)==6
    finally: plt.close('all')
    with pytest.raises(AssertionError): s.plot_kspace_trajectory(plotting_dims='xt')


def test_import_file_builds_adc_and_gradient_events(seqmod,monkeypatch):
    class Parser:
        def encounter(self, kind, start):
            return (0.1,0.2) if start==0.0 else None

        def hybrid(self, interval):
            assert interval==pytest.approx(150e-6)
            return []

        def duration(self):
            return 1.0

        def integrate_one(self, start, end):
            return NS(pulse=NS(angle=2.1,phase=.4))

        def events(self, kind, start, end):
            if kind=="adc":
                return [.4,.6]
            if kind=="grad x" and start<.3<end:
                return [.3]
            return []

        def sample(self, adcs):
            return NS(adc=NS(phase=[.1,.2]))

        def integrate(self, times):
            count=len(times)-1
            return NS(gradient=NS(
                x=list(range(1,count+1)),
                y=list(range(-1,-count-1,-1)),
                z=[0]*count,
            ))

    parser=Parser()
    monkeypatch.setattr(seqmod.mrstar,"load_gstar",lambda filename:parser,raising=False)
    seq=seqmod.Sequence.import_file("sequence.seq")

    assert len(seq)==1
    rep=seq[0]
    assert rep.pulse.usage is seqmod.PulseUsage.REFOC
    torch.testing.assert_close(rep.event_time,torch.tensor([.05,.1,.1,.2]))
    torch.testing.assert_close(rep.gradm,torch.tensor([
        [1.,-1.,0.],[2.,-2.,0.],[3.,-3.,0.],[4.,-4.,0.],
    ]))
    assert rep.adc_usage.tolist()==[0,0,1,1]
    torch.testing.assert_close(
        rep.adc_phase[2:],
        torch.tensor([torch.pi/2-.1,torch.pi/2-.2]),
    )
