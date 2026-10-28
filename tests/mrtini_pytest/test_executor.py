"""Execute tiny hand-authored graphs through the real Python signal engine.
Graph construction is intentionally independent of the native prepass.
"""
from types import SimpleNamespace as NS
import math
import pytest
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from reference_math import channel_six


def tiny_case(modules,delta=0,emitted=1.,multi_slice=False):
    s=modules.sequence.Sequence(normalized_grads=False)
    rep=s.new_rep(2);rep.event_time[:]=torch.tensor([.01,.02]);rep.adc_usage[:]=1
    parent=NS(mag=None,kt_vec=None,dist_type='z0')
    state=NS(mag=None,kt_vec=None,dist_type='+',latent_signal=1.,emitted_signal=emitted,ancestors=[('z+',parent,0,0)])
    # Physical ideal +90-degree y rotation, generated outside MRtini.
    c=channel_six(Rotation.from_rotvec([0.,math.pi/2,0.]).as_matrix())
    vox=modules.rf.Coeff6(**{k:torch.as_tensor(v,dtype=torch.complex64).reshape(1,1) for k,v in c.items()})
    water=NS(vox6=vox,supp_map={0:0},nominal_taps=NS(dk=100.,deltas=torch.tensor([delta])))
    art=NS(pulses={'p':NS(water=water,fat=None)})
    data=NS(D=torch.zeros(1),T1=torch.ones(1)*1.,T2=torch.ones(1)*.1,T2dash=torch.ones(1)*.2,
        device='cpu',voxel_motion=None,phantom_motion=None,voxel_pos=torch.zeros(1,3),
        size=torch.ones(3),coil_sens=torch.ones(1,1,dtype=torch.complex64),PD=torch.ones(1),
        B1=torch.ones(1,1,dtype=torch.complex64),B0=torch.zeros(1),nyquist=torch.ones(3),
        species_id=torch.zeros(1,dtype=torch.int64),multi_slice=multi_slice,
        dephasing_func=lambda k,n: torch.ones(k.shape[0]))
    return [[parent],[state]],s,data,art,state

@pytest.mark.parametrize('delta',[0,1,3,6,-1])
@pytest.mark.parametrize('multi_slice',[False,True])
def test_signal_decay_slice_sinc_and_relaxation(modules,delta,multi_slice):
    graph,seq,data,art,state=tiny_case(modules,delta=delta,multi_slice=multi_slice)
    signal=modules.executor.execute_graph(graph,seq,data,['p'],art,print_progress=False)
    t=torch.tensor([.01,.03])
    expected=torch.exp(-t/.1-t/.2)
    if not multi_slice: expected*=torch.sinc(torch.tensor(delta*100*.005))
    assert signal.shape==(2,1)
    torch.testing.assert_close(signal[:,0],expected.to(torch.complex64),atol=2e-6,rtol=2e-6)
    assert graph[0][0].mag is None
    torch.testing.assert_close(state.mag,torch.tensor([math.exp(-.03/.1)/math.sqrt(2)],dtype=torch.complex64))

@pytest.mark.parametrize('return_p,return_z',[(None,None),(True,None),(None,True),(True,True)])
def test_return_modes_and_signal_threshold(modules,return_p,return_z):
    graph,seq,data,art,state=tiny_case(modules,emitted=0.)
    out=modules.executor.execute_graph(graph,seq,data,['p'],art,return_mag_p=return_p,return_mag_z=return_z,print_progress=True)
    measured=out if return_p is None and return_z is None else out[0]
    assert measured.shape==(2,1) and not measured.any()
    assert state.mag is not None  # latent propagation continues below emitted threshold

@pytest.mark.parametrize('relation',['zz','+z','-z','++','-+'])
def test_state_transfer_conjugation_and_relaxation(modules,relation):
    graph,seq,data,art,state=tiny_case(modules)
    parent=graph[0][0]
    # Keep the longitudinal branch too, so zz has a longitudinal parent.
    longitudinal=NS(mag=None,kt_vec=None,dist_type='z0',latent_signal=1.,emitted_signal=0.,ancestors=[('zz',parent,0,0)])
    graph[1].append(longitudinal)
    source=longitudinal if relation=='zz' else state
    # First repetition creates transverse magnetization. Second transfers it.
    seq[0].pulse.phase[:]=.3
    rep=seq.new_rep(1);rep.event_time[:]=.04
    dest=NS(mag=None,kt_vec=None,dist_type='z' if relation in ['+z','-z'] else ('z0' if relation=='zz' else '+'),
            latent_signal=1.,emitted_signal=1.,ancestors=[(relation,source,0,0)])
    graph.append([dest])
    out=modules.executor.execute_graph(graph,seq,data,['p','p'],art,print_progress=False,return_mag_z=True)
    initial=torch.tensor([complex(math.cos(.3),math.sin(.3))*math.exp(-.03/.1)/math.sqrt(2)],dtype=torch.complex64)
    if relation in ['-z','-+']: initial=initial.conj()
    tau=1. if dest.dist_type in ['z','z0'] else .1
    coefficient={'++':.5,'-+':-.5,'+z':-1/math.sqrt(2),'-z':-1/math.sqrt(2),'zz':0.}[relation]
    expected=coefficient*initial*math.exp(-.04/tau)
    if relation=='zz': expected=torch.tensor([1-math.exp(-.04)],dtype=torch.complex64)
    torch.testing.assert_close(dest.mag,expected,atol=2e-6,rtol=2e-6)


def test_voxel_motion_zero_displacement_and_adc_phase(modules):
    graph,seq,data,art,state=tiny_case(modules)
    data.voxel_motion=lambda time: torch.zeros(len(time),1,3)
    seq[0].adc_phase[:]=math.pi/2
    out=modules.executor.execute_graph(graph,seq,data,['p'],art,print_progress=False)
    assert out.real.abs().max()<1e-6 and (out.imag>0).all()


def test_rigid_motion_translation(modules):
    pos=torch.tensor([[1.,2.,3.]])
    def motion(t):
        return torch.eye(3).expand(len(t),3,3),torch.tensor([[.1,.2,.3]]).expand(len(t),3)
    f=modules.executor.rigid_motion(pos,motion)
    torch.testing.assert_close(f(torch.tensor([0.,1.])),torch.tensor([[[1.1,2.2,3.3]],[[1.1,2.2,3.3]]]))
