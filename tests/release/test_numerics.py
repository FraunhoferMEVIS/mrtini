# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native equivalence and measured complex ADC checks (CPU)."""
import copy, json, types, unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
import mrtini_prepass_rs as rs
from native_fixtures import fixture, quiet, compare
from src.mrtini.optimized import PackedGraph, execute_graph_fast, compute_graph_fast


class Seq(list):
    normalized_grads = False


def case(species=1, nonuniform=False, reps=5, V=16, K=9):
    seq, pid, w, ws, f, fs = fixture(reps=reps, events=5, K=K)
    seq=Seq(seq);seq.mrtini_hybrid=[{'kind':'bloch'} for _ in seq]
    for r in seq:
        r.pulse.shim_array=torch.tensor([[1.,0.]])
        r.adc_phase=torch.linspace(0.,.2,r.event_count)
    def bank(t,s):
        coef={n:getattr(t,n).expand(V,-1).clone() for n in ['pp','pz','mp','mz','zp','zz']}
        if nonuniform:
            for a in coef.values():a*=torch.linspace(.9,1.1,V)[:,None]
        return NS(vox6=NS(**coef),supp_map={i:i for i in range(K)},nominal_taps=t,support=s)
    art=NS(span_key={i:'p' for i in range(len(seq))},pulses={'p':NS(water=bank(w,ws),fat=bank(f,fs) if species==2 else None)})
    ids=torch.arange(V)%species
    data=NS(PD=torch.linspace(.2,1.,V),species_id=ids,D=torch.linspace(.2,1.2,V),
            T1=torch.linspace(.8,1.4,V),T2=torch.linspace(.07,.11,V),T2dash=torch.linspace(.12,.22,V),
            B0=torch.linspace(-10.,10.,V)-440*(ids==1),B1=torch.exp(1j*torch.linspace(-.2,.2,V))[None,:],
            coil_sens=torch.stack([torch.ones(V,dtype=torch.cfloat),torch.exp(1j*torch.linspace(0,1,V))]),
            voxel_pos=torch.stack([torch.linspace(-.025,.025,V),torch.zeros(V),torch.linspace(-.001,.001,V)],1),
            size=torch.tensor([.064,.064,.005]),nyquist=torch.tensor([500.,500.,100.]),device='cpu',
            voxel_motion=None,phantom_motion=None,multi_slice=False,
            dephasing_func=lambda k,n: torch.sigmoid((n[0]-k[:,0].abs()+.5)*100)*torch.sigmoid((n[1]-k[:,1].abs()+.5)*100))
    return seq,pid,w,ws,f,fs,art,data


def original_executor():
    """Only optional profile imports disabled; actual supplied physics remains."""
    s=(Path(__file__).resolve().parent/'_executor_reference.py').read_text()
    start=s.index('from src.mrtini.event_profiles import')
    end=s.index('\ndef rigid_motion',start)
    s=s[:start]+s[end:]
    m=types.ModuleType('legacy_executor');exec(compile(s,'supplied_execute_graph.py','exec'),m.__dict__)
    return m.execute_graph_anz


class NativeTests(unittest.TestCase):
    def test_species_keep_distinct_subgrid_coordinates(self):
        seq,pid,w,ws,f,fs=fixture(reps=12,events=5,K=9);w.dk=f.dk=100.1
        tissues=[(1.2,.09,.18,.8),(.4,.04,.07,.2)]
        for normalized in [False,True]:
            tail=(20,1e-7,(500.,500.,100.),(.064,.064,.005),normalized)
            with quiet():
                old=[rs.compute_graph_multi(seq,pid,*t,*tail,[tap],[sup],9,.005,2.) for t,tap,sup in zip(tissues,[w,f],[ws,fs])]
                p=rs.compute_graph_packed(seq,pid,tissues,[[w],[f]],[[ws],[fs]],*tail,.005)
            for i,g in enumerate(old):
                report=compare(g,p,i);self.assertEqual(report['missing'],0,(normalized,i,report))
                self.assertLessEqual(report['max_abs_nominal_error'],2e-7)

    def test_retention_and_nominal_amplitudes(self):
        for cap in [5,30,150]:
            for selective in [False,True]:
                seq,pid,w,ws,f,fs=fixture(reps=10,events=5,K=9)
                if selective:
                    fs={r:s[::2] for r,s in fs.items()}
                t=(1.2,.09,.18,.8);tail=(cap,1e-7,(500.,500.,100.),(.064,.064,.005),False)
                with quiet():
                    old=[rs.compute_graph_multi(seq,pid,*t,*tail,[tap],[sup],9,.005,2.) for tap,sup in [(w,ws),(f,fs)]]
                    p=rs.compute_graph_packed(seq,pid,[t,t],[[w],[f]],[[ws],[fs]],*tail,.005)
                for i,g in enumerate(old):
                    report=compare(g,p,i)
                    self.assertEqual(report['missing'],0,(cap,selective,i,report))
                    self.assertLessEqual(report['max_abs_nominal_error'],2e-7)
                    fast=PackedGraph(p)
                    self.assertEqual(sum(len(row) for row in g),int(np.count_nonzero(fast.node_mask&(1<<i))))

    def test_bad_inputs(self):
        seq,pid,w,ws,f,fs=fixture(reps=2,K=9)
        with self.assertRaises(ValueError):rs.compute_graph_packed(seq,pid,[(1.,.1,.2,0.)],[[w]],[[ws]],0,1e-8,(1.,1.,1.),(1.,1.,1.),False,.005)


class ExecutorTests(unittest.TestCase):
    def test_factored_uniform_decay_and_bounded_tiles(self):
        for species in [1,2]:
            seq,_,_,_,_,_,art,data=case(species=species)
            data.D.zero_();data.T2dash.fill_(.15)
            graph,keys=compute_graph_fast(seq,data,art,max_state_count=20,min_state_mag=1e-7)
            ref=execute_graph_fast(copy.deepcopy(graph),seq,data,keys,art,print_progress=False,vectorized=False,rf_backend='reference')
            for S,A,V in [(16,64,1024),(1,1,3)]:
                actual=execute_graph_fast(copy.deepcopy(graph),seq,data,keys,art,print_progress=False,state_batch=S,adc_batch=A,voxel_batch=V)
                torch.testing.assert_close(actual,ref,atol=3e-5,rtol=3e-5)

    def test_whole_measured_adc(self):
        old=original_executor();records=[]
        for species in [1,2]:
            for nonuniform in [False,True]:
                seq,pid,w,ws,f,fs,art,data=case(species,nonuniform)
                graph,keys=compute_graph_fast(seq,data,art,max_state_count=20,min_state_mag=1e-7)
                ref=execute_graph_fast(copy.deepcopy(graph),seq,data,keys,art,print_progress=False,rf_backend='reference',min_emitted_signal=0.,vectorized=False)
                backends=['grouped','auto']+([] if nonuniform else ['dense'])
                for backend in backends:
                    result=execute_graph_fast(copy.deepcopy(graph),seq,data,keys,art,print_progress=False,rf_backend=backend,min_emitted_signal=0.)
                    error=float(torch.linalg.vector_norm(result-ref)/torch.linalg.vector_norm(ref))
                    torch.testing.assert_close(result,ref,atol=3e-5,rtol=3e-5)
                    records.append(dict(species=species,nonuniform=nonuniform,backend=backend,relative_complex_adc_error=error))
                if species==1:
                    # Compare against the actual pre-existing Python executor,
                    # not only the new explicit RF reference implementation.
                    with quiet():
                        legacy_graph=rs.compute_graph_multi(list(seq),pid,float(data.T1.mean()),float(data.T2.mean()),float(data.T2dash.mean()),float(data.D.mean()),20,1e-7,tuple(data.nyquist.tolist()),tuple(data.size.tolist()),False,[w],[ws],9,.005,2.)
                    before=old(legacy_graph,seq,data,keys,art,print_progress=False,min_emitted_signal=0.)
                    torch.testing.assert_close(ref,before,atol=3e-5,rtol=3e-5)
                    records.append(dict(species=1,nonuniform=nonuniform,backend='supplied_executor',relative_complex_adc_error=float(torch.linalg.vector_norm(ref-before)/torch.linalg.vector_norm(before))))
                else:
                    # Independent physical species simulations through the
                    # old real executor are an end-to-end mixed-species oracle.
                    for threshold in [0.,1e-4,.1]:
                        pieces=[]
                        for label,tap,support in [(0,w,ws),(1,f,fs)]:
                            with quiet():
                                sg=rs.compute_graph_multi(list(seq),pid,float(data.T1.mean()),float(data.T2.mean()),float(data.T2dash.mean()),float(data.D.mean()),20,1e-7,tuple(data.nyquist.tolist()),tuple(data.size.tolist()),False,[tap],[support],9,.005,2.)
                            sd=copy.copy(data);sd.PD=data.PD*(data.species_id==label);sd.species_id=torch.zeros_like(data.species_id)
                            sa=NS(pulses={'p':NS(water=art.pulses['p'].water if label==0 else art.pulses['p'].fat,fat=None)})
                            pieces.append(old(sg,seq,sd,keys,sa,print_progress=False,min_emitted_signal=threshold))
                        expect=sum(pieces)
                        result=execute_graph_fast(copy.deepcopy(graph),seq,data,keys,art,print_progress=False,min_emitted_signal=threshold)
                        torch.testing.assert_close(result,expect,atol=3e-5,rtol=3e-5)
                        records.append(dict(species=2,nonuniform=nonuniform,backend='two_independent_supplied_executors',threshold=threshold,relative_complex_adc_error=float(torch.linalg.vector_norm(result-expect)/torch.linalg.vector_norm(expect))))
        assert records

