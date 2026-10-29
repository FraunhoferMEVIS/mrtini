# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check actual native inputs while still running the real native builder."""
import pytest
import torch
import mrtini_prepass_rs as native
from src.mrtini.simulation import compute_graph
from test_numerics import case

def test_species_means_and_own_taps_reach_the_real_native_prepass(monkeypatch):
    seq,_,w,ws,f,fs,art,data=case(species=2)
    art.pulses['p'].combined_taps=object()  # must never enter the native builder
    art.pulses['p'].fat.support={r:x[::2] for r,x in fs.items()}
    calls=[];original=native.compute_graph_packed
    def record(*args,**kwargs):calls.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(native,'compute_graph_packed',record)
    for flag in (False,True):
        graph,keys=compute_graph(seq,data,art,max_state_count=10,per_species_tissue=flag)
        args=calls[-1];tissues,taps,supports=args[2:5]
        assert graph.case_species==[0,1] and len(keys)==len(seq)
        for label in (0,1):
            for i,name in enumerate(('T1','T2','T2dash','D')):
                values=getattr(data,name);expected=values[data.species_id==label].mean() if flag else values.mean()
                assert tissues[label][i]==pytest.approx(float(expected))
        torch.testing.assert_close(taps[0][0].pp,w.pp)
        torch.testing.assert_close(taps[1][0].pp,f.pp)
        torch.testing.assert_close(supports[0][0]['++'],ws['++'])
        torch.testing.assert_close(supports[1][0]['++'],fs['++'][::2])

def test_nonzero_species_cannot_be_omitted():
    seq,_,_,_,_,_,art,data=case(species=2)
    with pytest.raises(ValueError,match='omits a species'):
        compute_graph(seq,data,art,species=(0,))
