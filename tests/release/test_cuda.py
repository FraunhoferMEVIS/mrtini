# SPDX-License-Identifier: AGPL-3.0-or-later
"""These checks execute on real hardware; CPU-only jobs report explicit skips."""
import copy
import pytest
import torch
from src.mrtini.doctor import kernel_smoke
from src.mrtini.optimized.fast_graph import apply_rf
from src.mrtini.optimized import compute_graph_fast,execute_graph_fast
from test_rf import fixture
from test_numerics import case

pytestmark=[pytest.mark.cuda,pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device unavailable')]

def test_fused_kernel_missing_support_and_empty_rows():
    assert kernel_smoke('cuda')['relative_error']<3e-5

@pytest.mark.parametrize('species',[1,2])
@pytest.mark.parametrize('uniform',[False,True])
def test_fused_rf_against_explicit_sum(species,uniform):
    plan,bank,parent,phase=fixture(P=12,D=13,V=64,E=200,K=7,species=species,uniform=uniform,device='cuda')
    ref=apply_rf(plan,bank,parent,phase,'reference');out=apply_rf(plan,bank,parent,phase,'triton')
    torch.cuda.synchronize();torch.testing.assert_close(out,ref,rtol=5e-5,atol=5e-5)

def test_complete_complex_adc_on_cuda():
    seq,_,_,_,_,_,art,data=case(species=2,nonuniform=True)
    for name,value in vars(data).items():
        if isinstance(value,torch.Tensor):setattr(data,name,value.cuda())
    data.device='cuda'
    for rep in seq:
        for name,value in vars(rep).items():
            if isinstance(value,torch.Tensor):setattr(rep,name,value.cuda())
        rep.pulse.shim_array=rep.pulse.shim_array.cuda()
    for p in [art.pulses['p'].water,art.pulses['p'].fat]:
        for name,value in vars(p.vox6).items():setattr(p.vox6,name,value.cuda())
    graph,keys=compute_graph_fast(seq,data,art,max_state_count=20,min_state_mag=1e-7)
    ref=execute_graph_fast(copy.deepcopy(graph),seq,data,keys,art,print_progress=False,vectorized=False,rf_backend='reference',min_emitted_signal=0.)
    out=execute_graph_fast(copy.deepcopy(graph),seq,data,keys,art,print_progress=False,rf_backend='triton',min_emitted_signal=0.)
    torch.cuda.synchronize();torch.testing.assert_close(out,ref,rtol=8e-5,atol=8e-5)
