# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark an actual already-baked sequence/phantom on its target device."""
import gc,time
import numpy as np
import torch
from .prepass import compute_graph_fast
from .execute_graph_fast import execute_graph_fast
from .fast_graph import FastRFContext

def benchmark_case(seq,data,art,*,repeats=3,compare_reference=True,**graph_options):
    def sync():
        if data.PD.is_cuda:torch.cuda.synchronize(data.PD.device)
    gc.collect();sync();start=time.perf_counter()
    graph,keys=compute_graph_fast(seq,data,art,**graph_options);sync()
    report={'prepass_and_python_adapter_seconds':time.perf_counter()-start,
            'nodes':graph.payload['nodes'],'edges':graph.payload['edges'],
            'species':graph.case_species,'device':str(data.PD.device),'torch':torch.__version__}
    context=FastRFContext(graph,art,data)
    sync();start=time.perf_counter();out=execute_graph_fast(graph,seq,data,keys,art,rf_context=context,print_progress=False);sync()
    report['first_execution_seconds']=time.perf_counter()-start
    samples=[]
    for _ in range(repeats):
        sync();start=time.perf_counter();out=execute_graph_fast(graph,seq,data,keys,art,rf_context=context,print_progress=False);sync();samples.append(time.perf_counter()-start)
    report['warm_execution_seconds']=float(np.median(samples))
    if compare_reference:
        # Correct species-aware explicit edge sums, not the old mixed complex
        # taps. This can be slow on a large graph; run once for validation.
        sync();start=time.perf_counter()
        ref=execute_graph_fast(graph,seq,data,keys,art,rf_backend='reference',vectorized=False,print_progress=False)
        sync();report['explicit_reference_seconds']=time.perf_counter()-start
        report['relative_complex_adc_error']=float(torch.linalg.vector_norm(out-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-30))
        report['maximum_complex_adc_error']=float((out-ref).abs().max())
    return report,out
