# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark an actual already-baked sequence/phantom on its target device."""
import gc,time
import numpy as np
import torch
from .prepass import compute_graph_fast
from .execute_graph_fast import execute_graph_fast
from .fast_graph import FastRFContext

def benchmark_case(seq,data,art,*,repeats=3,compare_reference=True,executor_options=None,**graph_options):
    if repeats<1:raise ValueError("repeats must be positive")
    opts=dict(executor_options or {})
    opts.setdefault("print_progress",False)
    opts.setdefault("SLICE_THICKNESS",graph_options.get("slice_thickness",.005))
    for name in ("rf_context","performance"):
        if name in opts:raise ValueError(f"benchmark owns {name}")
    def sync():
        if data.PD.is_cuda:torch.cuda.synchronize(data.PD.device)
    gc.collect();sync();start=time.perf_counter()
    graph,keys=compute_graph_fast(seq,data,art,**graph_options);sync()
    report={'prepass_and_python_adapter_seconds':time.perf_counter()-start,
            'nodes':graph.payload['nodes'],'edges':graph.payload['edges'],
            'species':graph.case_species,'device':str(data.PD.device),'torch':torch.__version__}
    context=FastRFContext(graph,art,data,backend=opts.get("rf_backend","auto"))
    if data.PD.is_cuda:torch.cuda.reset_peak_memory_stats(data.PD.device)
    perf={}
    sync();start=time.perf_counter();out=execute_graph_fast(graph,seq,data,keys,art,rf_context=context,**opts);sync()
    report['first_execution_seconds']=time.perf_counter()-start
    samples=[]
    for _ in range(repeats):
        sync();start=time.perf_counter();out=execute_graph_fast(graph,seq,data,keys,art,rf_context=context,**opts);sync();samples.append(time.perf_counter()-start)
    report['warm_execution_seconds']=float(np.median(samples))
    # Separate instrumented run; GPU events and synchronization do not affect timed samples.
    out=execute_graph_fast(graph,seq,data,keys,art,rf_context=context,performance=perf,**opts);sync()
    report['execution_details']=perf
    if data.PD.is_cuda:
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(data.PD.device)
        report['peak_reserved_bytes']=torch.cuda.max_memory_reserved(data.PD.device)
    if compare_reference:
        # Correct species-aware explicit edge sums, not the old mixed complex
        # taps. This can be slow on a large graph; run once for validation.
        sync();start=time.perf_counter()
        ref_opts=dict(opts,rf_backend='reference',vectorized=False)
        ref=execute_graph_fast(graph,seq,data,keys,art,**ref_opts)
        sync();report['explicit_reference_seconds']=time.perf_counter()-start
        measured=out[0] if isinstance(out,tuple) else out
        reference=ref[0] if isinstance(ref,tuple) else ref
        report['relative_complex_adc_error']=float(torch.linalg.vector_norm(measured-reference)/torch.linalg.vector_norm(reference).clamp_min(1e-30))
        report['maximum_complex_adc_error']=float((measured-reference).abs().max())
    return report,out
