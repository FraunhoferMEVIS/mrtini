# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public graph API for the integrated MRtini engine."""
from .optimized import (
    compute_graph_fast,execute_graph_fast,PackedGraph,FastRFContext,benchmark_case,
)

compute_graph=compute_graph_fast
execute_graph=execute_graph_fast
__all__=['compute_graph','execute_graph','compute_graph_fast','execute_graph_fast',
         'PackedGraph','FastRFContext','benchmark_case']
