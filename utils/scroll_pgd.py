# SPDX-FileCopyrightText: 2023 MRzero
# SPDX-FileCopyrightText: 2026 Fraunhofer Institute for Digital Medicine MEVIS
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of MRTini.
#
# Portions of this file are derived from MRzero-Core:
# https://github.com/MRsources/MRzero-Core
#
# This file contains modifications and extensions made for MRTini.
# The original MRzero-Core copyright and attribution notices are retained.
#
# MRTini is research software and is not qualified for use as a medical
# device or as part of a medical device.
from __future__ import annotations
import math
from typing import Any, Dict, List, Sequence, Tuple, Optional
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from pyvis.network import Network
# Minimal fallback template for PyVis (uses vis-network CDN)
_PYVIS_FALLBACK_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <style type="text/css">html, body {margin:0; padding:0;} #mynetwork { width: 100%; height: {{ height }}; background: {{ bgcolor }}; }</style>
  <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
  <link href="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.css" rel="stylesheet" type="text/css" />
</head>
<body>
<div id="mynetwork"></div>
<script type="text/javascript">
  const nodes = new vis.DataSet({{ nodes | tojson }});
  const edges = new vis.DataSet({{ edges | tojson }});
  const container = document.getElementById('mynetwork');
  const data = {nodes: nodes, edges: edges};
  const options = {{ options | safe }};
  const network = new vis.Network(container, data, options);
</script>
</body>
</html>"""


# ---------- robust readers for your PyDistribution ----------
def _to_complex(x) -> complex:
    if isinstance(x, complex):
        return x
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            return complex(x)
    except Exception:
        pass
    if _HAS_TORCH and torch.is_tensor(x):
        x = x.detach().cpu()
        return complex(x.item()) if x.is_complex() else complex(float(x.item()))
    return complex(x)

def _mag_abs(node) -> float:
    # try runtime 'mag' (main pass) first, else prepass_mag
    if hasattr(node, "mag") and getattr(node, "mag") is not None:
        return abs(_to_complex(getattr(node, "mag")))
    if hasattr(node, "prepass_mag") and getattr(node, "prepass_mag") is not None:
        return abs(_to_complex(getattr(node, "prepass_mag")))
    return 0.0

def _kt(node) -> Tuple[float,float,float,float]:
    v = None
    if hasattr(node, "kt_vec") and getattr(node, "kt_vec") is not None:
        v = getattr(node, "kt_vec")
    elif hasattr(node, "prepass_kt_vec") and getattr(node, "prepass_kt_vec") is not None:
        v = getattr(node, "prepass_kt_vec")
    if v is None: return (0.0,0.0,0.0,0.0)
    if _HAS_TORCH and torch.is_tensor(v): v = v.detach().cpu().numpy()
    v = np.asarray(v, dtype=float).ravel()
    if v.size != 4: return (0.0,0.0,0.0,0.0)
    # v[:3] *= .256
    # v[:3] /= 2*np.pi
    
    return float(v[0]), float(v[1]), float(v[2]), float(v[3])

def _dist_type(node) -> str:
    return str(getattr(node, "dist_type", "?"))

def _emitted(node) -> Optional[float]:
    return float(getattr(node, "emitted_signal", np.nan)) if hasattr(node, "emitted_signal") else None

def _latent(node) -> Optional[float]:
    return float(getattr(node, "latent_signal", np.nan)) if hasattr(node, "latent_signal") else None

def _iter_ancestors(node):
    """Yield (relation:str, ancestor) for your tuple-style ancestors."""
    if not hasattr(node, "ancestors") or node.ancestors is None: return
    try:
        for e in list(node.ancestors):
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                yield str(e[0]), e[1]
    except Exception:
        return


# ---------- building and rendering ----------
_NODE_STYLE = {
    "z0": dict(color="#fed766", shape="star"),
    "+":  dict(color="#2ab7ca", shape="dot"),
    "z":  dict(color="#fe4a49", shape="square"),
    "?":  dict(color="#8d99ae", shape="diamond"),
    "zpos_shift": dict(color="#c20114", shape="square"),
    "zneg_shift": dict(color="#30d556", shape="square"),
    "+pos_shift": dict(color="#c20114", shape="dot"),
    "+neg_shift": dict(color="#30d556", shape="dot"),
}
_EDGE_COLOR = {
    "++": "#000000",   # orangered
    "z+": "#707070",   # seagreen
    "-+": "#cbcdcd",   # darkorange
    "+z": "#6e5774",   # slateblue
    "zz": "#231c35",   # gray
    "-z": "#242039",   # teal
}

def _scale_sizes_per_layer(nodes: List[Dict[str, Any]], min_size=8, max_size=28):
    """Set node 'value' (size) based on |mag| per layer; ensures small states still visible."""
    by_layer: Dict[int, List[float]] = {}
    for n in nodes:
        by_layer.setdefault(n["level"], []).append(n["mag"])
    for n in nodes:
        mags = by_layer[n["level"]]
        mlo, mhi = min(mags), max(mags)
        if mhi - mlo < 1e-12:
            s = 0.5
        else:
            # gentle log for dynamic range
            s = (math.log1p(n["mag"]) - math.log1p(mlo)) / (math.log1p(mhi) - math.log1p(mlo))
        n["value"] = min_size + (max_size - min_size) * s

def _make_node_id(node) -> str:
    return str(id(node))

def render_pdg_pyvis(
    graph: Sequence[Sequence[Any]],
    out_html: str = "pdg.html",
    *,
    nyquist: Tuple[float,float,float] = (256,256,256),  # cycles/m; only used for tooltip/clamp text
    hierarchical: bool = True,
    min_mag: float = 0.0,                 # prune tiny states (by |mag|)
    max_nodes_per_layer: Optional[int] = None,  # cap per-layer for readability
    global_node_cap: Optional[int] = None,      # hard cap total nodes
    show_edge_labels: bool = True,
    show_node_labels: bool = True,
    physics: bool = False,                # keep False for hierarchical (vis.js does hierarchy better w/o physics)
    notebook: bool = False,               # set True if running in Jupyter & want inline
) -> str:
    """
    Build an interactive PDG graph with PyVis (vis.js) and write to HTML.

    Layout:
      - Layers (repetitions) are 'levels' (Left→Right); edges connect parent→child.
      - Node color/shape by dist_type; size by |mag| per layer.
      - Edge color & label by RF relation.
    """
    nx, ny, nz = nyquist

    # 1) Collect nodes/edges with metadata
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    total_nodes = 0

    L = len(graph)
    for li in range(L):
        # nodes first
        layer_nodes: List[Dict[str, Any]] = []
        for n in graph[li]:
            typ = _dist_type(n)
            kx, ky, kz, t = _kt(n)
            if kz > 0+1e-12 and typ not in ["z0",]:
                typ +="pos_shift"
            if kz < 0-1e-12 and typ not in ["z0",]:
                typ +="neg_shift"
            mag = _mag_abs(n)
            if mag < float(min_mag):
                continue
            node_id = _make_node_id(n)

            title = (
                f"{typ} | |M|={mag:.2g} \n"
                f"k=({kx:.2g},{ky:.2g},{kz:.2g}) cyc/m, t={t:.2g} s"
            )
            em = _emitted(n)
            lt = _latent(n)
            if em is not None: title += f"emitted={em:.2g} "
            if lt is not None: title += f"latent={lt:.2g}"

            style = _NODE_STYLE.get(typ, _NODE_STYLE["?"])
            layer_nodes.append(dict(
                id=node_id,
                label=(f"{typ[0]}" if show_node_labels else " "),
                title=title,
                level=li,
                mag=mag,
                color=style["color"],
                shape=style["shape"],
            ))

        # optional cap per layer (largest |mag| first)
        if max_nodes_per_layer is not None and len(layer_nodes) > max_nodes_per_layer:
            layer_nodes.sort(key=lambda d: d["mag"], reverse=True)
            layer_nodes = layer_nodes[:max_nodes_per_layer]

        nodes.extend(layer_nodes)

        # edges: from parents in layer-1→current nodes (respect your tuple ancestor list)
        if li > 0:
            # Build a set of node ids in prev layer that survived pruning
            prev_ids_alive = { _make_node_id(n) for n in graph[li-1] if _mag_abs(n) >= float(min_mag) }
            if max_nodes_per_layer is not None:
                # need to recompute which of prev survived: mirror logic
                prev_layer_nodes = [
                    (n, _mag_abs(n)) for n in graph[li-1] if _mag_abs(n) >= float(min_mag)
                ]
                if len(prev_layer_nodes) > max_nodes_per_layer:
                    prev_layer_nodes.sort(key=lambda p: p[1], reverse=True)
                    prev_ids_alive = { _make_node_id(n) for (n, _) in prev_layer_nodes[:max_nodes_per_layer] }

            # Now walk current layer raw nodes (not just kept nodes) to gather edges to kept children
            kept_ids_cur = { d["id"] for d in layer_nodes }
            for child in graph[li]:
                child_id = _make_node_id(child)
                if child_id not in kept_ids_cur:  # child pruned
                    continue
                seen = set()
                if hasattr(child, "ancestors") and child.ancestors is not None:
                    for e in list(child.ancestors):
                        if isinstance(e, (tuple, list)) and len(e) >= 2:
                            rel, anc = str(e[0]), e[1]
                        else:
                            continue
                        aid = _make_node_id(anc)
                        if aid not in prev_ids_alive:
                            continue
                        key = (aid, child_id, rel)
                        if key in seen:  # de-dup repeated same-edge
                            continue
                        seen.add(key)
                        edges.append(dict(
                            from_node=aid,
                            to_node=child_id,
                            label=(rel if show_edge_labels else " "),
                            color=_EDGE_COLOR.get(rel, "#7f8c8d"),
                            arrows="to",
                            smooth=True,
                        ))

        total_nodes = len(nodes)
        if global_node_cap is not None and total_nodes >= global_node_cap:
            nodes = nodes[:global_node_cap]
            # also trim edges pointing to missing nodes
            kept = {n["id"] for n in nodes}
            edges = [e for e in edges if e["from_node"] in kept and e["to_node"] in kept]
            break

    # 2) Size nodes per layer by |mag|
    _scale_sizes_per_layer(nodes)
    import json
    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                # spacing controls to make repetition levels more distinct
                "nodeSpacing": 600,
                "levelSeparation": 800
                ## label inside the node
            }
        },
        # turn off physics explicitly for hierarchical layout
        "physics": {"enabled": False}
    }
    # 3) Create PyVis network
    #   height/width can be tuned; set directed for arrow heads
    net = Network(height="1800px", width="100%", directed=True, notebook=notebook,
                  bgcolor="#ffffff",
                  cdn_resources="remote",
                  layout=options)

    # vis.js options: hierarchical or physics
    if hierarchical:
        # Left-to-Right, levels are our repetition index
        # Build the options as a Python dict and serialize to JSON to avoid quoting issues.
        # Disable physics when using hierarchical layout for stable level placement.
        import json
        options = {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "LR",
                    "sortMethod": "directed",
                    # spacing controls to make repetition levels more distinct
                    "nodeSpacing": 200,
                    "levelSeparation": 180
                }
            },
            # turn off physics explicitly for hierarchical layout
            "physics": {"enabled": False}
        }
    

    # 4) Add nodes & edges
    for n in nodes:
        net.add_node(
            n["id"],
            label=n["label"],
            title=n["title"],
            level=n["level"],
            color=n["color"],
            shape=n["shape"],
            value=n["value"],   # controls size with 'scaling'
        )
    # net.add_nodes(nodes)
    # net.add_edges(edges)
    for e in edges:
        net.add_edge(
            e["from_node"], e["to_node"],
            label=e["label"],
            color=e["color"],
            arrows=e["arrows"],
            smooth=e["smooth"]
        )

    # Fallback template if PyVis lost its default
    # try:
    #     if getattr(net, "template", None) is None:
    #         net.set_template(_PYVIS_FALLBACK_TEMPLATE)
    # except Exception:
    #     net.set_template(_PYVIS_FALLBACK_TEMPLATE)

    net.show(out_html, notebook=notebook)
    return out_html

