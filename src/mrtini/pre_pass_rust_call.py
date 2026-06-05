'''
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of MRtini Simulator.
#
# <Snawar Hussain/ Fraunhofer MEVIS>.
#
# This derived work is distributed under the terms of the GNU Affero
# General Public License v3.0 or later. See LICENSE and NOTICE files
# for details.
'''



import torch
import mrtini_prepass_rs as prepass
import sys
sys.path.append("./")  # ensure we can import from src/
CMPLX_DTYPE = torch.complex64
INT_DTYPE = torch.int32    
import torch

def compute_graph_rust(seq, data, pulse_id_per_rep, taps_bank, support_bank, z,
                             max_state_count=400, min_state_mag=1e-8,
                             normalized_grads=False,
                             slice_thickness=1.0, pad=1.0):
    reps = list(seq) if not isinstance(seq, list) else seq

    def cplx(x): return x.detach().contiguous().to(CMPLX_DTYPE)
    def _int(x):    return x.detach().contiguous().to(INT_DTYPE)

    # convert inputs (Rust will .cpu() itself, but make dtype correct here)
    pulse_id_per_rep = _int(pulse_id_per_rep).cpu()

    taps_banks = []
    for t in taps_bank:
        # keep as a simple python object with attributes; we just overwrite attrs with correct dtype
        t.pp = cplx(t.pp).cpu(); t.pz = cplx(t.pz).cpu(); t.mp = cplx(t.mp).cpu()
        t.mz = cplx(t.mz).cpu(); t.zp = cplx(t.zp).cpu(); t.zz = cplx(t.zz).cpu()
        t.deltas = _int(t.deltas).cpu()
        # dk must be python float
        t.dk = float(t.dk)
        taps_banks.append(t)

    support_banks = []
    for s in support_bank:
        support_banks.append({
            "++": _int(s["++"]).cpu(),
            "+z": _int(s["+z"]).cpu(),
            "-+": _int(s["-+"]).cpu(),
            "-z": _int(s["-z"]).cpu(),
            "z+": _int(s["z+"]).cpu(),
            "zz": _int(s["zz"]).cpu(),
        })

    t1 = float(torch.mean(data.T1).item())
    t2 = float(torch.mean(data.T2).item())
    t2dash = float(torch.mean(data.T2dash).item())
    d = float(torch.mean(data.D).item())  # still 1e-3 mm^2/s

    nyq = tuple(float(v) for v in data.nyquist.detach().cpu().tolist())
    size = tuple(float(v) for v in data.size.detach().cpu().tolist())

    return prepass.compute_graph_multi(
        reps,
        pulse_id_per_rep,
        t1, t2, t2dash, d,
        int(max_state_count),
        float(min_state_mag),
        nyq, size, bool(normalized_grads),
        taps_banks,
        support_banks,
        int(z.numel()),
        float(slice_thickness),
        float(pad),
    )
