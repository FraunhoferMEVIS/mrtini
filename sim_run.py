# SPDX-FileCopyrightText: 2026 Fraunhofer Institute for Digital Medicine MEVIS
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of MRTini.
#
# MRTini is research software and is not qualified for use as a medical
# device or as part of a medical device.
import matplotlib
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
import torch
import sys

sys.path.append("./")
from src.mrtini.mrtini_main import MRtiniSim
from src.voxelgridphantom import voxel_grid_phantom
from src.mrtini.config import SLICE_THICKNESS, PAD
from src.sequence.sequence import Sequence
from src.recon.nufft_recon import _reconstruct
from src.mrtini.RF_banks import build_prepass_inputs_from_hybrid
import numpy as np
from src.mrtini.optimized import compute_graph_fast, execute_graph_fast
import src.mrtini as mrtini
Nx = Ny = 128
# seq_names = ["2D_spiral", "2D_radial", "SE_EPI","EPI", "FLASH", "RARE"]
seq_names = [f"FLASH"]
phantom_path = mrtini.download_brainwebphantom(subject=4)
for seq_name in seq_names:
    seq = Sequence.import_file_for_mrtini_sim(f"src/sequence/gstar_seqs/{seq_name}_sequence_raw.json")
    phantom = voxel_grid_phantom.VoxelGridPhantom.brainweb(str(phantom_path))
    phantom = phantom.interpolate(Nx, Ny, 31).slices([15])  # select slices
    # phantom = phantom.interpolate(Nx, Ny, 320).slices(list(np.linspace(160-16, 160+16, 32, dtype=int)))  # to use 3D slice
    # data.multi_slice = True  # enable multi-slice mode, which relies on the z grid samples to capture through-plane effects for each slice independently.
    phantom.size[-1] = torch.tensor([SLICE_THICKNESS])  # we need correct size for slice profile calculation
    data = phantom.build()

    art = MRtiniSim(device="cuda", seq=seq).bake_bloch_kernels(
        phantom_data=data,
        z_span_m=SLICE_THICKNESS,  # we need to specify the z span for the pre-pass graph construction, which relies on the z grid samples to capture through-plane effects. This should ideally be set to cover the whole phantom, but for simplicity we just set it to cover the 31 slices we are using.
        global_RF=False,  ##  Change to True to use global RF ~ hard pulse approximation for Faster sim
    )

    rep_keys, pulse_id_per_rep, taps_bank, supp_bank = build_prepass_inputs_from_hybrid(seq, art)
    seq = seq.cuda()
    data = data.cuda()
    graph, rep_keys = compute_graph_fast(
            seq, data, art,
            max_state_count=800,
            min_state_mag=1e-4,          # use your current value
            slice_thickness=SLICE_THICKNESS,
        )



        


    with torch.no_grad():
        signal = execute_graph_fast(
            graph, seq, data, rep_keys, art,
            gpu_engine=True,
            kernel_mode="dynamic",
            rf_backend="auto",
            signal_workspace_mb=128,
            print_progress=True,
        )

    reco = _reconstruct(
        signal,
        seq.get_kspace(),
        im_size=[128, 128],
    )
    img = torch.abs(reco)
    kspace = signal.view(128, -1)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    img = img.cpu().numpy()

    axes[0].imshow(torch.log(torch.abs(kspace)).cpu().numpy(), cmap="gray", origin="lower")
    axes[0].set_title("K space built from signal")
    im = axes[1].imshow(img, cmap="gray", origin="lower")
    ## show the colorbar values for axes1
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title("Adjoint Reco")

    plt.tight_layout()
    plt.savefig(f"recon_output/{seq_name}_reco.png")
    plt.close()