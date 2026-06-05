'''
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 Fraunhofer MEVIS, Germany
#
# This file is part of MRtini Simulator.
# It contains original code written for MRtini Simulator.
'''
import torch
import torchkbnufft as tkbn


def _reconstruct(signal, kspace_trajectory, im_size=(64, 64), FOV=(0.256, 0.256)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = tkbn.KbNufftAdjoint(im_size=im_size).to(device)

    sig = torch.as_tensor(signal, device=device)
    if sig.ndim == 1:              # [Nsamp]
        sig = sig[None, None, :]   # [1, 1, Nsamp]
    elif sig.ndim == 2:            # [Nsamp, Nc]
        sig = sig.T[None, ...]     # [1, Nc, Nsamp]
    else:
        raise ValueError(f"Unexpected signal shape {sig.shape}")

    k = torch.as_tensor(kspace_trajectory[:, :2], dtype=torch.float32, device=device).T  # [2, Nsamp]

    Nx, Ny = im_size
    FOVx, FOVy = FOV
    dx, dy = FOVx / Nx, FOVy / Ny

    omega = torch.stack([
        2 * torch.pi * k[0] * dx,
        2 * torch.pi * k[1] * dy,
    ], dim=0)  # radians/voxel

    # For debugging Cartesian RARE, skip DCF first
    # Calculate density compensation
    dcomp = tkbn.calc_density_compensation_function(ktraj=omega, im_size=im_size)

    # P
    img = adj(sig*dcomp, omega,)

    # single coil -> [1,1,Nx,Ny]
    if img.shape[1] == 1:
        return img[0, 0]
    # multi-coil, no smaps
    return torch.sqrt((img.abs() ** 2).sum(dim=1))[0]