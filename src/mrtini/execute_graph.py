
'''
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of MRtini Simulator.
#
# Portions of this file are derived from MRzero-Core:
#   https://github.com/MRsources/MRzero-Core
#
# Original MRzero-Core copyright belongs to its respective authors.
# Modifications and extensions in this file are Copyright (C) 2026
# <Fraunhofer MEVIS>.
# Contact <Snawar Hussain / Fraunhofer MEVIS> for details.
# This derived work is distributed under the terms of the GNU Affero
# General Public License v3.0 or later. See LICENSE and NOTICE files
# for details.
'''
from __future__ import annotations
import torch
import sys
sys.path.append("./")
from src.voxelgridphantom.sim_data import SimData
from src.sequence.sequence import Sequence
from src.mrtini.mrtini_main import HybridArtifacts


# NOTE: return encoding and magnetization is currently missing. If we want to
# move it into core, it probably should be done by providing a callback that
# has can extract all information it wants during simulation.
def rigid_motion(voxel_pos, motion_func):
    """Shape of returned tensor: events x voxels x 3"""
    def voxel_motion(time):
        # rot: events x 3 x 3, offset: events x 3
        rot, offset = motion_func(time)
        rot = rot.to(device=voxel_pos.device)
        offset = offset.to(device=voxel_pos.device)
        return torch.einsum("vi, eij -> evj", voxel_pos, rot) + offset[:, None, :]
    
    return voxel_motion

def execute_graph(graph,
                  seq: Sequence,
                  data: SimData,
                  rep_keys: list[str],
                  art:HybridArtifacts,
                  SLICE_THICKNESS: float = 0.005,
                  min_emitted_signal: float = 1e-4,
                  min_latent_signal: float = 1e-4,
                  print_progress: bool = True,
                  return_mag_p: int | bool | None = None,
                  return_mag_z: int | bool | None = None,

                  ) -> torch.Tensor | list:
    """Calculate the signal of the sequence by executing the phase graph.

    Parameters
    ----------
    graph : Graph
        Phase Distribution Graph that will be executed.
    seq : Sequence
        Sequence that will be simulated and was used to create :attr:`graph`.
    data : SimData
        Physical properties of phantom and scanner.
    min_emitted_signal : float
        Minimum "emitted_signal" metric of a state for it to be measured.
    min_latent_signal : float
        Minimum "latent_signal" metric of a state for it to be simulated.
        Should be <= than min_emitted_signal.
    print_progress : bool
        If true, the current repetition is printed while simulating.
    return_mag_z : int or bool, optional
        If set, returns the longitudinal magnetisation of either the given
        repetition (int) or all repetitions (``True``).


    Returns
    -------
    signal : torch.Tensor
        The simulated signal of the sequence.
    mag_z : torch.Tensor | list[torch.Tensor]
        The longitudinal magnetisation of the specified or all repetition(s).

    """
    # This is a function that maps time to voxel positions.
    # If it is defined, motion is simulated, otherwise the static data.voxel_pos is used
    
    '''
                    
        
        # ---------------------------------------------------------------------
        # MRtini extension: water fat handling, slice profiles, newer faster graph execution
        # Replaces/extends the original MRzero Execution to inlude Bloch Taps, 
        # signal equation modification to handle slice profiles, newer graph execution
        # and trajectory handling and more
        # ---------------------------------------------------------------------


    '''
    
    t0 = 0

    D_scaled = -1e-9 * data.D
    ONE_THIRD_2PI2 = (1.0 / 3.0) * (2.0 * torch.pi) ** 2
    SQRT2 = 1.41421356237

    abs_T1 = torch.abs(data.T1)
    abs_T2 = torch.abs(data.T2)
    abs_T2dash = torch.abs(data.T2dash)

    zero_kt = torch.zeros(4, device=data.device)
    ez = torch.zeros(4, device=data.device)
    ez[2] = 1.0

    NEG_KT_REL = {"-+", "-z"}
    NEG_MAG_REL = {"-z", "-+"}
    voxel_pos_func = data.voxel_motion
    if voxel_pos_func is None and data.phantom_motion is not None:
        voxel_pos_func = rigid_motion(data.voxel_pos, data.phantom_motion)


    if seq.normalized_grads:
        grad_scale = 1 / data.size
    else:
        grad_scale = torch.ones_like(data.size)
    signal: list[torch.Tensor] = []

    # Proton density can be baked into coil sensitivity. shape: voxels x coils
    coil_sensitivity = (
        data.coil_sens.t().to(torch.cfloat)
        * torch.abs(data.PD).unsqueeze(1)
    )
    coil_count = int(coil_sensitivity.shape[1])
    voxel_count = data.PD.numel()

    # The first repetition contains only one element: A fully relaxed z0
    graph[0][0].mag = torch.ones(
        voxel_count, dtype=torch.cfloat, device=data.device
    )
    # Calculate kt_vec ourselves for autograd
    graph[0][0].kt_vec = torch.zeros(4, device=data.device)    
    
    mag_p = []
    mag_z = []
    fat_mask = (data.species_id == 1).view(-1).to(data.device)   # [1, Nsupp]
    fat_mask = torch.zeros_like(fat_mask) 
    for i, (dists, rep) in enumerate(zip(graph[1:], seq)):
        if print_progress:
            print(f"\rCalculating repetition {i+1} / {len(seq)}", end='')
                    
        key = rep_keys[i]
        vox6 = art.pulses[key].water.vox6
        vox6_fat = art.pulses[key].fat.vox6 if art.pulses[key].fat is not None else None
        supp_map = art.pulses[key].water.supp_map
        phase = torch.as_tensor(rep.pulse.phase)
        shim_array = torch.as_tensor(rep.pulse.shim_array)
        dkz = art.pulses[key].water.nominal_taps.dk
        deltas = art.pulses[key].water.nominal_taps.deltas
        
        # 1Tx or pTx?
        if shim_array.shape[0] == 1:
            B1 = data.B1.sum(0)
        else:
            assert shim_array.shape[0] == data.B1.shape[0]
            shim = shim_array[:, 0] * torch.exp(-1j * shim_array[:, 1])
            B1 = (data.B1 * shim[:, None]).sum(0)
        phase = phase + B1.angle()

        # ##+================= slice profile enabled ==================##
        phi = phase
        e_p = torch.exp(1j*phi)      # e^{+iφ}
        e_m = e_p.conj()             # e^{-iφ}
        e_2p = e_p*e_p                  # e^{+i2φ}

        
        coeff_cache = {}

        vox6_dict = {
            "zz": vox6.zz, "pp": vox6.pp, "zp": vox6.zp,
            "pz": vox6.pz, "mz": vox6.mz, "mp": vox6.mp,
        }

        vox6_fat_dict = None
        if vox6_fat is not None:
            vox6_fat_dict = {
                "zz": vox6_fat.zz, "pp": vox6_fat.pp, "zp": vox6_fat.zp,
                "pz": vox6_fat.pz, "mz": vox6_fat.mz, "mp": vox6_fat.mp,
            }

        TRANSFORM = {
            "zz": ("zz", False, None),
            "++": ("pp", False, None),
            "z+": ("zp", False, e_p),
            "+z": ("pz", False, e_m),
            "-z": ("mz", True,  e_p),
            "-+": ("mp", True,  e_2p),
        }

        eff_coeff_cache = {}

        def pick_eff_coeff(relation, supp_key):
            key = (relation, supp_key)
            coeff = eff_coeff_cache.get(key)
            if coeff is not None:
                return coeff

            coeff_name, _, phase = TRANSFORM[relation]
            coeff = pick_coeff(coeff_name, supp_key)

            # bake RF phase into the coefficient once per relation/support
            if phase is not None:
                coeff = coeff * phase

            eff_coeff_cache[key] = coeff
            return coeff
        
        def pick_coeff(name, supp_key):
            key = (name, supp_key)
            coeff = coeff_cache.get(key)
            if coeff is not None:
                return coeff

            idx = supp_map[supp_key]
            water = vox6_dict[name][:, idx]

            if vox6_fat_dict is None:
                coeff = water
            else:
                fat = vox6_fat_dict[name][:, idx]
                mask = fat_mask[idx] if fat_mask.numel() != water.shape[-1] else fat_mask
                coeff = torch.where(mask[None, :], fat, water)

            coeff_cache[key] = coeff
            return coeff        
        # shape: events
        adc = rep.adc_usage > 0
        adc_idx = adc.nonzero(as_tuple=True)[0]

        # shape: events x 4
        trajectory = torch.cumsum(
            torch.cat([rep.gradm * grad_scale[None, :], rep.event_time[:, None]], 1),
            0
        )

        # Only ADC rows of trajectory, reused for all emitted states
        # Equivalent to trajectory[adc, :], but computed once.
        trajectory_adc = trajectory.index_select(0, adc_idx)

        # shape: adc_events x coils
        rep_sig = torch.zeros(
            adc_idx.numel(), coil_count,
            dtype=torch.cfloat,
            device=data.device
        )

        dt = rep.event_time
        total_time = rep.event_time.sum()

        r1 = torch.exp(-total_time / abs_T1)
        r2 = torch.exp(-total_time / abs_T2)

        # ADC phase only at sampled events
        adc_rot = torch.exp(1j * rep.adc_phase).unsqueeze(1)
        adc_rot_adc = adc_rot.index_select(0, adc_idx)

        # T2 relaxation only at ADC events
        T2_adc = torch.exp(-trajectory_adc[:, 3:] / abs_T2)
        # Calculate the additional phase carried of voxels because of motion
        motion_phase = 0
        if voxel_pos_func is not None:
            time = t0 + torch.cat([torch.zeros(1, device=data.device), trajectory[:, 3]])
            # Shape: events x voxels x 3
            voxel_traj = voxel_pos_func((time[:-1] + time[1:]) / 2) - data.voxel_pos[None, :, :]
            # Shape: events x voxels
            motion_phase = torch.einsum("evi, ei -> ev", voxel_traj, rep.gradm * grad_scale[None, :]).cumsum(0)
        t0 += total_time

        mag_p_rep = []
        mag_p.append(mag_p_rep)
        mag_z_rep = []
        mag_z.append(mag_z_rep)
        
        for dist in dists:

            # Skip early before scanning ancestors
            if dist.dist_type != 'z0' and dist.latent_signal < min_latent_signal:
                continue

            acc = None
            first_ancestor = None

            for ancestor in dist.ancestors:
                if ancestor[1].mag is None:
                    continue

                if first_ancestor is None:
                    first_ancestor = ancestor

                relation, parent, _, supp_key = ancestor

                coeff = pick_eff_coeff(relation, supp_key)
                mag = parent.mag.conj() if relation in NEG_MAG_REL else parent.mag

                if acc is None:
                    acc = mag * coeff
                else:
                    acc.addcmul_(mag, coeff)

            if dist.dist_type != 'z0' and first_ancestor is None:
                continue

            dist.mag = acc

            if dist.dist_type == '+' and return_mag_p in [i, True]:
                mag_p_rep.append(dist.mag)

            if dist.dist_type in ['z0', 'z'] and return_mag_z in [i, True]:
                mag_z_rep.append(dist.mag)

            # kt_vec update
            if dist.dist_type == 'z0':
                dist.kt_vec = zero_kt
            else:
                ancestor0 = first_ancestor
                base_kt = ancestor0[1].kt_vec
                offset = dkz * deltas[ancestor0[3]]

                if ancestor0[0] in NEG_KT_REL:
                    dist.kt_vec = -base_kt + offset * ez
                else:
                    dist.kt_vec = base_kt + offset * ez

            if dist.dist_type == '+':
                # All transverse states need full trajectory for final state update
                dist_traj = dist.kt_vec + trajectory

                k2 = dist_traj[:, :3]
                k1 = torch.empty_like(k2)
                k1[0, :] = dist.kt_vec[:3]
                k1[1:, :] = k2[:-1, :]

                b = ONE_THIRD_2PI2 * dt * (k1*k1 + k1*k2 + k2*k2).sum(1)
                cb = torch.cumsum(b, 0)

                # Final diffusion for state propagation: shape [Nvox]
                # This replaces diffusion[-1, :] without creating full [events, voxels].
                final_diffusion = torch.exp(D_scaled * cb[-1])

                # Only emitted + states contribute to signal
                if dist.emitted_signal >= min_emitted_signal:
                    # Equivalent to dist_traj[adc, :], but avoids boolean indexing every time.
                    adc_dist_traj = dist.kt_vec + trajectory_adc

                    cb_adc = cb.index_select(0, adc_idx)
                    diffusion_adc = torch.exp(D_scaled * cb_adc[:, None])

                    if isinstance(motion_phase, torch.Tensor):
                        adc_motion_phase = motion_phase.index_select(0, adc_idx)
                    else:
                        adc_motion_phase = motion_phase

                    T2dash = torch.exp(-torch.abs(adc_dist_traj[:, 3:]) / abs_T2dash)

                    t = adc_dist_traj[:, 3:]
                    t_eff = torch.clamp(t - rep.event_time[0] * 2, min=0.0)

                    rot = torch.exp(2j * torch.pi * (
                        (t * data.B0)
                        + (adc_dist_traj[:, :3] @ data.voxel_pos.T)
                        + adc_motion_phase
                    ))

                    dephasing = data.dephasing_func(
                        adc_dist_traj[:, :2], data.nyquist[:2]
                    )[:, None]

                    if not data.multi_slice:
                        wz = torch.sinc(adc_dist_traj[:, 2] * SLICE_THICKNESS)[:, None]
                        dephasing = dephasing * wz

                    transverse_mag = (
                        SQRT2 * dist.mag
                        * rot
                        * T2_adc
                        * T2dash
                        * diffusion_adc
                        * dephasing
                    )

                    rep_sig += transverse_mag @ coil_sensitivity

                # This happens for all + states, emitted or not
                dist.mag = dist.mag * r2 * final_diffusion

                if isinstance(motion_phase, torch.Tensor):
                    dist.mag = dist.mag * torch.exp(2j * torch.pi * motion_phase[-1, :])

                dist.kt_vec = dist_traj[-1]

            else:  # z or z0
                k2_z = (dist.kt_vec[:3] * dist.kt_vec[:3]).sum()
                diffusion_z = torch.exp(D_scaled * total_time * k2_z)

                dist.mag = dist.mag * r1 * diffusion_z

                if dist.dist_type == 'z0':
                    dist.mag = dist.mag + 1 - r1
        '''
        # ---------------------------------------------------------------------
        # End MRtini extension: Newer graph execution with slice profile handling
        # ---------------------------------------------------------------------
        '''
        rep_sig *= adc_rot_adc

        # Remove ancestors to save memory as we don't need them anymore.
        # When running with autograd this doesn't change memory consumption
        # bc. the values are still stored in the computation graph.

        for dist in dists:
            for ancestor in dist.ancestors:
                ancestor[1].mag = None
                
        

        signal.append(rep_sig)
        coeff_cache.clear()
        eff_coeff_cache.clear()
    if print_progress:
        print(" - done")
        
    measured = torch.cat(signal)

    if return_mag_p is not None:
        if return_mag_z is not None:
            return measured, mag_p, mag_z
        else:
            return measured, mag_p
    elif return_mag_z is not None:
        return measured, mag_z
    else:
        return measured
