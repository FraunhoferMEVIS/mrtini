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
from typing import Literal, Optional, Dict, Tuple
from warnings import warn
from scipy import io
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from .sim_data import SimData
from matplotlib.pyplot import imshow 


def sigmoid(trajectory: torch.Tensor, nyquist: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of the sinc voxel dephasing function.

    The true dephasing function of a sinc-shaped voxel (in real space) is a
    box - function, with the FFT conform size [-nyquist, nyquist[. This is not
    differentiable, so we approximate the edges with a narrow sigmod at
    ±(nyquist + 0.5). The difference is neglegible at usual nyquist freqs.
    """
    return torch.prod(torch.sigmoid(
        (nyquist - trajectory.abs() + 0.5) * 100
    ), dim=1)


def sinc(trajectory: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Box voxel (real space) dephasing function.

    The size describes the total extends of the box shape.
    """
    return torch.prod(torch.sinc(trajectory * size), dim=1)


def identity(trajectory: torch.Tensor) -> torch.Tensor:
    """Point voxel (real space) dephasing function.

    There is no dephasing.
    """
    return torch.ones_like(trajectory[:, 0])


def generate_B0_B1(PD):
    # Generate a somewhat plausible B0 and B1 map.
    # Visually fitted to look similar to the numerical_brain_cropped
    x_pos, y_pos, z_pos = torch.meshgrid(
        torch.linspace(-1, 1, PD.shape[0]),
        torch.linspace(-1, 1, PD.shape[1]),
        torch.linspace(-1, 1, PD.shape[2]),
        indexing="ij"
    )
    B1 = torch.exp(-(0.4*x_pos**2 + 0.2*y_pos**2 + 0.3*z_pos**2))
    dist2 = (0.4*x_pos**2 + 0.2*(y_pos - 0.7)**2 + 0.3*z_pos**2)
    B0 = 7 / (0.05 + dist2) - 45 / (0.3 + dist2)
    # Normalize such that the weighted average is 0 or 1
    weight = PD / PD.sum()
    B0 -= (B0 * weight).sum()
    B1 /= (B1 * weight).sum()
    return B0, B1


class VoxelGridPhantom:
    """Class for using typical phantoms like those provided by BrainWeb.

    The data is assumed to be defined by a uniform cartesian grid of samples.
    As it is bandwidth limited, we assume that there is no signal above the
    Nyquist frequency. This leads to the usage of sinc-shaped voxels.

    Attributes
    ----------
    PD : torch.Tensor
        (sx, sy, sz) tensor containing the Proton Density
    T1 : torch.Tensor
        (sx, sy, sz) tensor containing the T1 relaxation
    T2 : torch.Tensor
        (sx, sy, sz) tensor containing the T2 relaxation
    T2dash : torch.Tensor
        (sx, sy, sz) tensor containing the T2' dephasing
    D : torch.Tensor
        (sx, sy, sz) tensor containing the Diffusion coefficient
    B0 : torch.Tensor
        (sx, sy, sz) tensor containing the B0 inhomogeneities
    B1 : torch.Tensor
        (coil_count, sx, sy, sz) tensor of RF coil profiles
    coil_sens : torch.Tensor
        (coil_count, sx, sy, sz) tensor of coil sensitivities
    size : torch.Tensor
        Size of the data, in meters.
    """

    def __init__(
        self,
        PD: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        T2dash: torch.Tensor,
        D: torch.Tensor,
        B0: torch.Tensor,
        B1: torch.Tensor,
        coil_sens: torch.Tensor,
        size: torch.Tensor,
        repeat_z:int|None = None,
        PD_fat=None, T1_fat=None, T2_fat=None, T2dash_fat=None, D_fat=None,
        fat_shift_hz=0.0,
        species_id=None,
        chemical_shift=None,
        Masks: Optional[Dict[str,torch.Tensor]] = None,
        
    ) -> None:
        """Set the phantom attributes to the provided parameters.

        This function does no cloning nor contain any other funcionality. You
        probably want to use :meth:`brainweb` to load a phantom instead.
        """
        self.PD = torch.as_tensor(PD, dtype=torch.float32)
        self.T1 = torch.as_tensor(T1, dtype=torch.float32)
        self.T2 = torch.as_tensor(T2, dtype=torch.float32)
        self.T2dash = torch.as_tensor(T2dash, dtype=torch.float32)
        self.D = torch.as_tensor(D, dtype=torch.float32)
        self.B0 = torch.as_tensor(B0, dtype=torch.float32)
        self.B1 = torch.as_tensor(B1, dtype=torch.float32)
        self.coil_sens = torch.as_tensor(coil_sens, dtype=torch.float32)
        self.size = torch.as_tensor(size, dtype=torch.float32)
        self.repeat_count = repeat_z
        self.Masks = Masks
        self.PD_fat = torch.as_tensor(PD_fat, dtype=torch.float32) if PD_fat is not None else None
        self.T1_fat = torch.as_tensor(T1_fat, dtype=torch.float32) if T1_fat is not None else None
        self.T2_fat = torch.as_tensor(T2_fat, dtype=torch.float32) if T2_fat is not None else None
        self.T2dash_fat = torch.as_tensor(T2dash_fat, dtype=torch.float32) if T2dash_fat is not None else None
        self.D_fat = torch.as_tensor(D_fat, dtype=torch.float32) if D_fat is not None else None
        self.fat_shift_hz = fat_shift_hz
        self.species_id = species_id
        self.chemical_shift = chemical_shift

    def build(self, PD_threshold: float = 1e-6,
              voxel_shape: Literal["sinc", "box", "point"] = "sinc"
              ) -> SimData:
        """Build a :class:`SimData` instance for simulation.

        Arguments
        ---------
        PD_threshold : float
            All voxels with a proton density below this value are ignored.
        """
        
        """
        # ---------------------------------------------------------------------
        # MRTini extension: Separate Water and Fat based tissue handling with chemical shift
        # Replaces/extends the original MRzero RF event handling by resolving
        # shaped RF pulses on a slice grid and converting them to PDG tap sets.
        # ---------------------------------------------------------------------
        """

        if self.repeat_count:
            self.PD = self.PD.repeat(1, 1, self.repeat_count)
            self.T1 = self.T1.repeat(1, 1, self.repeat_count)
            self.T2 = self.T2.repeat(1, 1, self.repeat_count)
            self.T2dash = self.T2dash.repeat(1, 1, self.repeat_count)
            self.D = self.D.repeat(1, 1, self.repeat_count)
            self.B0 = self.B0.repeat(1, 1, self.repeat_count)
            self.B1 = self.B1.repeat(1, 1, 1, self.repeat_count)
            self.coil_sens = self.coil_sens.repeat(1, 1, 1, self.repeat_count)
            self.chemical_shift = self.chemical_shift.repeat(1, 1, self.repeat_count) if self.chemical_shift is not None else None
        
        if self.PD_fat is not None and self.repeat_count:
            self.PD_fat = self.PD_fat.repeat(1, 1, self.repeat_count)
            self.T1_fat = self.T1_fat.repeat(1, 1, self.repeat_count)
            self.T2_fat = self.T2_fat.repeat(1, 1, self.repeat_count)
            self.T2dash_fat = self.T2dash_fat.repeat(1, 1, self.repeat_count)
            self.D_fat = self.D_fat.repeat(1, 1, self.repeat_count)
            # self.chemical_shift = self.chemical_shift.repeat(1, 1, self.repeat_count) if self.chemical_shift is not None else None
        shape = torch.tensor(self.PD.shape, device=self.PD.device)

        pos_x, pos_y, pos_z = torch.meshgrid(
            self.size[0] * torch.fft.fftshift(torch.fft.fftfreq(int(shape[0]), device=self.PD.device)),
            self.size[1] * torch.fft.fftshift(torch.fft.fftfreq(int(shape[1]), device=self.PD.device)),
            self.size[2] * torch.fft.fftshift(torch.fft.fftfreq(int(shape[2]), device=self.PD.device)),
            indexing="ij"
        )
        all_pos = torch.stack([pos_x.flatten(), pos_y.flatten(), pos_z.flatten()], dim=1)

        def flatten_branch(PD, T1, T2, T2dash, D, chem_shift, extra_shift_hz=0.0, species_id=0):
            mask = PD > PD_threshold
            flat_mask = mask.flatten()

            pos = all_pos[flat_mask]
            pd = PD[mask]
            t1 = T1[mask]
            t2 = T2[mask]
            t2d = T2dash[mask]
            d = D[mask]
            chem_shift = chem_shift[mask] if chem_shift is not None else None

            # If B0 is stored in Hz, this is correct directly.
            # If B0 is stored in Tesla, convert extra_shift_hz -> Tesla before adding.
            b0 = self.B0[mask]
            
            b1 = self.B1[:, mask]
            c = self.coil_sens[:, mask]
            sp = torch.full_like(pd, species_id, dtype=torch.int64)

            return pos, pd, t1, t2, t2d, d, b0, b1, c, sp

        chunks = []
        chunks.append(flatten_branch(self.PD, self.T1, self.T2, self.T2dash, self.D,self.chemical_shift, extra_shift_hz=0.0, species_id=0))

        if self.PD_fat is not None:
            chunks.append(flatten_branch(self.PD_fat, self.T1_fat, self.T2_fat, self.T2dash_fat, self.D_fat,
                                        self.chemical_shift, extra_shift_hz=self.fat_shift_hz, species_id=1))

        voxel_pos = torch.cat([c[0] for c in chunks], dim=0)
        PD = torch.cat([c[1] for c in chunks], dim=0)
        T1 = torch.cat([c[2] for c in chunks], dim=0)
        T2 = torch.cat([c[3] for c in chunks], dim=0)
        T2dash = torch.cat([c[4] for c in chunks], dim=0)
        D = torch.cat([c[5] for c in chunks], dim=0)
        B0 = torch.cat([c[6] for c in chunks], dim=0)
        B1 = torch.cat([c[7] for c in chunks], dim=1)
        coil_sens = torch.cat([c[8] for c in chunks], dim=1)
        species_id = torch.cat([c[9] for c in chunks], dim=0)

        if voxel_shape == "box":
            def dephasing_func(t, n): return sinc(t, 0.5 / n)
        elif voxel_shape == "sinc":
            def dephasing_func(t, n): return sigmoid(t, n)
        elif voxel_shape == "point":
            def dephasing_func(t, _): return identity(t)
        else:
            raise ValueError(f"Unsupported voxel shape '{voxel_shape}'")

        return SimData(
            PD,
            T1,
            T2,
            T2dash,
            D,
            B0,
            B1,
            coil_sens,
            self.size,
            voxel_pos,
            torch.as_tensor(shape, device=self.PD.device) / 2 / self.size,
            dephasing_func,
            recover_func=None,
            species_id=species_id,
            Masks = self.Masks if self.Masks else None,
            chemical_shift=self.chemical_shift,
        )
    '''
    # ---------------------------------------------------------------------
    # End MRTini extension: Separate Water and Fat based tissue handling with chemical shift
    # ---------------------------------------------------------------------
    '''
            
    def build_resample_z(self, *,
        PD_threshold: float = 1e-6,
        voxel_shape: Literal["sinc","box","point"]="box",
        Nz_out: int = 8,
        size_z_out: Optional[float] = None,
        z_range_m: Optional[Tuple[float,float]] = None,
    ):
        dev = self.PD.device
        Nx, Ny, Nz_in = self.PD.shape

        # source z grid (meters) consistent with your build()
        z_in_full = (self.size[2] * torch.fft.fftshift(torch.fft.fftfreq(Nz_in, device=dev))).to(self.size.dtype)

        # pick source volumes (possibly cropped)
        if z_range_m is not None:
            zmin, zmax = z_range_m
            keep = (z_in_full >= zmin) & (z_in_full <= zmax)
            if keep.sum() == 0:
                raise ValueError("z_range_m excludes all source planes.")
            z_in = z_in_full[keep]
            PD, T1, T2, T2d, D, B0 = (self.PD[:,:,keep], self.T1[:,:,keep], self.T2[:,:,keep],
                                    self.T2dash[:,:,keep], self.D[:,:,keep], self.B0[:,:,keep])
            B1, coil = self.B1[:,:,:,keep], self.coil_sens[:,:,:,keep]
        else:
            z_in = z_in_full
            PD, T1, T2, T2d, D, B0 = self.PD, self.T1, self.T2, self.T2dash, self.D, self.B0
            B1, coil = self.B1, self.coil_sens

        Nz_src = int(z_in.numel())

        # output z extent + grid
        if size_z_out is None:
            size_z_out = float(self.size[2].item())
        size_out = torch.as_tensor([self.size[0], self.size[1], size_z_out], device=dev, dtype=self.size.dtype)
        z_out = (size_out[2] * torch.fft.fftshift(torch.fft.fftfreq(Nz_out, device=dev))).to(self.size.dtype)

        # helper: repeat last axis to Nz_out
        def repeat_last(vol):
            # vol: (..., Nz_src) with Nz_src==1
            return vol.index_select(-1, torch.tensor([0], device=dev)).expand(*vol.shape[:-1], Nz_out)

        # helper: linear interp along last axis to Nz_out (Nz_src>=2)
        def lerp_last(vol):
            j1 = torch.searchsorted(z_in, z_out).clamp(1, Nz_src-1).to(torch.long)
            j0 = (j1 - 1).to(torch.long)
            z0, z1 = z_in[j0], z_in[j1]
            w1 = ((z_out - z0) / (z1 - z0 + 1e-12)).to(vol.real.dtype)  # (Nz_out,)
            v0 = vol.index_select(-1, j0)
            v1 = vol.index_select(-1, j1)
            wv = w1.view(*([1]*(vol.ndim-1)), -1).to(v0.dtype)
            return v0*(1-wv) + v1*wv

        # route: if only one source plane -> repeat; else interpolate
        def resample(vol):
            if Nz_src == 1: return repeat_last(vol)
            if vol.is_complex():
                vr = lerp_last(vol.real)
                vi = lerp_last(vol.imag)
                return vr.to(torch.complex64) + 1j*vi.to(torch.complex64)
            return lerp_last(vol)

        PDv, T1v, T2v, T2dv, Dv, B0v = map(resample, (PD, T1, T2, T2d, D, B0))
        B1v, coilv = resample(B1), resample(coil)  # works for real or complex

        # positions (same convention as build, but with z_out/Nz_out)
        x = size_out[0] * torch.fft.fftshift(torch.fft.fftfreq(Nx, device=dev))
        y = size_out[1] * torch.fft.fftshift(torch.fft.fftfreq(Ny, device=dev))
        pos_x, pos_y, pos_z = torch.meshgrid(x, y, z_out, indexing="ij")

        mask = PDv > PD_threshold
        voxel_pos = torch.stack([pos_x[mask], pos_y[mask], pos_z[mask]], dim=1)

        # dephasing + nyquist
        def sigmoid(t, n): return torch.prod(torch.sigmoid((n - t.abs() + 0.5) * 100), dim=1)
        def sinc(t, size): return torch.prod(torch.sinc(t * size), dim=1)
        def identity(t): return torch.ones_like(t[:,0])

        nyq = torch.as_tensor([Nx,Ny,Nz_out], device=dev, dtype=self.size.dtype) / 2 / size_out
        if voxel_shape == "box":
            dephasing_func = lambda t, n: sinc(t, 0.5/n)
        elif voxel_shape == "sinc":
            dephasing_func = lambda t, n: sigmoid(t, n)
        elif voxel_shape == "point":
            dephasing_func = lambda t, _: identity(t)
        else:
            raise ValueError(voxel_shape)

        return SimData(
            PDv[mask], T1v[mask], T2v[mask], T2dv[mask], Dv[mask], B0v[mask],
            B1v[:,mask], coilv[:,mask],
            size_out, voxel_pos, nyq, dephasing_func,
            recover_func=lambda dat: recover(mask, dat),
            multi_slice=True,
        ),mask
    
    @classmethod
    def brainweb(cls, file_name: str) -> VoxelGridPhantom:
        warn("brainweb() will be removed in a future version, use load() instead", DeprecationWarning)
        return cls.load(file_name=file_name)

    @classmethod
    def load(cls,file_name: Optional[str], data:Optional[dict]=None,
            FAT_PROPS_3T = dict(T1=0.37, T2=0.125, T2dash=0.012, D=0.1, chem_shift_hz=-412.72)
            ) -> VoxelGridPhantom:
        """Load a phantom from data produced by `generate_maps.py`."""
        Masks = {}
        if file_name is None and data is None:
            raise ValueError("Either file_name or data must be provided.")
        # Ensure fat-related variables are always defined
        PD_fat = T1_fat = T2_fat = T2dash_fat = D_fat = None
        fat_shift_hz = 0.0

        if data is not None:
            src = data
            getter = lambda k, default=None: src.get(k, default)
            PD = torch.tensor(getter('PD_map'))
            T1 = torch.tensor(getter('T1_map'))
            T2 = torch.tensor(getter('T2_map'))
            T2dash = torch.tensor(getter('T2dash_map'))
            D = torch.tensor(getter('D_map'))
            chem_shift =  torch.zeros_like(PD)  # Assuming water is the reference with 0 ppm shift
            B0_map = getter('B0_map')
            B1_map = getter('B1_map')
            if B0_map is not None and B1_map is not None:
                B0 = torch.tensor(B0_map)
                B1 = torch.tensor(B1_map)
            else:
                B0, B1 = generate_B0_B1(PD)
            try:
                size = torch.tensor(getter('FOV'), dtype=torch.float)
            except KeyError:
                size = torch.tensor([0.192, 0.192, 0.192])
            PD_fat = getter("PD_fat", None)
            if PD_fat is not None:
                PD_fat = torch.tensor(PD_fat, dtype=torch.float32)
                fat_mask = PD_fat > 0
                T1_fat = torch.where(fat_mask, torch.full_like(PD_fat, FAT_PROPS_3T["T1"]), torch.zeros_like(PD_fat))
                T2_fat = torch.where(fat_mask, torch.full_like(PD_fat, FAT_PROPS_3T["T2"]), torch.zeros_like(PD_fat))
                T2dash_fat = torch.where(fat_mask, torch.full_like(PD_fat, FAT_PROPS_3T["T2dash"]), torch.zeros_like(PD_fat))
                D_fat = torch.where(fat_mask, torch.full_like(PD_fat, FAT_PROPS_3T["D"]), torch.zeros_like(PD_fat))
                fat_shift_hz = FAT_PROPS_3T["chem_shift_hz"]
                chem_shift = torch.where(fat_mask, torch.full_like(PD_fat, fat_shift_hz), torch.zeros_like(PD_fat))
            Masks = {}
            try:
                for key in src.keys():
                    if key.startswith("tissue_"):
                        tissue = key[7:]
                        Masks[tissue] = torch.tensor(getter(key), dtype=torch.uint8)
            except Exception as e:
                print("No masks found or error occurred:", e)
        else:
            src = np.load(file_name)
            getter = lambda k, default=None: src[k] if k in src else default
        
            T1 = torch.tensor(getter('T1_map'))
            T2 = torch.tensor(getter('T2_map'))
            T2dash = torch.tensor(getter('T2dash_map'))
            PD = torch.tensor(getter('PD_map'))
            D = torch.tensor(getter('D_map'))
            chem_shift =  torch.zeros_like(PD)  # Assuming water is the reference with 0 ppm shift
            B0_map = getter('B0_map')
            B1_map = getter('B1_map')
            if B0_map is not None and B1_map is not None:
                B0 = torch.tensor(B0_map)
                B1 = torch.tensor(B1_map)
            else:
                B0, B1 = generate_B0_B1(PD)
            
            size = getter('FOV')
            if size is None:
                size = torch.tensor([0.192, 0.192, 0.192])
            else:
                size = torch.tensor(size, dtype=torch.float)
                
            PD_fat = getter("PD_fat", None)
            if PD_fat is not None:
                PD_fat = torch.tensor(PD_fat, dtype=torch.float32)
                fat_mask = PD_fat > 0

                # Rebuild intrinsic fat maps from constants, not weighted maps
                T1_fat = torch.where(fat_mask, torch.full_like(PD_fat, FAT_PROPS_3T["T1"]), torch.zeros_like(PD_fat))
                T2_fat = torch.where(fat_mask, torch.full_like(PD_fat, FAT_PROPS_3T["T2"]), torch.zeros_like(PD_fat))
                T2dash_fat = torch.where(fat_mask, torch.full_like(PD_fat, FAT_PROPS_3T["T2dash"]), torch.zeros_like(PD_fat))
                D_fat = torch.where(fat_mask, torch.full_like(PD_fat, FAT_PROPS_3T["D"]), torch.zeros_like(PD_fat))
                fat_shift_hz = FAT_PROPS_3T["chem_shift_hz"]
                chem_shift = torch.where(fat_mask, torch.full_like(PD_fat, fat_shift_hz), torch.zeros_like(PD_fat))

            Masks = {}
            try:
                for key in src.keys():
                    if key.startswith("tissue_"):
                        tissue = key[7:]
                        Masks[tissue] = torch.tensor(getter(key), dtype=torch.uint8)
            except Exception as e:
                print("No masks found or error occurred:", e)
            
        if B1.ndim == 3:
            # Add coil-dimension
            B1 = B1[None, ...]

        return cls(
            PD=PD, T1=T1, T2=T2, T2dash=T2dash, D=D, B0=B0, B1=B1,
            coil_sens=torch.ones(1, *PD.shape), size=size,
            PD_fat=PD_fat, T1_fat=T1_fat, T2_fat=T2_fat, T2dash_fat=T2dash_fat, D_fat=D_fat,
            fat_shift_hz=fat_shift_hz,
            chemical_shift=chem_shift,
            Masks = Masks if Masks != {} else None
        )

    @classmethod
    def load_mat(
        cls,
        file_name: str,
        T2dash: float | torch.Tensor = 0.03,
        D: float | torch.Tensor = 1.0,
        size = [0.2, 0.2, 8e-3]
    ) -> VoxelGridPhantom:
        """Load a :class:`VoxelGridPhantom` from a .mat file.

        The file must contain exactly one array, of which the last dimension
        must have size 5. This dimension is assumed to specify (in that order):

        * Proton density
        * T1
        * T2
        * B0
        * B1

        All data is per-voxel, multiple coils are not yet supported.
        Data will be normalized (see constructor).

        Parameters
        ----------
        file_name : str
            Name of the matlab .mat file to be loaded
        T2dash : float, optional
            T2dash value set uniformly for all voxels, by default 0.03
        T2dash : float, optional
            Diffusion value set uniformly for all voxels, by default 1

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing the loaded data.

        Raises
        ------
        Exception
            The loaded file does not contain the expected data.
        """
        data = _load_tensor_from_mat(file_name)

        if data.ndim < 2 or data.shape[-1] != 5:
            raise Exception(
                f"Expected a tensor with shape [..., 5], "
                f"but got {list(data.shape)}"
            )

        if data.ndim == 3:
            # Expand to 3D: [x, y, i] -> [x, y, z, i]
            data = data.unsqueeze(2)

        if isinstance(T2dash, float):
            T2dash = torch.full_like(data[..., 0], T2dash)
        if isinstance(D, float):
            D = torch.full_like(data[..., 0], D)

        return cls(
            data[..., 0],  # PD
            data[..., 1],  # T1
            data[..., 2],  # T2
            T2dash,
            D,
            data[..., 3],  # B0
            data[..., 4][None, ...],  # B1
            coil_sens=torch.ones(1, *data.shape[:-1]),
            size=torch.as_tensor(size),
        )

    def slices(self, slices: list[int]) -> VoxelGridPhantom:
        """Generate a copy that only contains the selected slice(s).

        Parameters
        ----------
        slice: int or tuple
            The selected slice(s)

        Returns
        -------
        SimData
            A new instance containing the selected slice(s).
        """
        assert 0 <= any([slices]) < self.PD.shape[2]

        def select(tensor: torch.Tensor):
            return tensor[..., slices].view(
                *list(self.PD.shape[:2]), len(slices)
            )

        return VoxelGridPhantom(
            select(self.PD),
            select(self.T1),
            select(self.T2),
            select(self.T2dash),
            select(self.D),
            select(self.B0),
            select(self.B1).unsqueeze(0),
            select(self.coil_sens).unsqueeze(0),
            self.size.clone(),
            self.repeat_count,
            select(self.PD_fat) if self.PD_fat is not None else None,
            select(self.T1_fat) if self.T1_fat is not None else None,
            select(self.T2_fat) if self.T2_fat is not None else None,
            select(self.T2dash_fat) if self.T2dash_fat is not None else None,
            select(self.D_fat) if self.D_fat is not None else None,
            fat_shift_hz=self.fat_shift_hz,
            chemical_shift=select(self.chemical_shift) if self.chemical_shift is not None else None,
            
             Masks= {key: mask[..., slices] for key, mask in self.Masks.items()} if self.Masks else None
            
        )

    def scale_fft(self, x: int, y: int, z: int) -> VoxelGridPhantom:
        """This is experimental, shows strong ringing and is not recommended"""
        # This function currently only supports downscaling
        assert x <= self.PD.shape[0]
        assert y <= self.PD.shape[1]
        assert z <= self.PD.shape[2]

        # Normalize signal, otherwise magnitude changes with scaling
        norm = (
            (x / self.PD.shape[0]) *
            (y / self.PD.shape[1]) *
            (z / self.PD.shape[2])
        )
        # Center for FT
        cx = self.PD.shape[0] // 2
        cy = self.PD.shape[1] // 2
        cz = self.PD.shape[2] // 2

        def scale(map: torch.Tensor) -> torch.Tensor:
            FT = torch.fft.fftshift(torch.fft.fftn(map))
            FT = FT[
                cx - x // 2:cx + (x+1) // 2,
                cy - y // 2:cy + (y+1) // 2,
                cz - z // 2:cz + (z+1) // 2
            ] * norm
            return torch.fft.ifftn(torch.fft.ifftshift(FT)).abs()

        return VoxelGridPhantom(
            scale(self.PD),
            scale(self.T1),
            scale(self.T2),
            scale(self.T2dash),
            scale(self.D),
            scale(self.B0),
            scale(self.B1.squeeze()).unsqueeze(0),
            scale(self.coil_sens.squeeze()).unsqueeze(0),
            self.size.clone(),
            Masks = {key: scale(mask) for key, mask in self.Masks.items()} if hasattr(self, "Masks") else None
        )

    def interpolate(self, x: int, y: int, z: int) -> VoxelGridPhantom:
        """Return a resized copy of this :class:`SimData` instance.

        This uses torch.nn.functional.interpolate in 'area' mode, which is not
        very good: Assumes pixels are squares -> has strong aliasing.

        Parameters
        ----------
        x : int
            The new resolution along the 1st dimension
        y : int
            The new resolution along the 2nd dimension
        z : int
            The new resolution along the 3rd dimension
        mode : str
            Algorithm used for upsampling (via torch.nn.functional.interpolate)

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing resized tensors.
        """
        def resample(tensor: torch.Tensor) -> torch.Tensor:
            # Introduce additional dimensions: mini-batch and channels
            return torch.nn.functional.interpolate(
                tensor[None, None, ...], size=(x, y, z), mode='area'
            )[0, 0, ...]
            
        def resample_chem_shift(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            # Instead of interpolating, just expand or repeat to the new shape
            # If tensor is already the correct shape, return as is
            if tensor.shape == (x, y, z):
             return tensor
            # If tensor is 3D, use nearest neighbor up/downsampling to avoid intermediate values
            return torch.nn.functional.interpolate(
                tensor[None, None, ...], size=(x, y, z), mode='nearest'
                )[0, 0, ...]
                
        def resample_multicoil(tensor: torch.Tensor) -> torch.Tensor:
            coils = tensor.shape[0]
            output = torch.zeros(coils, x, y, z)
            for i in range(coils):
                output[i, ...] = resample(tensor[i, ...])
            return output

        def resample_masks(tensors: Dict) -> Dict:
            output = {}
            for key, mask in tensors.items():
                # Interpolate the mask
                interpolated_mask = torch.nn.functional.interpolate(
                    mask[None, None, ...].float(), size=(x, y, z), mode='area'
                )[0, 0, ...]

                # Threshold the result to ensure binary output
                binary_mask = (interpolated_mask > 0.5).to(mask.dtype)

                # Store the result
                output[key] = binary_mask

            return output
        
        
        return VoxelGridPhantom(
            resample(self.PD),
            resample(self.T1),
            resample(self.T2),
            resample(self.T2dash),
            resample(self.D),
            resample(self.B0),
            resample_multicoil(self.B1),
            resample_multicoil(self.coil_sens),
            self.size.clone(),
            self.repeat_count,
            resample(self.PD_fat) if self.PD_fat is not None else None,
            resample(self.T1_fat) if self.T1_fat is not None else None,
            resample(self.T2_fat) if self.T2_fat is not None else None,
            resample(self.T2dash_fat) if self.T2dash_fat is not None else None,
            resample(self.D_fat) if self.D_fat is not None else None,
             fat_shift_hz=self.fat_shift_hz,
             chemical_shift=resample_chem_shift(self.chemical_shift) if self.chemical_shift is not None else None,
             Masks = resample_masks(self.Masks) if self.Masks else None,
        )


    def plot(self) -> None:
        """Print and plot all data stored in this phantom."""
        print("VoxelGridPhantom")
        print(f"size = {self.size}")
        # Center slice
        s = self.PD.shape[2] // 2
        # Warn if we only print a part of all data
        if self.coil_sens.shape[0] > 1:
            print(f"Plotting 1st of {self.coil_sens.shape[0]} coil sens maps")
        if self.B1.shape[0] > 1:
            print(f"Plotting 1st of {self.B1.shape[0]} B1 maps")
        if self.PD.shape[2] > 1:
            print(f"Plotting slice {s} / {self.PD.shape[2]}")

        plt.figure(figsize=(12, 10))
        plt.subplot(331)
        plt.title("PD")

        imshow(self.PD, vmin=0)
        plt.colorbar()
        plt.subplot(332)
        plt.title("T1")
        imshow(self.T1, vmin=0)
        plt.colorbar()
        plt.subplot(333)
        plt.title("T2")
        imshow(self.T2, vmin=0)
        plt.colorbar()
        plt.subplot(334)
        plt.title("T2'")
        imshow(self.T2dash, vmin=0)
        plt.colorbar()
        plt.subplot(335)
        plt.title("D")
        imshow(self.D, vmin=0)
        plt.colorbar()
        plt.subplot(337)
        plt.title("B0")
        imshow(self.B0)
        plt.colorbar()
        plt.subplot(338)
        plt.title("B1")
        imshow(self.B1[0, ...])
        plt.colorbar()
        plt.subplot(339)
        plt.title("coil sens")
        imshow(self.coil_sens[0, ...], vmin=0)
        plt.colorbar()
        plt.show()

    def plot_mask(self) -> None:
        """Print and plot all data stored in this phantom."""
        print("VoxelGridPhantom")
        print(f"size = {self.size}")
        # Center slice
        s = self.PD.shape[2] // 2
        # Warn if we only print a part of all data
        if self.coil_sens.shape[0] > 1:
            print(f"Plotting 1st of {self.coil_sens.shape[0]} coil sens maps")
        if self.B1.shape[0] > 1:
            print(f"Plotting 1st of {self.B1.shape[0]} B1 maps")
        if self.PD.shape[2] > 1:
            print(f"Plotting slice {s} / {self.PD.shape[2]}")

        plt.figure(figsize=(12, 10))
        plt.subplot(4,4,1)
        plt.title("PD")

        imshow(self.PD, vmin=0)
        plt.colorbar()
        plt.subplot(4,4,2)
        plt.title("T1")
        imshow(self.T1, vmin=0)
        plt.colorbar()
        plt.subplot(4,4,3)
        plt.title("T2")
        imshow(self.T2, vmin=0)
        plt.colorbar()
        plt.subplot(4,4,4)
        plt.title("T2'")
        imshow(self.T2dash, vmin=0)
        plt.colorbar()
        plt.subplot(4,4,5)
        plt.title("D")
        imshow(self.D, vmin=0)
        plt.colorbar()
        plt.subplot(4,4,6)
        plt.title("B0")
        imshow(self.B0)
        plt.colorbar()
        plt.subplot(4,4,7)
        plt.title("B1")
        imshow(self.B1[0, ...])
        plt.colorbar()
        plt.subplot(4,4,8)
        plt.title("coil sens")
        imshow(self.coil_sens[0, ...], vmin=0)
        plt.colorbar()
        if self.PD_fat is not None:
            plt.subplot(4,4,9)
            plt.title("PD fat")

            imshow(self.PD_fat, vmin=0)
            plt.colorbar()
            plt.subplot(4,4,10)
            plt.title("T1 fat")
            imshow(self.T1_fat, vmin=0)
            plt.colorbar()
            plt.subplot(4,4,11)
            plt.title("T2 fat")
            imshow(self.T2_fat, vmin=0)
            plt.colorbar()
            plt.subplot(4,4,12)
            plt.title("D fat")
            imshow(self.D_fat, vmin=0)
            plt.colorbar()
        plt.tight_layout()            
        plt.show()


def recover(mask, sim_data: SimData) -> VoxelGridPhantom:
    """Provided to :class:`SimData` to reverse the ``build()``"""
    def to_full(sparse):
        assert sparse.ndim < 3
        if sparse.ndim == 2:
            full = torch.zeros(
                [sparse.shape[0], *mask.shape], dtype=sparse.dtype)
            full[:, mask] = sparse.cpu()
        else:
            full = torch.zeros(mask.shape)
            full[mask] = sparse.cpu()
        return full

    return VoxelGridPhantom(
        to_full(sim_data.PD),
        to_full(sim_data.T1),
        to_full(sim_data.T2),
        to_full(sim_data.T2dash),
        to_full(sim_data.D),
        to_full(sim_data.B0),
        to_full(sim_data.B1),
        to_full(sim_data.coil_sens),
        sim_data.size
    )


def _load_tensor_from_mat(file_name: str) -> torch.Tensor:
    mat = io.loadmat(file_name)

    keys = [
        key for key in mat
        if not (key.startswith('__') and key.endswith('__'))
    ]

    arrays = [mat[key] for key in keys if isinstance(mat[key], np.ndarray)]

    if len(keys) == 0:
        raise Exception("The loaded mat file does not contain any variables")

    if len(arrays) != 1:
        raise Exception("The loaded mat file must contain exactly one array")

    return torch.from_numpy(arrays[0]).float()
