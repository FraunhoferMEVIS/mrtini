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
from typing import Dict
from time import time
import torch
import numpy as np
from enum import Enum
from typing import Iterable
import matplotlib.pyplot as plt
import mrstar
import copy


def _copy_sequence_metadata(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_sequence_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_sequence_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_sequence_metadata(item) for item in value)
    return copy.deepcopy(value)
class PulseUsage(Enum):
    """:class:`Enum` of all pulse usages, needed for reconstruction.

    The simulation always simulates all magnetization pathways and will ignore
    the pulse usage. It is only used for reconstruction, by the
    :meth:`Sequence.get_kspace` method, to understand the role of a pulse.
    Additionally, Pulseq exportes might use the pulse usage to select the
    correct pulse type.

    Attributes
    ----------
    UNDEF : str
        ``"undefined"``: No specified use case.
    EXIT : str
        ``"excitation"``: Will set the kspace position back to zero
        or the position stored by the last `STORE` pulse.
    REFOC : str
        ``"refocussing"``: Mirrors the kspace position.
    STORE : str
        ``"storing"``: Stores the current kspace position.
        Can be used for DREAM-like sequences.
    FATSAT : str
        ``"fatsaturation"``: Not handled differently by the simulation, but
        can be used by Pulseq exporters to emit a fat-saturation pulse.
    """

    UNDEF = "undefined"
    EXCIT = "excitation"
    REFOC = "refocussing"
    STORE = "storing"
    FATSAT = "fatsaturation"


class Pulse:
    """Contains the definition of an instantaneous RF Pulse.

    Attributes
    ----------
    usage : PulseUsage
        Specifies how this pulse is used, needed only for reconstruction
    angle : torch.Tensor
        Flip angle in radians
    phase : torch.Tensor
        Pulse phase in radians
    shim_array : torch.Tensor
        Contains B1 mag and phase, used for pTx. 2D tensor([[1, 0]]) for 1Tx.
    selective : bool
        Specifies if this pulse should be slice-selective (z-direction)
    """

    def __init__(
        self,
        usage: PulseUsage,
        angle: torch.Tensor,
        phase: torch.Tensor,
        shim_array: torch.Tensor,
        selective: bool,
    ):
        """Create a Pulse instance."""
        self.usage = usage
        self.angle = angle
        self.phase = phase
        self.shim_array = shim_array
        self.selective = selective

    def cpu(self) -> Pulse:
        """Move this pulse to the CPU and return it."""
        return Pulse(
            self.usage,
            torch.as_tensor(self.angle, dtype=torch.float32).cpu(),
            torch.as_tensor(self.phase, dtype=torch.float32).cpu(),
            torch.as_tensor(self.shim_array, dtype=torch.float32).cpu(),
            self.selective
        )

    def cuda(self, device: int | None = None) -> Pulse:
        """Move this pulse to the specified CUDA device and return it."""
        return Pulse(
            self.usage,
            torch.as_tensor(self.angle, dtype=torch.float32).cuda(device),
            torch.as_tensor(self.phase, dtype=torch.float32).cuda(device),
            torch.as_tensor(self.shim_array, dtype=torch.float32).cuda(device),
            self.selective
        )

    @property
    def device(self) -> torch.device:
        """Return the device this pulse is stored on."""
        return self.angle.device

    @classmethod
    def zero(cls):
        """Create a pulse with zero flip and phase."""
        return cls(
            PulseUsage.UNDEF,
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            torch.asarray([[1, 0]], dtype=torch.float32),
            True
        )

    def clone(self) -> Pulse:
        """Return a cloned copy of self."""
        return Pulse(
            self.usage,
            self.angle.clone(),
            self.phase.clone(),
            self.shim_array.clone(),
            self.selective
        )


class Repetition:
    """A :class:`Repetition` starts with a RF pulse and ends before the next.

    Attributes
    ----------
    pulse : Pulse
        The RF pulse at the beginning of this :class:`Repetition`
    event_time : torch.Tensor
        Duration of each event (seconds)
    gradm : torch.Tensor
        Gradient moment of every event, shape (:attr:`event_count`, 3)
    adc_phase : torch.Tensor
        Float tensor describing the adc phase, shape (:attr:`event_count`, 3)
    adc_usage: torch.Tensor
        Int tensor specifying which contrast a sample belongs to, shape
        (:attr:`event_count`, 3). Samples with ```adc_usage <= 0``` will not be
        measured. For single contrast sequences, just use 0 or 1.
    event_count : int
        Number of events in this :class:`Repetition`
    """

    def __init__(
        self,
        pulse: Pulse,
        event_time: torch.Tensor,
        gradm: torch.Tensor,
        adc_phase: torch.Tensor,
        adc_usage: torch.Tensor,
        raw_waveform: Dict,
        gradm_z_eff: torch.Tensor | None = None,
    ):
        """Create a repetition based on the given tensors.

        Raises
        ------
        ValueError
            If not all tensors have the same shape or have zero elements.
        """
        if event_time.numel() == 0:
            raise ValueError("Can't create a repetition with zero elements")

        self.pulse = pulse
        self.event_count = event_time.numel()

        if event_time.shape != torch.Size([self.event_count]):
            raise ValueError(
                f"Wrong event_time shape {tuple(event_time.shape)}, "
                f"expected {(self.event_count, )}"
            )
        if gradm.shape != torch.Size([self.event_count, 3]):
            raise ValueError(
                f"Wrong gradm shape {tuple(gradm.shape)}, "
                f"expected {(self.event_count, 3)}"
            )
        if adc_phase.shape != torch.Size([self.event_count]):
            raise ValueError(
                f"Wrong adc_phase shape {tuple(adc_phase.shape)}, "
                f"expected {(self.event_count, )}"
            )
        if adc_usage.shape != torch.Size([self.event_count]):
            raise ValueError(
                f"Wrong adc_usage shape {tuple(adc_usage.shape)}, "
                f"expected {(self.event_count, )}"
            )

        self.event_time = event_time
        self.gradm = gradm
        self.gradm_z_eff = gradm_z_eff if gradm_z_eff is not None else gradm[:, 2].clone()  # use effective gradient if available
        self.adc_phase = adc_phase
        self.adc_usage = adc_usage
        self.raw_waveform = raw_waveform

    def cuda(self, device: int | None = None) -> Repetition:
        return Repetition(
            pulse=self.pulse.cuda(device), event_time=self.event_time.cuda(device),
            gradm=self.gradm.cuda(device), adc_phase=self.adc_phase.cuda(device),
            adc_usage=self.adc_usage.cuda(device),
            raw_waveform=_copy_sequence_metadata(self.raw_waveform),
            gradm_z_eff=self.gradm_z_eff.cuda(device),
        )

    def cpu(self) -> Repetition:
        return Repetition(
            pulse=self.pulse.cpu(), event_time=self.event_time.cpu(),
            gradm=self.gradm.cpu(), adc_phase=self.adc_phase.cpu(),
            adc_usage=self.adc_usage.cpu(),
            raw_waveform=_copy_sequence_metadata(self.raw_waveform),
            gradm_z_eff=self.gradm_z_eff.cpu(),
        )


    @property
    def device(self) -> torch.device:
        """Return the repetition this pulse is stored on."""
        return self.gradm.device

    @classmethod
    def zero(cls, event_count: int) -> Repetition:
        """Create a ``Repetition`` instance with everything set to zero.

        Parameters
        ----------
        event_count : int
            Number of events in the new :class:`Repetition`.

        Returns
        -------
        :class:`Repetition`
            A :class:`Repetition` that is part of this :class:`Sequence`.
        """
        return cls(
            Pulse.zero(),
            torch.zeros(event_count, dtype=torch.float32),
            torch.zeros((event_count, 3), dtype=torch.float32),
            torch.zeros(event_count, dtype=torch.float32),
            torch.zeros(event_count, dtype=torch.int32),
            {},
            
        )

    def clone(self) -> Repetition:
            return Repetition(
                pulse=self.pulse.clone(), event_time=self.event_time.clone(),
                gradm=self.gradm.clone(), adc_phase=self.adc_phase.clone(),
                adc_usage=self.adc_usage.clone(),
                raw_waveform=_copy_sequence_metadata(self.raw_waveform),
                gradm_z_eff=self.gradm_z_eff.clone(),
            )

    def get_contrasts(self) -> list[int]:
        """Return a sorted list of contrasts used by this ``Repetition``."""
        return sorted(self.adc_usage[self.adc_usage > 0].unique().tolist())

    def shift_contrasts(self, offset: int):
        """Increment all contrasts used by this repetition by ``offset``.

        Only operates on elements that are already larger than zero, so this
        function does not change which elements are measured.
        """
        self.adc_usage[self.adc_usage > 0] += offset


class Sequence(list):
    """Defines a MRI sequence.

    This extends a standard python list and inherits all its functions. It
    additionally implements MRI sequence specific methods.
    """
    
    def __init__(self, repetitions: Iterable[Repetition] | None = None,
                 normalized_grads: bool = True,
                 mrtini_hybrid: list[dict] | None = None,
                 mrtini_rep_bounds: list[tuple] | None = None,
                 mrtini_spans_by_rep: list[list[int]] | None = None):
        """Create a ``Sequence`` instance by passing repetitions.

        Parameters
        ----------
        repetitions
            Initialize this Sequence directly with a list of repetitions
        normalized_grads : bool
            When this sequence is loaded from pulseq, this is set to `False`.
            The default of `True` flags this sequence for using normalized
            k-values, which is what is usually desired when building the
            sequence in mr0, using gradient steps of 1. If true, these
            gradients are then scaled to the phantom size on simulation.
            If false, no scaling happens and SI units are assumed.
        """
        super().__init__([] if repetitions is None else repetitions)
        self.normalized_grads = normalized_grads
        self.mrtini_hybrid = [] if mrtini_hybrid is None else mrtini_hybrid
        self.mrtini_rep_bounds = [] if mrtini_rep_bounds is None else mrtini_rep_bounds
        self.mrtini_spans_by_rep = [] if mrtini_spans_by_rep is None else mrtini_spans_by_rep
        
    def _copy_with_repetitions(self, repetitions):
        result = Sequence(repetitions, normalized_grads=self.normalized_grads)
        # Preserve imported and dynamically attached MRtini metadata, not just
        # the constructor's three hybrid fields.
        for name, value in self.__dict__.items():
            setattr(result, name, _copy_sequence_metadata(value))
        return result
    
    def cuda(self) -> Sequence:
        return self._copy_with_repetitions([rep.cuda() for rep in self])

    def cpu(self) -> Sequence:
        return self._copy_with_repetitions([rep.cpu() for rep in self])

    @property
    def device(self) -> torch.device:
        """Return the sequence this pulse is stored on."""
        return self[0].device

    def clone(self) -> Sequence:
        return self._copy_with_repetitions([rep.clone() for rep in self])

    def new_rep(self, event_count) -> Repetition:
        """Return a zeroed out repetition that is part of this ``Sequence``."""
        rep = Repetition.zero(event_count)
        self.append(rep)
        return rep

    def get_full_kspace(self) -> list[torch.Tensor]:
        """Compute the kspace trajectory produced by the gradient moments.

        This function relies on the values of ``Repetition.pulse_usage`` to
        determine which trajectory the sequence tries to achieve.

        The trajectory is 4-dimensional as it also includes dephasing time.

        Returns
        -------
        list[torch.Tensor]
            A tensor of shape (``event_count``, 4) for every repetition.
        """
        k_pos = torch.zeros(4, device=self.device)
        trajectory = []
        # Pulses with usage STORE store magnetisation and update this variable,
        # following excitation pulses will reset to stored instead of origin
        stored = torch.zeros(4, device=self.device)

        for rep in self:
            if rep.pulse.usage == PulseUsage.EXCIT:
                k_pos = stored
            elif rep.pulse.usage == PulseUsage.REFOC:
                k_pos = -k_pos
            elif rep.pulse.usage == PulseUsage.STORE:
                stored = k_pos

            rep_traj = k_pos + torch.cumsum(
                torch.cat([rep.gradm, rep.event_time[:, None]], 1),
                dim=0
            )
            k_pos = rep_traj[-1, :]
            trajectory.append(rep_traj)

        return trajectory

    def get_kspace(self) -> torch.Tensor:
        """Calculate the trajectory described by the signal of this sequence.

        This function returns only the kspace positions of the events that were
        actually measured (i.e. ``adc_usage > 0``) as one continuous tensor.
        The kspace includes the dephasing time as 4th dimension.

        Returns
        -------
        torch.Tensor
            Float tensor of shape (sample_count, 4)
        """
        # - Iterate over the full kspace and the sequence repetitions
        # - Mask the kspace to only retain samples that were measured
        # - Concatenate all repetitions and return the result
        return torch.cat([
            shot[rep.adc_usage > 0]
            for shot, rep in zip(self.get_full_kspace(), self)
        ])

    def get_contrast_mask(self, contrast: int) -> torch.Tensor:
        """Return a mask for a specific contrast as bool tensor.

        The returned tensor only contains measured events and is designed to be
        used together with ``get_kspace()`` or the simulated signal.

        Parameters
        ----------
        contrast : int
            The index for the contrast of which the signal mask is requested.

        Returns
        -------
        torch.Tensor
            The signal mask for the requested contrast as bool tensor

        Examples
        --------
        >>> signal = execute_graph(graph, seq, data)
        >>> kspace = seq.get_kspace()
        >>> mask = seq.get_contrast_mask(7)
        >>> contrast_reco = reco(signal[mask], kspace[mask])
        """
        return torch.cat(
            [rep.adc_usage[rep.adc_usage > 0] == contrast for rep in self]
        )

    def get_contrasts(self) -> list[int]:
        """Return a sorted list of all contrasts used by this ``Sequence``."""
        # flat list of all contrasts of all sequences
        tmp = [c for rep in self for c in rep.get_contrasts()]
        # Use a set to remove duplicates
        return sorted(list(set(tmp)))

    def shift_contrasts(self, offset: int):
        """Increment all offsets used by this sequence by ``offset``.

        Only operates on elements that are already larger than zero, so this
        function does not change which elements are measured. Modifies the
        sequence in-place, use :meth:`clone()` if you want to keep the original
        sequence as well.
        """
        for rep in self:
            rep.shift_contrasts(offset)

    def get_duration(self) -> float:
        """Calculate the total duration of self in seconds."""
        return sum(rep.event_time.sum().item() for rep in self)

    @classmethod
    def import_file(cls, file_name: str,
                    exact_trajectories: bool = True,
                    print_stats: bool = False
                    ) -> Sequence:
        """Import a pulseq .seq file.

        Parameters
        ----------
        file_name : str
            The path to the file that is imported
        exact_trajectories : bool
            If true, the gradients before and after the ADC blocks are imported
            exactly. If false, they are summed into a single event. Depending
            on the sequence, simulation might be faster if set to false, but
            the simulated diffusion changes with simplified trajectoreis.
        print_stats : bool
            If set to true, additional information is printed during import

        Returns
        -------
        mr0.Sequence
            The imported file as mr0 Sequence
        """
        start = time()
        # parser = pydisseqt.load_pulseq(file_name)
        parser = mrstar.load_gstar(file_name)
        # parser = pygstar.load_gstar(file_name)
        if print_stats:
            print(f"Importing the .seq file took {time() - start} s")
        start = time()
        seq = cls(normalized_grads=False)

        # We should do at least _some_ guess for the pulse usage
        def pulse_usage(angle: float) -> PulseUsage:
            if abs(angle) < 100 * np.pi / 180:
                return PulseUsage.EXCIT
            else:
                return PulseUsage.REFOC

        # Get time points of all pulses
        pulses = []  # Contains pairs of (pulse_start, pulse_end)
        tmp = parser.encounter("rf", 0.0)
        spans = parser.hybrid(150e-6)
        while tmp is not None:
            pulses.append(tmp)
            tmp = parser.encounter("rf", tmp[1])  # pulse_end

        # Iterate over all repetitions (we ignore stuff before the first pulse)
        for i in range(len(pulses)):
            # Calculate repetition start and end time based on pulse centers
            rep_start = (pulses[i][0] + pulses[i][1]) / 2
            if i + 1 < len(pulses):
                rep_end = (pulses[i + 1][0] + pulses[i + 1][1]) / 2
            else:
                rep_end = parser.duration()

            # Fetch additional data needed for building the mr0 sequence
            pulse = parser.integrate_one(pulses[i][0], pulses[i][1]).pulse

            adcs = parser.events("adc", rep_start, rep_end)

            # To simulate diffusion, we want to more exactly simulate gradient
            # trajectories between pulses and the ADC block
            if exact_trajectories:
                # First and last timepoint in repetition with a gradient sample
                first = pulses[i][1]
                last = (pulses[i + 1][0] if i + 1 < len(pulses) else rep_end)
                eps = 1e-6  # Move a bit past start / end of repetition
                # Gradient samples can be duplicated between x, y, z.
                # They are deduplicated after rounding to `precision` digits
                precision = 6

                if len(adcs) > 0:
                    grad_before = sorted(set([round(t, precision) for t in (
                        parser.events("grad x", first + eps, adcs[0] - eps) +
                        parser.events("grad y", first + eps, adcs[0] - eps) +
                        parser.events("grad z", first + eps, adcs[0] - eps)
                    )]))
                    grad_after = sorted(set([round(t, precision) for t in (
                        parser.events("grad x", adcs[-1] + eps, last - eps) +
                        parser.events("grad y", adcs[-1] + eps, last - eps) +
                        parser.events("grad z", adcs[-1] + eps, last - eps)
                    )]))
                    # Last repetition: no pulse, ignore [last, rep_end]
                    if i == len(pulses) - 1:
                        abs_times = [rep_start, first] + grad_before + adcs
                    else:
                        abs_times = ([rep_start, first] + grad_before + adcs +
                                     grad_after + [last, rep_end])
                    # Index of first ADC: -1 - we count spans between indices
                    adc_start = 2 + len(grad_before) - 1
                else:
                    grad = sorted(set([round(t, precision) for t in (
                        parser.events("grad x", first + eps, last - eps) +
                        parser.events("grad y", first + eps, last - eps) +
                        parser.events("grad z", first + eps, last - eps)
                    )]))
                    # Last repetition: no pulse, ignore [last, rep_end]
                    if i == len(pulses) - 1:
                        abs_times = [rep_start, first] + grad
                    else:
                        abs_times = [rep_start, first] + grad + [last, rep_end]
                    adc_start = None
            else:
                # No gradient samples, only adc and one final to the next pulse
                abs_times = [rep_start] + adcs + [rep_end]
                adc_start = 0

            event_count = len(abs_times) - 1
            samples = parser.sample(adcs)
            moments = parser.integrate(abs_times)

            if print_stats:
                print(
                    f"Rep. {i + 1}: {event_count} samples, of which "
                    f"{len(adcs)} are ADC (starting at {adc_start})"
                )

            # -- Now we build the mr0 Sequence repetition --

            rep = seq.new_rep(event_count)
            rep.pulse.angle = pulse.angle
            rep.pulse.phase = pulse.phase
            rep.pulse.usage = pulse_usage(pulse.angle)

            rep.event_time[:] = torch.as_tensor(np.diff(abs_times))

            rep.gradm[:, 0] = torch.as_tensor(moments.gradient.x)
            rep.gradm[:, 1] = torch.as_tensor(moments.gradient.y)
            rep.gradm[:, 2] = torch.as_tensor(moments.gradient.z)

            if adc_start is not None:
                phases = np.pi / 2 - torch.as_tensor(samples.adc.phase)
                rep.adc_usage[adc_start:adc_start + len(adcs)] = 1
                rep.adc_phase[adc_start:adc_start + len(adcs)] = phases

        if print_stats:
            print(f"Converting the sequence to mr0 took {time() - start} s")
        return seq

    @classmethod
    def from_seq_file(cls, file_name: str) -> Sequence:
        """Import a sequence from a pulseq .seq file.

        # This function is deprecated, use `Sequence.import_file` instead

        The importer currently minimizes the amount of used `mr0` sequence
        events. This can be problematic for diffusion weighted sequences,
        because gradients that are not directly measured by ADC samples can be
        removed from the sequence, even if they are important for the targeted
        contrast.

        Parameters
        ----------
        file_name : str
            Path to the imported .seq file

        Returns
        -------
        Sequence
            Imported sequence, converted to MRzero
        """
        print(
            "WARNING: Use of deprecated Sequence.from_seq_file,"
            "use Sequence.import_file instead"
        )
        seq = Sequence(normalized_grads=False)
        for tmp_rep in intermediate(PulseqFile(file_name)):
            rep = seq.new_rep(tmp_rep[0])
            rep.pulse.angle = torch.as_tensor(
                tmp_rep[1].angle, dtype=torch.float32)
            rep.pulse.phase = torch.as_tensor(
                tmp_rep[1].phase, dtype=torch.float32)
            rep.pulse.shim_array = torch.as_tensor(
                tmp_rep[1].shim_array, dtype=torch.float32)

            if rep.pulse.angle > 100 * torch.pi/180:
                rep.pulse.usage = PulseUsage.REFOC
            else:
                rep.pulse.usage = PulseUsage.EXCIT

            # to combine multiple blocks into one repetition, we need to keep
            # track of the current event offset
            i = 0
            for block in tmp_rep[2]:
                if isinstance(block, Spoiler):
                    rep.event_time[i] = block.duration
                    rep.gradm[i, :] = torch.tensor(block.gradm)
                    i += 1
                else:
                    assert isinstance(block, Adc)
                    num = len(block.event_time)
                    rep.event_time[i:i+num] = torch.tensor(block.event_time)
                    rep.gradm[i:i+num, :] = torch.tensor(block.gradm)
                    rep.adc_phase[i:i+num] = torch.pi/2 - block.phase
                    rep.adc_usage[i:i+num] = 1
                    i += num
            assert i == tmp_rep[0]
        return seq

    @staticmethod
    def _holes_in_interval(t0: float, t1: float, bloch_windows: list[tuple[float, float]]):
        """Return the list of [a,b] holes (Bloch spans) intersecting [t0,t1]."""
        holes = []
        for a, b in bloch_windows:
            if b <= t0 or a >= t1:
                continue
            holes.append((max(t0, a), min(t1, b)))
        return holes

    @staticmethod
    def _build_epg_soa_from_rep_slice(
        *,
        t0: float, t1: float,
        rep_abs_times: list[float],   # len = N+1
        gm_x: list[float],            # len = N
        gm_y: list[float],
        gm_z: list[float],
        adc_start: int | None, adc_count: int,
        adc_phase: list[float] | None,
        holes: list[tuple[float, float]],
    ):
        import numpy as np
        import torch

        starts = np.asarray(rep_abs_times[:-1], dtype=np.float64)  # (N,)
        ends   = np.asarray(rep_abs_times[1:],  dtype=np.float64)  # (N,)

        # --- 1) pick events that overlap [t0, t1] (not only fully inside) ---
        overlap = (ends > t0) & (starts < t1)
        if not np.any(overlap):
            # empty EPG span
            return dict(
                time=torch.zeros(0, dtype=torch.float32),
                gradm=torch.zeros(3, 0, dtype=torch.float32),
                adc_mask=torch.zeros(0, dtype=torch.bool),
                adc_phase=torch.zeros(0, dtype=torch.float32),
            )

        starts_sel = starts[overlap]      # full-event start times
        ends_sel   = ends[overlap]        # full-event end times
        dt_full    = ends_sel - starts_sel  # full-event durations

        # --- 2) clip to [t0, t1] at the boundaries ---
        s_clip = np.maximum(starts_sel, t0)
        e_clip = np.minimum(ends_sel, t1)
        dt_clip = e_clip - s_clip

        # guard against numerical issues
        keep = dt_clip > 0.0
        if not np.any(keep):
            return dict(
                time=torch.zeros(0, dtype=torch.float32),
                gradm=torch.zeros(3, 0, dtype=torch.float32),
                adc_mask=torch.zeros(0, dtype=torch.bool),
                adc_phase=torch.zeros(0, dtype=torch.float32),
            )

        s_clip = s_clip[keep]
        e_clip = e_clip[keep]
        dt_clip = dt_clip[keep]
        dt_full = dt_full[keep]

        # which original event indices did we keep?
        idx_all = np.nonzero(overlap)[0][keep]   # indices into original gm_x/y/z

        # --- 3) gradient moments: scale full-event moment by dt_clip / dt_full ---
        gm_full = np.stack([
            np.asarray(gm_x, np.float64),
            np.asarray(gm_y, np.float64),
            np.asarray(gm_z, np.float64),
        ], axis=0)[:, idx_all]   # (3, K)

        scale = np.ones_like(dt_clip, dtype=np.float64)
        nz = dt_full > 0
        scale[nz] = dt_clip[nz] / dt_full[nz]
        gm = gm_full * scale[None, :]      # (3, K)

        # now these are the per-event times we feed into the rest of the function
        s_sel = s_clip
        e_sel = e_clip
        dt_sel = dt_clip

        # adc arrays for the selected events
        K = int(dt_sel.shape[0])
        adc_mask = np.zeros(K, dtype=bool)
        adc_phi  = np.zeros(K, dtype=np.float64)
        if adc_start is not None and adc_count > 0 and adc_phase is not None:
            a0, a1 = adc_start, adc_start + adc_count
            # idx_all already holds the original event indices for [0..K)
            for j_local, j_abs in enumerate(idx_all):
                if a0 <= j_abs < a1:
                    adc_mask[j_local] = True
                    adc_phi[j_local]  = float(adc_phase[j_abs - a0])

        # --- the rest of your original function stays the same from here on ---
        # replace your old s_sel/e_sel/dt_sel/gm/adc_mask/adc_phi setup
        # and keep the holes-splitting logic as-is, e.g.:

        if len(holes) == 0:
            return dict(
                time=torch.as_tensor(dt_sel, dtype=torch.float32),
                gradm=torch.as_tensor(gm,     dtype=torch.float32),
                adc_mask=torch.as_tensor(adc_mask, dtype=torch.bool),
                adc_phase=torch.as_tensor(adc_phi,  dtype=torch.float32),
            )
        # otherwise, subdivide each [s,e] by hole boundaries; keep time, zero gradm in holes
        out_dt: list[float] = []
        out_gx: list[float] = []
        out_gy: list[float] = []
        out_gz: list[float] = []
        out_adc: list[bool] = []
        out_phi: list[float] = []

        # precompute all hole boundaries
        H = [(a, b) for (a, b) in holes if b > a]
        for k in range(K):
            s, e = float(s_sel[k]), float(e_sel[k])
            if not (e > s):
                continue
            # boundaries inside (s,e)
            cuts = [s, e]
            for (a, b) in H:
                if a > s and a < e: cuts.append(a)
                if b > s and b < e: cuts.append(b)
            cuts = sorted(set(cuts))

            # split current event
            dt_k = e - s
            gx_k, gy_k, gz_k = gm[0, k], gm[1, k], gm[2, k]

            for u, v in zip(cuts[:-1], cuts[1:]):
                dur = v - u
                if dur <= 0.0:
                    continue
                # inside any hole?
                inside_hole = any((u >= a and v <= b) for (a, b) in H)

                if inside_hole:
                    # keep time, zero grad moment, no ADC
                    out_dt.append(dur)
                    out_gx.append(0.0); out_gy.append(0.0); out_gz.append(0.0)
                    out_adc.append(False)
                    out_phi.append(0.0)
                else:
                    # proportionally split the parent's moment by duration
                    scale = dur / dt_k if dt_k > 0 else 0.0
                    out_dt.append(dur)
                    out_gx.append(gx_k * scale)
                    out_gy.append(gy_k * scale)
                    out_gz.append(gz_k * scale)
                    out_adc.append(bool(adc_mask[k]))
                    out_phi.append(float(adc_phi[k]) if adc_mask[k] else 0.0)

        # assemble SoA tensors
        time_t = torch.as_tensor(out_dt, dtype=torch.float32)
        gradm_t = torch.stack([
            torch.as_tensor(out_gx, dtype=torch.float32),
            torch.as_tensor(out_gy, dtype=torch.float32),
            torch.as_tensor(out_gz, dtype=torch.float32),
        ], dim=0)  # (3,E)
        adc_mask_t = torch.as_tensor(out_adc, dtype=torch.bool)
        adc_phase_t = torch.as_tensor(out_phi, dtype=torch.float32)

        return dict(time=time_t, gradm=gradm_t, adc_mask=adc_mask_t, adc_phase=adc_phase_t)
    @staticmethod
    def soa_collapse_keep_adcs(soa):
        """
        Collapse exterior non-ADC events; preserve every event between first/last ADC.
        Input  soa: dict with tensors:
            time:(E,), gradm:(3,E), adc_mask:(E,), adc_phase:(E,)
        Output: same keys. Interior non-ADC gaps remain non-ADC events.
        """

        t   = soa["time"]        # (E,)
        G   = soa["gradm"]       # (3,E)
        m   = soa["adc_mask"]    # (E,)
        ph  = soa["adc_phase"]   # (E,)
        E = int(t.numel())
        idx = torch.nonzero(m, as_tuple=False).flatten()

        # no ADC at all → one lump
        if idx.numel() == 0:
            return dict(
                time      = t.sum().reshape(1),
                gradm     = G.sum(dim=1, keepdim=True),
                adc_mask  = torch.tensor([False], dtype=torch.bool, device=t.device),
                adc_phase = torch.tensor([0.0],  dtype=t.dtype, device=t.device),
            )

        lo, hi = int(idx[0]), int(idx[-1])

        out_t   = []
        out_G   = []
        out_m   = []
        out_ph  = []

        # pre lump
        if lo > 0:
            out_t.append(t[:lo].sum().reshape(1))
            out_G.append(G[:, :lo].sum(dim=1, keepdim=True))
            out_m.append(torch.zeros(1, dtype=torch.bool, device=t.device))
            out_ph.append(torch.zeros(1, dtype=t.dtype, device=t.device))

        # ADCs 1:1
        out_t.append(t[lo:hi+1])
        out_G.append(G[:, lo:hi+1])
        out_m.append(m[lo:hi+1])
        out_ph.append(ph[lo:hi+1])
        # post lump
        if hi + 1 < E:
            out_t.append(t[hi+1:].sum().reshape(1))
            out_G.append(G[:, hi+1:].sum(dim=1, keepdim=True))
            out_m.append(torch.zeros(1, dtype=torch.bool, device=t.device))
            out_ph.append(torch.zeros(1, dtype=t.dtype, device=t.device))
        return dict(
            time      = torch.cat(out_t, dim=0),
            gradm     = torch.cat(out_G, dim=1),
            adc_mask  = torch.cat(out_m, dim=0),
            adc_phase = torch.cat(out_ph, dim=0),
        )

    @staticmethod
    def _preceding_bloch_pad(t0: float,
                            bloch_windows: list[tuple[float, float]],
                            tol: float = 1e-9) -> float:
        """
        Return total duration of Bloch windows that end at t0 (allowing tiny tol),
        chaining backwards if multiple Bloch windows are contiguous.
        """
        if not bloch_windows:
            return 0.0
        # sort by end time
        bw = sorted(bloch_windows, key=lambda ab: ab[1])
        pad = 0.0
        cur_end = t0
        # walk backwards over windows that butt against the current end
        for a, b in reversed(bw):
            if abs(b - cur_end) <= tol and b > a:
                pad += (b - a)
                cur_end = a  # chain further back if previous Bloch abuts this one
        return pad

    @staticmethod
    def _prepend_relax_only(events_soa):
        t   = events_soa["time"]        # (E,)
        G   = events_soa["gradm"]       # (3,E)
        m   = events_soa["adc_mask"]    # (E,)
        ph  = events_soa["adc_phase"]   # (E,)

        # dt to prepend is temporarily carried in events_soa["_relax_pad_dt"]
        dt = float(events_soa.get("_relax_pad_dt", 0.0))
        if dt <= 0.0:
            # nothing to do
            events_soa.pop("_relax_pad_dt", None)
            return events_soa

        dev = t.device
        t0   = torch.tensor([dt], dtype=t.dtype, device=dev)
        G0   = torch.zeros((3,1), dtype=G.dtype, device=G.device)
        m0   = torch.zeros((1,), dtype=torch.bool, device=m.device)
        ph0  = torch.zeros((1,), dtype=ph.dtype, device=ph.device)

        out = {
            "time":      torch.cat([t0, t], dim=0),
            "gradm":     torch.cat([G0, G], dim=1),
            "adc_mask":  torch.cat([m0, m], dim=0),
            "adc_phase": torch.cat([ph0, ph], dim=0),
        }
        return out
    @classmethod
    def import_file_for_mrtini_sim(cls,
                                file_name: str,
                                pad: float = 150e-6,
                                exact_trajectories: bool = True,
                                print_stats: bool = False) -> "Sequence":
        """
        Build a standard MRzero Sequence AND attach hybrid spans:
        - EPG spans: MRzero-derived events (dt, gradm, adc_on, adc_phase)
        - Bloch spans: raw waveforms (t, rf_amp, rf_phase, rf_freq, gx, gy, gz, adc_active, adc_phase, adc_freq)

        Added attributes on returned seq (safe for MRzero):
        seq.mrtini_hybrid           : list[dict]  # time-ordered spans enriched
        seq.mrtini_rep_bounds       : list[(rep_start, rep_end)]
        seq.mrtini_spans_by_rep     : list[list[int]]  # span indices per repetition
        seq.mrtini_epg_indices      : list[int]
        seq.mrtini_bloch_indices    : list[int]
        seq.mrtini_pad              : float
        """
        from time import time
        import numpy as np
        import torch

        start = time()
        parser = mrstar.load_gstar(file_name)   # your Rust-backed parser
        if print_stats:
            print(f"Importing the file took {time() - start:.3f}s")

        # ---------- 1) Build the standard MRzero sequence (UNCHANGED) ----------
        seq = cls(normalized_grads=False)

        def pulse_usage(angle: float):
            return PulseUsage.EXCIT if abs(angle) < 100 * np.pi / 180 else PulseUsage.REFOC
        
        
        def collect_intervals(parser, name):
            out = []
            t = 0.0
            while True:
                iv = parser.encounter(name, t+1e-12)
                if iv is None:
                    break
                out.append(iv)
                t = iv[1]
            return out

        def overlaps(a0, a1, b0, b1, eps=1e-12):
            return max(a0, b0) < min(a1, b1) - eps

        def overlap_len(a0, a1, b0, b1):
            return max(0.0, min(a1, b1) - max(a0, b0))

        def merge_touching(intervals, eps=1e-12):
            if not intervals:
                return []
            xs = sorted(intervals)
            out = [list(xs[0])]
            for a, b in xs[1:]:
                if a <= out[-1][1] + eps:
                    out[-1][1] = max(out[-1][1], b)
                else:
                    out.append([a, b])
            return [tuple(x) for x in out]        
        
        pulses = collect_intervals(parser, "rf")
        z_blocks = collect_intervals(parser, "grad z")
        
        pulse_meta = []
        for rf0, rf1 in pulses:
            c = 0.5 * (rf0 + rf1)
            support0, support1  = rf0, rf1
            pulse_meta.append({
                "rf_start": rf0,
                "rf_end": rf1,
                "center": c,
                "support_start": support0,
                "support_end": support1,
            })
        # repetition bounds by pulse centers
        rep_bounds: list[tuple[float, float]] = []
        for i in range(len(pulses)):
            rep_start = pulse_meta[i]["center"] 
            rep_end = pulse_meta[i + 1]["center"] if i + 1 < len(pulses) else parser.duration()
            rep_bounds.append((rep_start, rep_end))

        # keep MRzero segmentation per repetition so we can mirror it onto EPG spans
        per_rep_data = []  # list of dicts with abs_times/moments/adc info, in rep order
        eps = 1e-6
        precision = 6


        
        for i, (rep_start, rep_end) in enumerate(rep_bounds):
            pulse = parser.integrate_one(pulses[i][0], pulses[i][1]).pulse
            adcs = parser.events("adc", rep_start, rep_end)
            if exact_trajectories:
                first = pulses[i][1]  # after pulse i
                last = (pulses[i + 1][0] if i + 1 < len(pulses) else rep_end)
                eps = 1e-6
                precision = 6

                if len(adcs) > 0:
                    grad_before = sorted(set([round(t, precision) for t in (
                        parser.events("grad x", first + eps, adcs[0] - eps) +
                        parser.events("grad y", first + eps, adcs[0] - eps) +
                        parser.events("grad z", first + eps, adcs[0] - eps)
                    )]))
                    grad_after = sorted(set([round(t, precision) for t in (
                        parser.events("grad x", adcs[-1] + eps, last - eps) +
                        parser.events("grad y", adcs[-1] + eps, last - eps) +
                        parser.events("grad z", adcs[-1] + eps, last - eps)
                    )]))
                    if i == len(pulses) - 1:
                        abs_times = [rep_start, first] + grad_before + adcs
                    else:
                        abs_times = [rep_start, first] + grad_before + adcs + grad_after + [last, rep_end]
                    adc_start = 2 + len(grad_before) - 1
                else:
                    grad = sorted(set([round(t, precision) for t in (
                        parser.events("grad x", first + eps, last - eps) +
                        parser.events("grad y", first + eps, last - eps) +
                        parser.events("grad z", first + eps, last - eps)
                    )]))
                    if i == len(pulses) - 1:
                        abs_times = [rep_start, first] + grad
                    else:
                        abs_times = [rep_start, first] + grad + [last, rep_end]
                    adc_start = None
            else:
                abs_times = [rep_start] + adcs + [rep_end]


            abs_times = sorted(set(abs_times))
        
            
            event_count = len(abs_times) - 1
            moments = parser.integrate(abs_times)                # grad moments per event

            # RF-head correction for peak-to-peak reps:
            # subtract the first half of the NEXT RF block from z-bookkeeping
            gradm_z_eff = moments.gradient.z.copy()  # start with full gradient moment
            def subtract_overlap_from_gradm(gradm_arr, a0, a1):
                for e, (a, b) in enumerate(zip(abs_times[:-1], abs_times[1:])):
                    lo = max(a, a0)
                    hi = min(b, a1)
                    if hi > lo:
                        gradm_arr[e] -= float(parser.integrate([lo, hi]).gradient.z[0])

            # current pulse tail belongs to current pulse operator
            tail_start = pulse_meta[i]["center"]
            tail_end   = pulse_meta[i]["support_end"]
            if tail_end > tail_start:
                subtract_overlap_from_gradm(gradm_z_eff, tail_start, tail_end)

            # next pulse head belongs to next pulse operator
            if i + 1 < len(pulse_meta):
                head_start = pulse_meta[i + 1]["support_start"]
                head_end   = pulse_meta[i + 1]["center"]
                if head_end > head_start:
                    subtract_overlap_from_gradm(gradm_z_eff, head_start, head_end)
                            
            # build MRzero rep (unchanged)
            rep = seq.new_rep(event_count)
            rep.pulse.angle = pulse.angle
            rep.pulse.phase = pulse.phase
            rep.pulse.usage = pulse_usage(pulse.angle)

            rep.event_time[:] = torch.as_tensor(np.diff(abs_times))
            rep.gradm[:, 0] = torch.as_tensor(moments.gradient.x)
            rep.gradm[:, 1] = torch.as_tensor(moments.gradient.y)
            rep.gradm[:, 2] = torch.as_tensor(moments.gradient.z)
            # rep.gradm[:, 2] = torch.as_tensor(gradm_z_eff)
            support0 = pulse_meta[i]["support_start"]
            support1 = pulse_meta[i]["support_end"]
            rw = parser.raw_waveforms(support0, support1)
            rep.raw_waveform = rw
            # sample ADC state at *end* of each event (one per interval)
            
            t_end = np.asarray(abs_times[1:], dtype=np.float64)
            # tiny epsilon to avoid edge-inclusion issues if your sampler is half-open
            t_end = t_end - 1e-12

            s_end = parser.sample(t_end)  # expects vectorized sampling
            adc_mask  = np.asarray(s_end.adc.active, dtype=bool)          # (E,)
            adc_phase = np.asarray(s_end.adc.phase,  dtype=np.float64)    # (E,) valid where mask True

            rep.adc_usage[:] = 0
            rep.adc_phase[:] = 0.0
            if adc_mask.any():
                m = torch.as_tensor(adc_mask, dtype=torch.bool)
                rep.adc_usage[m] = 1
                rep.adc_phase[m] = torch.as_tensor(np.pi/2 - adc_phase[adc_mask], dtype=torch.float32)

           
            # store the MRzero segmentation so we can annotate EPG spans with it
            per_rep_data.append(dict(
                abs_times=abs_times,                                 # length = event_count+1
                event_times = torch.as_tensor(np.diff(abs_times)),
                gradm_x=moments.gradient.x,                         # length = event_count
                gradm_y=moments.gradient.y,
                gradm_z=moments.gradient.z,
                gradm_z_eff=gradm_z_eff,                # NEW
                adc_start=None,                                 # None or index
                adc_count=len(adcs),
                adc_phase=adc_phase,                                # list[float] or None
                pulse_angle=pulse.angle,
                pulse_phase=pulse.phase,
                pulse_usage=str(pulse_usage(pulse.angle).value),
            ))

        if print_stats:
            print(f"Converting to MRzero took {time() - start:.3f}s")

        # ---------- 2) Build/enrich hybrid spans ----------
        # parser.hybrid(pad) should return ordered spans with fields {kind,t0,t1}.
        # For Bloch spans we attach raw waveforms; for EPG spans we attach MRzero-derived events.
        spans = parser.hybrid(pad, False)

        # map time -> repetition index
        def which_rep(tmid: float) -> int | None:
            for idx, (rs, re) in enumerate(rep_bounds):
                if rs <= tmid < re:
                    return idx
            if rep_bounds and abs(tmid - rep_bounds[-1][1]) < 1e-12:
                return len(rep_bounds) - 1
            return None

        spans_by_rep: list[list[int]] = [[] for _ in rep_bounds]
        epg_indices, bloch_indices = [], []
        bloch_windows = [(sp["t0"], sp["t1"]) for sp in spans if sp["kind"] == "bloch"]
        enriched: list[dict] = []
        for si, s in enumerate(spans):
            t0, t1 = s["t0"], s["t1"]
            tmid = 0.5 * (t0 + t1)
            ri = which_rep(tmid)
            spans_by_rep[ri].append(si) if ri is not None else None
            if s["kind"] == "bloch":
                # raw waveforms for Bloch ODE
                # assumes the parser exposes raw_waveforms(t0,t1)
                t0, t1 = pulse_meta[ri]["support_start"], pulse_meta[ri]["support_end"] if ri is not None else (t0, t1)
                rw = parser.raw_waveforms(t0, t1)
                enriched.append(dict(
                    kind="bloch",
                    t0=t0, t1=t1,
                    # raw arrays as-is (seconds, T/m, Hz, rad)
                    t=rw.get('t'),
                    rf_amp=rw.get('rf_amp'), rf_phase=rw.get('rf_phase'), rf_freq=rw.get('rf_freq'),
                    gx=rw.get('gx'), gy=rw.get('gy'), gz=rw.get('gz'),
                    adc_active=rw.get('adc_active'), adc_phase=rw.get('adc_phase'), adc_freq=rw.get('adc_freq'),
                    rep_index=ri,
                    rf_center=pulse_meta[ri]["center"] if ri is not None else None,
                    pulse_type=seq[ri].pulse.usage if ri is not None else None,
                ))
                bloch_indices.append(si)
            # ---------- collect Bloch windows once ----------

            # ... inside the for si, s in enumerate(spans): loop ...
            elif s["kind"] == "epg":
                
                if ri is None:
                    # simple fallback: one event, no ADC
                    m = parser.integrate_one(t0, t1).gradient
                    events_soa = dict(
                        time=torch.as_tensor([t1 - t0], dtype=torch.float32),
                        gradm=torch.as_tensor([[m.x], [m.y], [m.z]], dtype=torch.float32),
                        adc_mask=torch.as_tensor([False], dtype=torch.bool),
                        adc_phase=torch.as_tensor([0.0], dtype=torch.float32),
                    )
                    enriched.append(dict(
                        kind="epg", t0=t0, t1=t1,
                        events=[],                        # keep shape-compatible legacy field
                        events_soa=events_soa,            # NEW structure-of-arrays
                        g_area_total=[m.x, m.y, m.z],
                        flip=0.0, phase=0.0, rep_index=None,
                    ))
                    epg_indices.append(si)
                    continue

                # Grab repetition materials prepared earlier
                R = per_rep_data[ri]
                abs_times = R["abs_times"]
                gx = R["gradm_x"]; gy = R["gradm_y"]; gz = R["gradm_z"]
                adc_start = R["adc_start"]; adc_count = R["adc_count"]; adc_phase = R["adc_phase"]

                events_soa = cls._build_epg_soa_from_rep_slice(
                    t0=t0, t1=t1,
                    rep_abs_times=abs_times,
                    gm_x=gx, gm_y=gy, gm_z=gz,
                    adc_start=adc_start, adc_count=adc_count, adc_phase=adc_phase,
                    holes=[],
                )
                # prepend relax-only time equal to preceding Bloch
                pad_dt = cls._preceding_bloch_pad(t0, bloch_windows)
                events_soa["_relax_pad_dt"] = pad_dt
                events_soa = cls._prepend_relax_only(events_soa)
                collapsed = cls.soa_collapse_keep_adcs(events_soa)
                enriched.append(dict(
                    kind="epg",
                    t0=t0, t1=t1,
                    events_soa=events_soa,             # NEW SoA (time, gradm(3,E), adc_mask, adc_phase)
                    events=collapsed,                  # legacy shape-compatible (E',3), (E',), (E',)
                    flip=0.0, phase=0.0,
                    pulse_at_rep=dict(angle=R["pulse_angle"],
                                    phase=R["pulse_phase"],
                                    usage=R["pulse_usage"]),
                    rep_index=ri,
                ))
                epg_indices.append(si)
        # attach non-intrusive metadata for the hybrid sim
        seq.mrtini_hybrid        = enriched
        seq.mrtini_rep_bounds    = rep_bounds
        seq.mrtini_spans_by_rep  = spans_by_rep
        seq.mrtini_epg_indices   = epg_indices
        seq.mrtini_bloch_indices = bloch_indices
        seq.mrtini_pad           = pad

        return seq


    
    def plot_kspace_trajectory(self,
                               figsize: tuple[float, float] = (5, 5),
                               plotting_dims: str = 'xy',
                               plot_timeline: bool = True
                               ):
        """Plot the kspace trajectory produced by self.

        Parameters
        ----------
        kspace : list[Tensor]
            The kspace as produced by ``Sequence.get_full_kspace()``
        figsize : (float, float), optional
            The size of the plotted matplotlib figure.
        plotting_dims : string, optional
            String defining what is plotted on the x and y axis ('xy' 'zy' ...)
        plot_timeline : bool, optional
            Plot a second subfigure with the gradient components per-event.
        """
        assert len(plotting_dims) == 2
        assert plotting_dims[0] in ['x', 'y', 'z']
        assert plotting_dims[1] in ['x', 'y', 'z']
        dim_map = {'x': 0, 'y': 1, 'z': 2}

        # TODO: We could (optionally) plot which contrast a sample belongs to,
        # currently we only plot if it is measured or not

        kspace = self.get_full_kspace()
        adc_mask = [rep.adc_usage > 0 for rep in self]

        cmap = plt.get_cmap('rainbow')
        plt.figure(figsize=figsize)
        if plot_timeline:
            plt.subplot(211)
        for i, (rep_traj, mask) in enumerate(zip(kspace, adc_mask)):
            kx = rep_traj[:, dim_map[plotting_dims[0]]]
            ky = rep_traj[:, dim_map[plotting_dims[1]]]

            plt.plot(kx, ky, c=cmap(i / len(kspace)))
            plt.plot(kx[mask], ky[mask], 'r.')
            plt.plot(kx[~mask], ky[~mask], 'k.')
        plt.xlabel(f"$k_{plotting_dims[0]}$")
        plt.ylabel(f"$k_{plotting_dims[1]}$")
        plt.grid()

        if plot_timeline:
            plt.subplot(212)
            event = 0
            for i, rep_traj in enumerate(kspace):
                x = torch.arange(event, event + rep_traj.shape[0], 1)
                event += rep_traj.shape[0]

                if i == 0:
                    plt.plot(x, rep_traj[:, 0], c='r', label="$k_x$")
                    plt.plot(x, rep_traj[:, 1], c='g', label="$k_y$")
                    plt.plot(x, rep_traj[:, 2], c='b', label="$k_z$")
                else:
                    plt.plot(x, rep_traj[:, 0], c='r', label="_")
                    plt.plot(x, rep_traj[:, 1], c='g', label="_")
                    plt.plot(x, rep_traj[:, 2], c='b', label="_")
            plt.xlabel("Event")
            plt.ylabel("Gradient Moment")
            plt.legend()
            plt.grid()

        plt.show()


def chain(*sequences: Sequence, oneshot: bool = False) -> Sequence:
    """Chain multiple sequences into one.

    This function modifies the contrast of the sequences so that they don't
    overlap by shifting them by the maximum contrast of the previous sequence.

    Parameters
    ----------
    *sequences : Sequence
        Arbitrary number of sequences that will be chained

    oneshot : bool
        If true, contrasts of individual sequences are not shifted, which means
        that duplicates are possible.

    Returns
    -------
    Sequence
        A single sequence

    Examples
    --------
    >>> seq_a = build_a_seq()
    ... seq_b = build_another_seq()
    ... seq_ab = chain(seq_a, seq_b)
    >>> seq_a.get_contrasts()
    [1, 3]
    >>> seq_b.get_contrasts()
    [2]
    >>> seq_ab.get_contrasts()
    [1, 3, 5]
    """
    combined = Sequence()
    contrast_offset = 0

    for seq in sequences:
        temp = seq.clone()
        temp.shift_contrasts(contrast_offset)
        if not oneshot:
            contrast_offset = max(temp.get_contrasts())
        for rep in temp:
            combined.append(rep)

    return combined
