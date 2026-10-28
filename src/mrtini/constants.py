from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SimulationConstants:
    GAMMA_HZ: float = 42.577478e6  # Hz/T ~ 267.5e6 rad/s/T
    GAMMA_RAD: float = 2*np.pi*GAMMA_HZ  # rad/T/s


@dataclass(frozen=True)
class OCRA_scanner:
    frequency: float = 11264000  # Hz
    max_grad_amp: float = 0.250  # T/m
    max_grad_slew: float = 1000  # T/m/s
    max_rf_amp: float = 0.0002  # T
    raster_time_grad: float = 10e-6  # sec
    raster_time_rf: float = 1e-6  # sec
    raster_time_adc: float = 1e-7  # sec
    B0_T:float = 0.26
    ppm_rms:float = 1.0
    ppm_floor:float = 0.1
@dataclass(frozen=True)
class MAG_3T:
    max_grad_amp:float = 0.50
    max_grad_slew:float= 1800
    slice_thicness:float = 2e-3
    B0_T:float = 3
    ppm_rms:float = 0.05
    ppm_floor:float = 0.01
     