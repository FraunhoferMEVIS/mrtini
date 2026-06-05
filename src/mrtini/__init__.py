# src/mrtini/__init__.py
from .bloch import solve_slice
from .mrtini_main import MRtiniSim
__all__ = ["solve_slice"]

__all__.append("MRtiniSim")
__all__.append("Artifacts")