"""Rust-accelerated propagator-method Boltzmann solver."""

from .crosssections import CrossSection, Gas, Mixture, load_argon, parse_lxcat
from .mesh import VelocityMesh
from .output import SwarmResult, SwarmResultRF
from .parallel import solve_dc_sweep
from .solver import PMSolver

__version__ = "0.1.0"

__all__ = [
    "CrossSection",
    "Gas",
    "Mixture",
    "PMSolver",
    "SwarmResult",
    "SwarmResultRF",
    "VelocityMesh",
    "load_argon",
    "parse_lxcat",
    "solve_dc_sweep",
    "__version__",
]
