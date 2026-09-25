"""Rust-accelerated propagator-method Boltzmann solver."""

from importlib.metadata import PackageNotFoundError, version

from .crosssections import CrossSection, Gas, Mixture, load_argon, parse_lxcat
from .mesh import VelocityMesh, graded_energy_grid
from .output import SwarmResult, SwarmResultRF
from .parallel import solve_dc_sweep
from .solver import PMSolver

try:
    __version__ = version("boltzpmp")
except PackageNotFoundError:  # ソースツリーから直接読み込んだとき
    __version__ = "0.0.0"

__all__ = [
    "CrossSection",
    "Gas",
    "Mixture",
    "PMSolver",
    "SwarmResult",
    "SwarmResultRF",
    "VelocityMesh",
    "graded_energy_grid",
    "load_argon",
    "parse_lxcat",
    "solve_dc_sweep",
    "__version__",
]
