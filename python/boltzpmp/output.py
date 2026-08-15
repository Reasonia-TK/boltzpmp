"""Python向け計算結果データ構造。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mesh import VelocityMesh


@dataclass(kw_only=True)
class SwarmResult:
    energy_grid: np.ndarray
    eedf: np.ndarray
    eepf: np.ndarray
    mean_energy: float
    drift_velocity: float
    reduced_ionization_frequency: float
    rate_coefficients: dict[str, float]
    xi_used: float
    n: np.ndarray
    mesh: VelocityMesh
    converged: bool
    n_steps: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class SwarmResultRF(SwarmResult):
    time_grid: np.ndarray
    mean_energy_t: np.ndarray
    drift_velocity_t: np.ndarray
    reduced_ionization_frequency_t: np.ndarray
    E_t: np.ndarray
    phase_delay_energy: float
    phase_delay_W: float
    mean_energy_rms: float
    drift_velocity_rms: float
    nu_ion_rms_over_N: float
