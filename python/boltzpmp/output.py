"""Python向け計算結果データ構造。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mesh import VelocityMesh


@dataclass(kw_only=True)
class SwarmResult:
    """定常解。

    `rate_coefficients` は過程ごとの速度係数（その過程の標的1個あたり、m³/s）で、キーは
    `気体名:過程名`。混合気体全体への寄与は `fractions` の割合を掛けて足す。
    `reduced_ionization_frequency` と `reduced_attachment_frequency` はその和（m³/s）。
    """

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
    reduced_attachment_frequency: float = 0.0
    fractions: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def alpha_over_N(self) -> float:
        """換算電離係数 α/N (m²)。"""
        return self.reduced_ionization_frequency / self.drift_velocity

    @property
    def eta_over_N(self) -> float:
        """換算付着係数 η/N (m²)。"""
        return self.reduced_attachment_frequency / self.drift_velocity


@dataclass(kw_only=True)
class SwarmResultRF(SwarmResult):
    """RFの周期定常解。

    - `eedf`、`eepf`、`rate_coefficients` などの基底クラスの値は、電場の大きさが最大の時刻の値
      （0.4 までと同じ）。`mean_energy`、`drift_velocity`、`reduced_ionization_frequency` は波形の実効値。
    - `*_avg` は最後の1周期で時間平均した値（0.5 から）。プラズマの速度係数には周期平均を使う。
    - `eedf_t` は保存点ごとの EEDF（形は `(len(time_grid), len(energy_grid))`、eV⁻¹）。
    """

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
    mean_energy_avg: float = float("nan")
    eedf_avg: np.ndarray = field(default_factory=lambda: np.empty(0))
    eepf_avg: np.ndarray = field(default_factory=lambda: np.empty(0))
    rate_coefficients_avg: dict[str, float] = field(default_factory=dict)
    reduced_ionization_frequency_avg: float = float("nan")
    reduced_attachment_frequency_avg: float = float("nan")
    eedf_t: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
