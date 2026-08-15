"""Rustコアへ委譲するPython公開ソルバー。"""

from __future__ import annotations

import warnings

import numpy as np

from . import _core
from .crosssections import Mixture
from .mesh import VelocityMesh
from .output import SwarmResult, SwarmResultRF


class PMSolver:
    def __init__(
        self,
        mixture: Mixture,
        eps_max_eV: float,
        d_eps_eV: float,
        n_theta: int = 90,
        safety: float = 0.2,
        parallel: bool | None = None,
    ) -> None:
        self.mixture = mixture
        self.mesh = VelocityMesh(eps_max_eV, d_eps_eV, n_theta)
        self.safety = float(safety)
        # 現在の代表メッシュではセル内Rayon並列の同期コストが上回るため、
        # 既定は逐次とする。独立したE/N点はsolve_dc_sweepで並列化できる。
        self.parallel = False if parallel is None else bool(parallel)
        processes = mixture.processes()
        kinds: list[str] = []
        gas_names: list[str] = []
        process_names: list[str] = []
        fractions: list[float] = []
        thresholds: list[float] = []
        mass_ratios: list[float] = []
        sigma_rows: list[list[float]] = []
        for gas, cross_section in processes:
            kinds.append(cross_section.kind)
            gas_names.append(gas.name)
            process_names.append(cross_section.name)
            fractions.append(float(gas.fraction))
            thresholds.append(float(cross_section.threshold))
            if cross_section.kind in ("ELASTIC", "EFFECTIVE"):
                mass_ratios.append(float(gas.mass_ratio(cross_section)))
            else:
                mass_ratios.append(0.0)
            sigma_rows.append(
                np.asarray(cross_section.sigma(self.mesh.eps_c), dtype=float).tolist()
            )
        self._core_solver = _core.CoreSolver(
            self.mesh.eps_max_eV,
            self.mesh.d_eps_eV,
            self.mesh.n_theta,
            self.safety,
            self.parallel,
            self.mixture.N,
            kinds,
            gas_names,
            process_names,
            fractions,
            thresholds,
            mass_ratios,
            sigma_rows,
        )

    def _initial_state(
        self,
        init: str,
        init_n: np.ndarray | None,
    ) -> list[float]:
        if init_n is not None:
            state = np.asarray(init_n, dtype=float).ravel()
            if state.shape != (self.mesh.n_cells,):
                raise ValueError(
                    f"init_n has shape {state.shape}, expected ({self.mesh.n_cells},)"
                )
            return state.tolist()
        if init != "maxwell":
            raise ValueError(f"unknown init: {init!r}")
        return []

    def solve_dc(
        self,
        EN_Td: float,
        scheme: str = "blending",
        xi: float | None = None,
        tol: float = 1e-6,
        max_steps: int = int(2e6),
        check_every: int = 200,
        init: str = "maxwell",
        init_T_eV: float = 1.0,
        dt: float | None = None,
        init_n: np.ndarray | None = None,
    ) -> SwarmResult:
        raw = self._core_solver.solve_dc(
            float(EN_Td),
            scheme,
            np.nan if xi is None else float(xi),
            float(tol),
            int(max_steps),
            int(check_every),
            float(init_T_eV),
            np.nan if dt is None else float(dt),
            self._initial_state(init, init_n),
        )
        eepf = np.asarray(raw["eepf"], dtype=float)
        tail_ratio = float(eepf[-1] / max(eepf.max(), 1e-300))
        if tail_ratio > 1e-6:
            warnings.warn(
                f"EEPF at eps_max is {tail_ratio:.2e} of its peak (> 1e-6); "
                "increase eps_max_eV for a fully converged tail.",
                stacklevel=2,
            )
        return SwarmResult(
            energy_grid=self.mesh.eps_c.copy(),
            eedf=np.asarray(raw["eedf"], dtype=float),
            eepf=eepf,
            mean_energy=float(raw["mean_energy"]),
            drift_velocity=float(raw["drift_velocity"]),
            reduced_ionization_frequency=float(raw["reduced_ionization_frequency"]),
            rate_coefficients=dict(raw["rate_coefficients"]),
            xi_used=float(raw["xi_used"]),
            n=np.asarray(raw["n"], dtype=float),
            mesh=self.mesh,
            converged=bool(raw["converged"]),
            n_steps=int(raw["n_steps"]),
            extra={
                "EN_Td": float(EN_Td),
                "a": float(raw["acceleration"]),
                "dt": float(raw["dt"]),
                "eepf_tail_ratio": tail_ratio,
            },
        )

    def solve_rf(
        self,
        EN_rms_Td: float,
        freq_Hz: float,
        scheme: str = "blending",
        xi: float | None = None,
        cycles_max: int = 200,
        tol: float = 1e-4,
        steps_per_cycle: int | None = None,
        init: str = "maxwell",
        init_T_eV: float = 1.0,
        n_store: int = 200,
        dt: float | None = None,
        init_n: np.ndarray | None = None,
    ) -> SwarmResultRF:
        raw = self._core_solver.solve_rf(
            float(EN_rms_Td),
            float(freq_Hz),
            scheme,
            np.nan if xi is None else float(xi),
            int(cycles_max),
            float(tol),
            0 if steps_per_cycle is None else int(steps_per_cycle),
            float(init_T_eV),
            int(n_store),
            np.nan if dt is None else float(dt),
            self._initial_state(init, init_n),
        )
        steps = int(raw["steps_per_cycle"])
        cycles = int(raw["n_cycles"])
        return SwarmResultRF(
            energy_grid=self.mesh.eps_c.copy(),
            eedf=np.asarray(raw["eedf"], dtype=float),
            eepf=np.asarray(raw["eepf"], dtype=float),
            mean_energy=float(raw["mean_energy_rms"]),
            drift_velocity=float(raw["drift_velocity_rms"]),
            reduced_ionization_frequency=float(raw["nu_ion_rms_over_N"]),
            rate_coefficients=dict(raw["rate_coefficients"]),
            xi_used=float(raw["xi_used"]),
            n=np.asarray(raw["n"], dtype=float),
            mesh=self.mesh,
            converged=bool(raw["converged"]),
            n_steps=cycles * steps,
            extra={
                "EN_rms_Td": float(EN_rms_Td),
                "freq_Hz": float(freq_Hz),
                "dt": float(raw["dt"]),
                "steps_per_cycle": steps,
                "n_cycles": cycles,
            },
            time_grid=np.asarray(raw["time"], dtype=float),
            mean_energy_t=np.asarray(raw["mean_energy_t"], dtype=float),
            drift_velocity_t=np.asarray(raw["drift_velocity_t"], dtype=float),
            reduced_ionization_frequency_t=np.asarray(
                raw["reduced_ionization_frequency_t"], dtype=float
            ),
            E_t=np.asarray(raw["field"], dtype=float),
            phase_delay_energy=float(raw["phase_delay_energy"]),
            phase_delay_W=float(raw["phase_delay_W"]),
            mean_energy_rms=float(raw["mean_energy_rms"]),
            drift_velocity_rms=float(raw["drift_velocity_rms"]),
            nu_ion_rms_over_N=float(raw["nu_ion_rms_over_N"]),
        )
