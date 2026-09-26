"""Rustコアへ委譲するPython公開ソルバー。"""

from __future__ import annotations

import warnings
from typing import Any, Iterable

import numpy as np

from . import _core
from .crosssections import Mixture
from .mesh import VelocityMesh
from .output import SwarmResult, SwarmResultRF


class PMSolver:
    """プロパゲータ法のソルバー。

    格子は `eps_max_eV` と `d_eps_eV`（一様）か、`energy_grid`（0 eVから始まるエネルギー境界）で与える。

    物理モデル（既定はどちらも有効）:
    - `superelastic`: 超弾性衝突。ROTATIONの準位集団、`<->`、生成物の気体、2準位の占有の規則は
      `boltzpmp_core::processes` の説明どおり。占有の温度は `Mixture(T_K, T_exc_K, transition_energy_eV)`。
    - `gas_heating`: 弾性衝突での気体の熱運動によるエネルギー交換（`Mixture.T_K`）。

    断面積の表に運動量移行断面積（`CrossSection.mt_data`、LXCat表の3列目）があれば、
    その比から遮蔽Rutherford型の角度分布を作り、異方散乱として扱う。

    移流スキーム（`solve_dc`などの`scheme`）:
    - `limiter`（既定）: van Leer制限関数の2次精度TVDスキーム。負の値を作らない。
    - `upwind`: 1次精度。刻みと電場に比例する数値拡散で平均エネルギーを高めに出す。
    - `blending`: ξ = 1（中心差分）から始め、負の値が出るたびに ξ を下げてやり直す。
    """

    def __init__(
        self,
        mixture: Mixture,
        eps_max_eV: float | None = None,
        d_eps_eV: float | None = None,
        n_theta: int = 90,
        safety: float = 0.2,
        parallel: bool | None = None,
        *,
        energy_grid=None,
        superelastic: bool = True,
        gas_heating: bool = True,
    ) -> None:
        self.mixture = mixture
        self.safety = float(safety)
        # 現在の代表メッシュではセル内Rayon並列の同期コストが上回るため、
        # 既定は逐次とする。独立したE/N点はsolve_dc_sweepで並列化できる。
        self.parallel = False if parallel is None else bool(parallel)
        self.superelastic = bool(superelastic)
        self.gas_heating = bool(gas_heating)
        if energy_grid is None:
            if eps_max_eV is None or d_eps_eV is None:
                raise ValueError("give eps_max_eV and d_eps_eV, or energy_grid")
            edges = None
        else:
            if eps_max_eV is not None or d_eps_eV is not None:
                raise ValueError("give either energy_grid or eps_max_eV/d_eps_eV, not both")
            edges = np.asarray(energy_grid, dtype=float).ravel().tolist()
        self._core_solver = _core.CoreSolver.from_mixture(
            [gas.to_core() for gas in mixture.gases],
            mixture.N,
            mixture.T_K,
            mixture.T_exc_K,
            mixture.transition_energy_eV,
            edges,
            float("nan") if eps_max_eV is None else float(eps_max_eV),
            float("nan") if d_eps_eV is None else float(d_eps_eV),
            int(n_theta),
            self.safety,
            self.parallel,
            self.superelastic,
            self.gas_heating,
        )
        self.mesh = VelocityMesh._from_data(self._core_solver.mesh_data())

    def processes(self) -> list[dict[str, Any]]:
        """組み立てた衝突過程（逆過程を含む）の一覧。"""
        return list(self._core_solver.processes())

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

    def _dc_arguments(
        self,
        scheme: str,
        xi: float | None,
        tol: float | None,
        max_steps: int,
        check_every: int,
        init: str,
        init_T_eV: float,
        dt: float | None,
        init_n: np.ndarray | None,
        method: str,
    ) -> tuple:
        if tol is None:
            if method not in _DEFAULT_TOL:
                raise ValueError(f"unknown method: {method!r} (use 'explicit' or 'implicit')")
            tol = _DEFAULT_TOL[method]
        return (
            scheme,
            np.nan if xi is None else float(xi),
            float(tol),
            int(max_steps),
            int(check_every),
            float(init_T_eV),
            np.nan if dt is None else float(dt),
            self._initial_state(init, init_n),
        )

    def solve_dc(
        self,
        EN_Td: float,
        scheme: str = "limiter",
        xi: float | None = None,
        tol: float | None = None,
        max_steps: int = int(2e6),
        check_every: int = 200,
        init: str = "maxwell",
        init_T_eV: float = 1.0,
        dt: float | None = None,
        init_n: np.ndarray | None = None,
        method: str = "implicit",
    ) -> SwarmResult:
        """DC定常解。

        - `method="implicit"`（既定）: 定常方程式を輸送スイープのソース反復とAnderson加速で直接解く。
          - エネルギー緩和のような遅いモードは、エネルギーだけの二項近似の演算子で補正する（0.5 から）。
          - `init_n` を与えなければ、二項近似の定常解から始める（`init_T_eV` の Maxwell 分布は、その
            電子数の増加率を決めるのに使う）。
          - `tol`（既定1e-8）は1反復の残差 ‖g(n) − n‖₁、`max_steps` は反復回数の上限。`dt` は使わない。
          - 二項近似を使えなかったときは、その理由が `extra["two_term_error"]` に入る（使えたら `None`）。
        - `method="explicit"`: 時間発展で定常まで進める。`tol`（既定1e-6）は判定間隔ごとの相対変化、
          `max_steps` はステップ数。`scheme="blending"`（ξの探索）はこちらだけで使える。

        どちらも同じ離散方程式の解になる。
        """
        arguments = self._dc_arguments(
            scheme, xi, tol, max_steps, check_every, init, init_T_eV, dt, init_n, method
        )
        raw = self._core_solver.solve_dc(float(EN_Td), *arguments, method)
        return self._dc_result(raw, float(EN_Td), stacklevel=3)

    def solve_dc_many(
        self,
        EN_Td_values: Iterable[float],
        *,
        max_workers: int | None = None,
        scheme: str = "limiter",
        xi: float | None = None,
        tol: float | None = None,
        max_steps: int = int(2e6),
        check_every: int = 200,
        init: str = "maxwell",
        init_T_eV: float = 1.0,
        dt: float | None = None,
        init_n: np.ndarray | None = None,
        method: str = "implicit",
    ) -> list[SwarmResult]:
        """複数のDC換算電場をRustのスレッドで並列に解く（結果は入力順）。引数は `solve_dc` と同じ。"""
        values = [float(value) for value in EN_Td_values]
        if not values:
            return []
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be positive or None")
        arguments = self._dc_arguments(
            scheme, xi, tol, max_steps, check_every, init, init_T_eV, dt, init_n, method
        )
        raws = self._core_solver.solve_dc_many(values, *arguments, max_workers, method)
        results = []
        for raw, value in zip(raws, values, strict=True):
            results.append(self._dc_result(raw, value, stacklevel=3))
        return results

    def _dc_result(self, raw: dict[str, Any], EN_Td: float, stacklevel: int) -> SwarmResult:
        tail_ratio = float(raw["eepf_tail_ratio"])
        _warn_tail(tail_ratio, stacklevel + 1)
        return SwarmResult(
            energy_grid=self.mesh.eps_c.copy(),
            eedf=np.asarray(raw["eedf"], dtype=float),
            eepf=np.asarray(raw["eepf"], dtype=float),
            mean_energy=float(raw["mean_energy"]),
            drift_velocity=float(raw["drift_velocity"]),
            reduced_ionization_frequency=float(raw["reduced_ionization_frequency"]),
            reduced_attachment_frequency=float(raw["reduced_attachment_frequency"]),
            rate_coefficients=dict(raw["rate_coefficients"]),
            fractions=dict(raw["fractions"]),
            xi_used=float(raw["xi_used"]),
            n=np.asarray(raw["n"], dtype=float),
            mesh=self.mesh,
            converged=bool(raw["converged"]),
            n_steps=int(raw["n_steps"]),
            extra={
                "EN_Td": EN_Td,
                "a": float(raw["acceleration"]),
                "dt": float(raw["dt"]),
                "eepf_tail_ratio": tail_ratio,
                "two_term_error": raw["two_term_error"],
            },
        )

    def solve_rf(
        self,
        EN_rms_Td: float,
        freq_Hz: float,
        scheme: str = "limiter",
        xi: float | None = None,
        cycles_max: int = 200,
        tol: float | None = None,
        steps_per_cycle: int | None = None,
        init: str = "maxwell",
        init_T_eV: float = 1.0,
        n_store: int = 200,
        dt: float | None = None,
        init_n: np.ndarray | None = None,
        method: str = "implicit",
    ) -> SwarmResultRF:
        """RFの周期定常解。

        - `method="implicit"`（既定）: 各段をBDF2で陰的に解き、1周期の写像の不動点をAnderson加速で求める。
          - `steps_per_cycle` の既定は `n_store` の倍数で256段以上（`dt` を与えればそこから決める）。
          - `tol`（既定1e-6）は、周期写像の残差 ‖Φ(n) − n‖₁ と平均エネルギー波形の相対変化の許容値。
          - `init_n` を与えないときは実効値の電場でのDC解から始める（`init_T_eV` はそのDC計算の初期温度）。
        - `method="explicit"`: 陽的な時間発展（刻みは安定条件から決まる）。`tol`（既定1e-4）は周期ごとの
          平均エネルギー波形の相対変化。`scheme="blending"`（ξの探索）はこちらだけで使える。

        保存点は周期全体に等間隔に取る（0.5 から陽解法も同じ）。周期平均の EEDF と速度係数は
        `eedf_avg`、`rate_coefficients_avg` など、保存点ごとの EEDF は `eedf_t`（`SwarmResultRF` を参照）。
        """
        arguments = self._rf_arguments(
            scheme, xi, cycles_max, tol, steps_per_cycle, init, init_T_eV, n_store, dt, init_n,
            method,
        )
        raw = self._core_solver.solve_rf(float(EN_rms_Td), float(freq_Hz), *arguments, method)
        return self._rf_result(raw, float(EN_rms_Td), float(freq_Hz))

    def solve_rf_many(
        self,
        EN_rms_Td_values: Iterable[float],
        freq_Hz: float | Iterable[float],
        *,
        max_workers: int | None = None,
        scheme: str = "limiter",
        xi: float | None = None,
        cycles_max: int = 200,
        tol: float | None = None,
        steps_per_cycle: int | None = None,
        init: str = "maxwell",
        init_T_eV: float = 1.0,
        n_store: int = 200,
        dt: float | None = None,
        init_n: np.ndarray | None = None,
        method: str = "implicit",
    ) -> list[SwarmResultRF]:
        """複数のRF条件をRustのスレッドで並列に解く（結果は入力順）。

        `freq_Hz` は1つの値（すべてに共通）か、`EN_rms_Td_values` と同じ長さの列。ほかの引数は `solve_rf` と同じ。
        """
        values = [float(value) for value in EN_rms_Td_values]
        if not values:
            return []
        if np.ndim(freq_Hz) == 0:
            frequencies = [float(freq_Hz)] * len(values)
        else:
            frequencies = [float(value) for value in freq_Hz]
            if len(frequencies) != len(values):
                raise ValueError(
                    f"got {len(values)} EN_rms_Td values but {len(frequencies)} frequencies"
                )
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be positive or None")
        arguments = self._rf_arguments(
            scheme, xi, cycles_max, tol, steps_per_cycle, init, init_T_eV, n_store, dt, init_n,
            method,
        )
        raws = self._core_solver.solve_rf_many(
            values, frequencies, *arguments, max_workers, method
        )
        return [
            self._rf_result(raw, value, frequency)
            for raw, value, frequency in zip(raws, values, frequencies, strict=True)
        ]

    def _rf_arguments(
        self,
        scheme: str,
        xi: float | None,
        cycles_max: int,
        tol: float | None,
        steps_per_cycle: int | None,
        init: str,
        init_T_eV: float,
        n_store: int,
        dt: float | None,
        init_n: np.ndarray | None,
        method: str,
    ) -> tuple:
        if tol is None:
            if method not in _DEFAULT_RF_TOL:
                raise ValueError(f"unknown method: {method!r} (use 'explicit' or 'implicit')")
            tol = _DEFAULT_RF_TOL[method]
        return (
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

    def _rf_result(self, raw: dict[str, Any], EN_rms_Td: float, freq_Hz: float) -> SwarmResultRF:
        steps = int(raw["steps_per_cycle"])
        cycles = int(raw["n_cycles"])
        return SwarmResultRF(
            energy_grid=self.mesh.eps_c.copy(),
            eedf=np.asarray(raw["eedf"], dtype=float),
            eepf=np.asarray(raw["eepf"], dtype=float),
            mean_energy=float(raw["mean_energy_rms"]),
            drift_velocity=float(raw["drift_velocity_rms"]),
            reduced_ionization_frequency=float(raw["nu_ion_rms_over_N"]),
            reduced_attachment_frequency=float(raw["reduced_attachment_frequency"]),
            rate_coefficients=dict(raw["rate_coefficients"]),
            fractions=dict(raw["fractions"]),
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
                "inner_iterations": int(raw["inner_iterations"]),
                "cycle_residuals": np.asarray(raw["cycle_residuals"], dtype=float),
                "two_term_error": raw["two_term_error"],
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
            mean_energy_avg=float(raw["mean_energy_avg"]),
            eedf_avg=np.asarray(raw["eedf_avg"], dtype=float),
            eepf_avg=np.asarray(raw["eepf_avg"], dtype=float),
            rate_coefficients_avg=dict(raw["rate_coefficients_avg"]),
            reduced_ionization_frequency_avg=float(raw["reduced_ionization_frequency_avg"]),
            reduced_attachment_frequency_avg=float(raw["reduced_attachment_frequency_avg"]),
            eedf_t=np.asarray(raw["eedf_t"], dtype=float),
        )


_DEFAULT_TOL = {"explicit": 1e-6, "implicit": 1e-8}
_DEFAULT_RF_TOL = {"explicit": 1e-4, "implicit": 1e-6}


def _warn_tail(tail_ratio: float, stacklevel: int) -> None:
    if tail_ratio > 1e-6:
        warnings.warn(
            f"EEPF at eps_max is {tail_ratio:.2e} of its peak (> 1e-6); "
            "increase eps_max_eV for a fully converged tail.",
            stacklevel=stacklevel,
        )
