"""RFの陰解法（method="implicit"）の確認。陽解法との一致、時間の2次精度、高周波と低周波の極限。"""

from __future__ import annotations

import numpy as np
import pytest

from boltzpmp import CrossSection, Gas, Mixture, PMSolver
from boltzpmp.constants import E_CHARGE, M_E

pytestmark = pytest.mark.filterwarnings("ignore:EEPF at eps_max.*")

N = 3.2e22
TOWNSEND = 1e-21


def table(values: list[tuple[float, float]]) -> np.ndarray:
    return np.array(values, dtype=float)


def synthetic(ionization: bool = True) -> Mixture:
    sections = [
        CrossSection(
            kind="ELASTIC", species="G", name="el", mass_ratio=1e-4,
            data=table([(0.0, 3e-20), (100.0, 3e-20)]),
        ),
        CrossSection(
            kind="EXCITATION", species="G", name="exc", threshold=0.3,
            data=table([(0.3, 0.0), (0.6, 3e-20), (40.0, 3e-20)]),
        ),
    ]
    if ionization:
        sections.append(CrossSection(
            kind="IONIZATION", species="G", name="ion", threshold=4.0,
            data=table([(4.0, 0.0), (6.0, 2e-20), (40.0, 2e-20)]),
        ))
    return Mixture([Gas("G", 1.0, sections)], N=N, T_K=300.0)


SIGMA0 = 1e-19
NU = N * SIGMA0 * np.sqrt(2.0 * E_CHARGE / M_E)  # σ = σ0 (ε / 1 eV)^(-1/2) で運動量移行の周波数が一定


def maxwell_gas() -> Mixture:
    eps = np.geomspace(1e-4, 40.0, 200)
    data = np.vstack([[0.0, SIGMA0 / 1e-2], np.column_stack([eps, SIGMA0 / np.sqrt(eps)])])
    section = CrossSection(kind="ELASTIC", species="G", name="el", mass_ratio=1e-4, data=data)
    return Mixture([Gas("G", 1.0, [section])], N=N, T_K=300.0)


def on_phases(result, values: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """時刻の格子が異なる波形を、同じ256点の位相に周期的に補間する（`shift` は時刻のずれ）。"""
    period = 1.0 / result.extra["freq_Hz"]
    t = (np.asarray(result.time_grid) + shift) % period
    order = np.argsort(t)
    phase = np.linspace(0.0, period, 256, endpoint=False)
    return np.interp(phase, t[order], np.asarray(values)[order], period=period)


@pytest.fixture(scope="module")
def ionizing() -> PMSolver:
    return PMSolver(synthetic(), eps_max_eV=30.0, d_eps_eV=0.1, n_theta=8)


def test_implicit_rf_matches_explicit(ionizing: PMSolver) -> None:
    implicit = ionizing.solve_rf(60.0, 5e7, n_store=64, steps_per_cycle=512)
    assert implicit.converged
    assert implicit.extra["inner_iterations"] > 0
    # 陽解法は時間の1次精度なので、刻みを安定条件の1/4にして比べる
    stable = ionizing.solve_rf(60.0, 5e7, n_store=64, method="explicit", cycles_max=1, tol=0.0)
    explicit = ionizing.solve_rf(
        60.0, 5e7, n_store=64, method="explicit", tol=1e-7, cycles_max=500,
        dt=stable.extra["dt"] / 4.0, init_n=implicit.n,
    )
    assert explicit.converged
    assert implicit.mean_energy_rms == pytest.approx(explicit.mean_energy_rms, rel=5e-4)
    assert implicit.drift_velocity_rms == pytest.approx(explicit.drift_velocity_rms, rel=1e-3)
    assert implicit.nu_ion_rms_over_N == pytest.approx(explicit.nu_ion_rms_over_N, rel=2e-3)
    assert implicit.phase_delay_W == pytest.approx(explicit.phase_delay_W, abs=2e-3)
    # 0.5.0 から陽解法も、保存点の時刻の状態を記録する（周期全体に等間隔）
    energy = on_phases(implicit, implicit.mean_energy_t)
    explicit_energy = on_phases(explicit, explicit.mean_energy_t)
    assert np.max(np.abs(energy - explicit_energy)) < 1e-3 * np.max(energy)
    # 周期平均の値は陰解法と陽解法で一致する
    assert implicit.mean_energy_avg == pytest.approx(explicit.mean_energy_avg, rel=5e-4)
    assert implicit.reduced_ionization_frequency_avg == pytest.approx(
        explicit.reduced_ionization_frequency_avg, rel=2e-3
    )


def test_implicit_rf_is_second_order_in_time(ionizing: PMSolver) -> None:
    runs = {
        steps: ionizing.solve_rf(60.0, 5e7, n_store=64, steps_per_cycle=steps)
        for steps in (128, 256, 1024)
    }
    reference = runs[1024]
    errors = {
        steps: abs(runs[steps].drift_velocity_rms / reference.drift_velocity_rms - 1.0)
        for steps in (128, 256)
    }
    assert errors[128] < 1e-3
    assert errors[256] < errors[128] / 3.0


def test_high_frequency_limit_is_effective_field_dc() -> None:
    # 運動量移行の周波数が一定なら、ω ≫ エネルギー緩和の周波数で、時間平均の分布は
    # 実効電場 E_eff = E_rms ν/√(ν² + ω²) のDC解に一致し、ドリフトは Drude 型になる
    solver = PMSolver(maxwell_gas(), eps_max_eV=3.0, d_eps_eV=0.01, n_theta=8, gas_heating=False)
    omega = NU
    en_rms = 2.0
    result = solver.solve_rf(en_rms, omega / (2.0 * np.pi), n_store=64)
    assert result.converged
    dc = solver.solve_dc(en_rms / np.hypot(1.0, omega / NU), tol=1e-12)
    assert result.mean_energy_rms == pytest.approx(dc.mean_energy, rel=1e-3)
    drude = E_CHARGE * en_rms * TOWNSEND * N / M_E / np.hypot(NU, omega)
    assert result.drift_velocity_rms == pytest.approx(drude, rel=1e-2)
    assert result.phase_delay_W == pytest.approx(np.pi - np.arctan(omega / NU), abs=2e-3)


def test_low_frequency_limit_follows_the_field() -> None:
    # 周期がエネルギー緩和の時間よりずっと長いと、各時刻の分布はその時刻の電場のDC解になる
    solver = PMSolver(synthetic(ionization=False), eps_max_eV=8.0, d_eps_eV=0.04, n_theta=8)
    en_rms = 20.0
    result = solver.solve_rf(en_rms, 1e5, n_store=64)
    assert result.converged
    peak = int(np.argmax(np.abs(result.E_t)))
    dc = solver.solve_dc(np.sqrt(2.0) * en_rms, tol=1e-12)
    assert result.mean_energy_t[peak] == pytest.approx(dc.mean_energy, rel=1e-3)


@pytest.mark.parametrize("method", ["implicit", "explicit"])
def test_cycle_averages_are_time_averages(ionizing: PMSolver, method: str) -> None:
    result = ionizing.solve_rf(60.0, 5e7, n_store=64, method=method, cycles_max=300)
    assert result.converged
    widths = result.mesh.d_eps
    # 保存点は周期全体に等間隔なので、滑らかな周期波形の平均は時間平均に一致する
    assert result.mean_energy_avg == pytest.approx(np.mean(result.mean_energy_t), rel=1e-5)
    assert result.reduced_ionization_frequency_avg == pytest.approx(
        np.mean(result.reduced_ionization_frequency_t), rel=1e-5
    )
    assert np.sum(result.eedf_avg * widths) == pytest.approx(1.0, rel=1e-12)
    assert result.eedf_t.shape == (64, result.mesh.n_eps)
    np.testing.assert_allclose(result.eedf_t @ widths, 1.0, rtol=1e-12)
    np.testing.assert_allclose(result.eedf_t.mean(axis=0), result.eedf_avg, rtol=1e-4, atol=1e-12)
    ionization = [key for key in result.rate_coefficients_avg if key.endswith(":ion")]
    assert ionization and result.rate_coefficients_avg[ionization[0]] > 0.0


def test_solve_rf_many_matches_single_solves(ionizing: PMSolver) -> None:
    values = [40.0, 60.0]
    many = ionizing.solve_rf_many(values, 5e7, n_store=32, max_workers=2)
    for value, result in zip(values, many, strict=True):
        single = ionizing.solve_rf(value, 5e7, n_store=32)
        assert result.extra["EN_rms_Td"] == value
        np.testing.assert_array_equal(result.n, single.n)
        assert result.mean_energy_avg == single.mean_energy_avg
    frequencies = ionizing.solve_rf_many([60.0, 60.0], [5e7, 1e8], n_store=32)
    assert [r.extra["freq_Hz"] for r in frequencies] == [5e7, 1e8]
    with pytest.raises(ValueError, match="frequencies"):
        ionizing.solve_rf_many([60.0, 60.0], [5e7], n_store=32)


def test_implicit_rf_at_low_pressure() -> None:
    # 同梱Arの 10 Pa（1周期のうちに非等方成分がほぼ減衰し、エネルギー緩和は数千周期かかる条件）
    import boltzpmp as bp

    argon = bp.load_argon()
    solver = PMSolver(
        Mixture(argon.gases, p_Pa=10.0, T_K=300.0), eps_max_eV=40.0, d_eps_eV=0.2, n_theta=16
    )
    result = solver.solve_rf(10.0, 13.56e6, n_store=64)
    assert result.converged
    assert result.extra["n_cycles"] <= 40
    assert result.extra["two_term_error"] is None
    tight = solver.solve_rf(10.0, 13.56e6, n_store=64, tol=1e-9, init_n=result.n)
    assert tight.converged
    assert result.mean_energy_rms == pytest.approx(tight.mean_energy_rms, rel=1e-5)
    assert result.drift_velocity_rms == pytest.approx(tight.drift_velocity_rms, rel=1e-5)


def test_implicit_rf_rejects_invalid_options(ionizing: PMSolver) -> None:
    with pytest.raises(ValueError, match="blending"):
        ionizing.solve_rf(60.0, 5e7, scheme="blending")
    with pytest.raises(ValueError, match="method"):
        ionizing.solve_rf(60.0, 5e7, method="newton")
    with pytest.raises(ValueError, match="2 steps"):
        ionizing.solve_rf(60.0, 5e7, steps_per_cycle=1)
