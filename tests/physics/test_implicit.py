"""DC定常解の陰解法（method="implicit"）が陽解法と同じ解を与えることの確認。"""

from __future__ import annotations

import numpy as np
import pytest

from boltzpmp import CrossSection, Gas, Mixture, PMSolver, parse_lxcat

pytestmark = pytest.mark.filterwarnings("ignore:EEPF at eps_max.*")


def table(values: list[tuple[float, float]]) -> np.ndarray:
    return np.array(values, dtype=float)


def elastic(sigma: float = 3e-20, mass_ratio: float = 1e-4) -> CrossSection:
    return CrossSection(
        kind="ELASTIC", species="G", name="el", mass_ratio=mass_ratio,
        data=table([(0.0, sigma), (100.0, sigma)]),
    )


def excitation(mt_ratio: float | None = None) -> CrossSection:
    data = table([(0.3, 0.0), (0.6, 3e-20), (20.0, 3e-20)])
    mt = None if mt_ratio is None else np.column_stack([data[:, 0], mt_ratio * data[:, 1]])
    return CrossSection(
        kind="EXCITATION", species="G", name="exc", threshold=0.3, data=data, mt_data=mt
    )


def solver_for(*sections: CrossSection, eps_max: float = 6.0, d_eps: float = 0.01) -> PMSolver:
    mixture = Mixture([Gas("G", 1.0, list(sections))], N=3.2e22, T_K=300.0)
    return PMSolver(mixture, eps_max_eV=eps_max, d_eps_eV=d_eps, n_theta=12)


def both(solver: PMSolver, en: float, scheme: str = "limiter"):
    explicit = solver.solve_dc(en, scheme=scheme, method="explicit", tol=1e-8, max_steps=3_000_000)
    implicit = solver.solve_dc(en, scheme=scheme, method="implicit", tol=1e-10)
    return explicit, implicit


@pytest.mark.parametrize("scheme", ["limiter", "upwind"])
def test_implicit_matches_explicit(scheme: str) -> None:
    solver = solver_for(elastic(), excitation())
    explicit, implicit = both(solver, 20.0, scheme)
    assert explicit.converged and implicit.converged
    assert implicit.mean_energy == pytest.approx(explicit.mean_energy, rel=2e-5)
    assert implicit.drift_velocity == pytest.approx(explicit.drift_velocity, rel=2e-5)
    assert np.isnan(implicit.extra["dt"])
    assert implicit.n_steps < explicit.n_steps / 20


def test_implicit_with_anisotropic_scattering() -> None:
    solver = solver_for(elastic(), excitation(mt_ratio=0.05))
    explicit, implicit = both(solver, 20.0)
    assert implicit.mean_energy == pytest.approx(explicit.mean_energy, rel=2e-5)
    assert implicit.drift_velocity == pytest.approx(explicit.drift_velocity, rel=2e-5)


def test_implicit_with_growth_and_loss() -> None:
    # 電離（電子数が増える）と付着（減る）で、増加率の扱いが左辺・右辺に分かれる
    ionization = CrossSection(
        kind="IONIZATION", species="G", name="ion", threshold=4.0,
        data=table([(4.0, 0.0), (6.0, 2e-20), (40.0, 2e-20)]),
    )
    attachment = CrossSection(
        kind="ATTACHMENT", species="G", name="att", data=table([(0.0, 3e-21), (40.0, 3e-21)])
    )
    for sections, sign in (((elastic(), ionization), 1.0), ((elastic(), attachment), -1.0)):
        solver = solver_for(*sections, eps_max=30.0, d_eps=0.05)
        explicit, implicit = both(solver, 150.0)
        growth = implicit.reduced_ionization_frequency - implicit.reduced_attachment_frequency
        assert np.sign(growth) == sign
        assert implicit.mean_energy == pytest.approx(explicit.mean_energy, rel=2e-5)
        assert implicit.drift_velocity == pytest.approx(explicit.drift_velocity, rel=2e-5)


def test_two_term_acceleration_resolves_slow_energy_relaxation() -> None:
    # 弾性衝突だけ（m/M = 1e-5）の気体では、エネルギー緩和に約 1e5 回の衝突がかかり、ソース反復だけでは
    # 数千反復でも収束しない。二項近似の合成加速と、二項近似の解から始める初期状態で数十反復になる
    solver = solver_for(elastic(mass_ratio=1e-5), eps_max=10.0, d_eps=0.02)
    result = solver.solve_dc(2.0)
    assert result.converged
    assert result.extra["two_term_error"] is None
    assert result.n_steps < 80
    reference = solver.solve_dc(2.0, tol=1e-12, max_steps=20_000)
    assert reference.converged
    assert result.mean_energy == pytest.approx(reference.mean_energy, rel=1e-6)
    assert result.drift_velocity == pytest.approx(reference.drift_velocity, rel=1e-6)


def test_implicit_molecular_gas_from_the_default_start() -> None:
    # 回転励起と逆過程が支配的な低い E/N（0.4 までは既定の 1 eV の Maxwell 分布から収束しなかった型）
    rotations = parse_lxcat(
        """
ROTATION
G
 0.0 1.0
 0.005 3.0
-----
 0.005 0.0
 0.05 5.0e-19
 10.0 5.0e-19
-----
ROTATION
G
 0.005 3.0
 0.015 5.0
-----
 0.010 0.0
 0.05 5.0e-19
 10.0 5.0e-19
-----
"""
    )
    vibration = CrossSection(
        kind="EXCITATION", species="G", name="vib", threshold=0.5,
        data=table([(0.5, 0.0), (0.8, 1e-20), (10.0, 1e-20)]),
    )
    mixture = Mixture(
        [Gas("G", 1.0, [elastic(sigma=1e-19, mass_ratio=2.7e-5), *rotations, vibration])],
        N=3.2e22,
        T_K=300.0,
    )
    solver = PMSolver(mixture, eps_max_eV=3.0, d_eps_eV=0.005, n_theta=8)
    for en in (1.0, 10.0):
        result = solver.solve_dc(en)
        assert result.converged, en
        assert result.n_steps < 100, (en, result.n_steps)
        explicit = solver.solve_dc(
            en, method="explicit", tol=1e-9, max_steps=3_000_000, init_n=result.n
        )
        assert result.mean_energy == pytest.approx(explicit.mean_energy, rel=2e-5)
        assert result.drift_velocity == pytest.approx(explicit.drift_velocity, rel=2e-5)


def test_implicit_rejects_xi_search() -> None:
    solver = solver_for(elastic())
    with pytest.raises(ValueError, match="blending"):
        solver.solve_dc(5.0, scheme="blending", method="implicit")
    with pytest.raises(ValueError, match="method"):
        solver.solve_dc(5.0, method="newton")
