"""DC定常解の陰解法（method="implicit"）が陽解法と同じ解を与えることの確認。"""

from __future__ import annotations

import numpy as np
import pytest

from boltzpmp import CrossSection, Gas, Mixture, PMSolver

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


def test_implicit_rejects_xi_search() -> None:
    solver = solver_for(elastic())
    with pytest.raises(ValueError, match="blending"):
        solver.solve_dc(5.0, scheme="blending", method="implicit")
    with pytest.raises(ValueError, match="method"):
        solver.solve_dc(5.0, method="newton")
