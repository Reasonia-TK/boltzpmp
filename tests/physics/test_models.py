"""0.2.0で加えた物理モデル（超弾性、気体温度、異方散乱、非一様格子）の確認。"""

from __future__ import annotations

import numpy as np
import pytest

from boltzpmp import CrossSection, Gas, Mixture, PMSolver, graded_energy_grid, parse_lxcat
from boltzpmp.constants import E_CHARGE

K_B_EV = 1.380649e-23 / E_CHARGE
THERMAL_MEAN = 1.5 * K_B_EV * 300.0  # 3/2 kT (eV)

pytestmark = pytest.mark.filterwarnings("ignore:EEPF at eps_max.*")


def elastic(sigma: float = 1e-19, mass_ratio: float = 1e-3) -> CrossSection:
    return CrossSection(
        kind="ELASTIC",
        species="G",
        name="elastic",
        mass_ratio=mass_ratio,
        data=np.array([[0.0, sigma], [100.0, sigma]]),
    )


ROTATIONS = """
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


def rotor_mixture() -> Mixture:
    gas = Gas("G", 1.0, [elastic(mass_ratio=1e-5), *parse_lxcat(ROTATIONS)])
    return Mixture([gas], N=3.2e22, T_K=300.0)


def solve(solver: PMSolver, en: float, **kwargs):
    options = {"scheme": "upwind", "tol": 1e-7, "max_steps": 400_000, "init_T_eV": 0.05}
    options.update(kwargs)
    return solver.solve_dc(en, **options)


def test_rotational_superelastic_relaxes_to_gas_temperature() -> None:
    mixture = rotor_mixture()
    hot = PMSolver(mixture, eps_max_eV=0.6, d_eps_eV=0.001, n_theta=8)
    result = solve(hot, 0.05)
    assert result.converged
    assert result.mean_energy == pytest.approx(THERMAL_MEAN, rel=0.03)
    # 逆過程がないと電子は回転しきい値の下へ冷える
    cold = PMSolver(
        mixture, eps_max_eV=0.6, d_eps_eV=0.001, n_theta=8, superelastic=False, gas_heating=False
    )
    assert solve(cold, 0.05).mean_energy < 0.5 * THERMAL_MEAN


def test_gas_heating_thermalizes_elastic_gas() -> None:
    mixture = Mixture([Gas("G", 1.0, [elastic()])], N=3.2e22, T_K=300.0)
    heated = PMSolver(mixture, eps_max_eV=0.3, d_eps_eV=0.005, n_theta=8)
    # upwind の数値拡散（刻みと電場に比例）は電子を温めるので、中心差分で確かめる
    result = solve(heated, 0.01, tol=1e-6, init_T_eV=0.035, scheme="blending")
    assert result.converged
    assert result.xi_used == 1.0
    # 格子上のMaxwell分布が定常解（連続の 3/2 kT とは刻みの分だけ異なる）
    eps = heated.mesh.eps_c
    weight = np.sqrt(eps) * np.exp(-eps / (K_B_EV * 300.0))
    discrete_mean = float(np.sum(eps * weight) / np.sum(weight))
    assert result.mean_energy == pytest.approx(discrete_mean, rel=2e-3)
    assert discrete_mean == pytest.approx(THERMAL_MEAN, rel=0.06)
    cold = PMSolver(mixture, eps_max_eV=0.3, d_eps_eV=0.005, n_theta=8, gas_heating=False)
    assert solve(cold, 0.01, tol=1e-6, init_T_eV=0.035).mean_energy < 0.5 * THERMAL_MEAN


def test_process_list_and_fractions() -> None:
    solver = PMSolver(rotor_mixture(), eps_max_eV=0.6, d_eps_eV=0.002, n_theta=8)
    names = [process["name"] for process in solver.processes()]
    assert names.count("G rotation") == 2
    assert names.count("G rotation (superelastic)") == 2
    fractions = [process["fraction"] for process in solver.processes()]
    kt = K_B_EV * 300.0
    weights = np.array([1.0, 3.0 * np.exp(-0.005 / kt), 5.0 * np.exp(-0.015 / kt)])
    y = weights / weights.sum()
    np.testing.assert_allclose(sorted(fractions[1:]), sorted([y[0], y[1], y[1], y[2]]), rtol=1e-12)
    result = solve(solver, 1.0, max_steps=2000, tol=0.0)
    assert set(result.fractions) == set(result.rate_coefficients)


def test_attachment_frequency_is_weighted_by_fraction() -> None:
    attachment = CrossSection(
        kind="ATTACHMENT", species="G", name="att", data=np.array([[0.0, 1e-22], [10.0, 1e-22]])
    )
    mixture = Mixture(
        [Gas("G", 0.25, [elastic(), attachment]), Gas("H", 0.75, [elastic()])], N=3.2e22
    )
    solver = PMSolver(mixture, eps_max_eV=2.0, d_eps_eV=0.01, n_theta=8)
    result = solve(solver, 5.0, max_steps=5000, tol=0.0)
    rate = result.rate_coefficients["G:att"]
    assert result.reduced_attachment_frequency == pytest.approx(0.25 * rate, rel=1e-12)
    assert result.eta_over_N == pytest.approx(0.25 * rate / result.drift_velocity, rel=1e-12)


def test_uniform_edges_reproduce_uniform_grid() -> None:
    mixture = Mixture([Gas("G", 1.0, [elastic()])], N=3.2e22)
    uniform = PMSolver(mixture, eps_max_eV=2.0, d_eps_eV=0.02, n_theta=8)
    edges = np.arange(101) * 0.02
    graded = PMSolver(mixture, energy_grid=edges, n_theta=8)
    assert graded.mesh.d_eps_eV is None
    a = solve(uniform, 5.0, max_steps=3000, tol=0.0)
    b = solve(graded, 5.0, max_steps=3000, tol=0.0)
    assert b.mean_energy == pytest.approx(a.mean_energy, rel=1e-9)
    assert b.drift_velocity == pytest.approx(a.drift_velocity, rel=1e-9)


def test_graded_grid_agrees_with_fine_uniform_grid() -> None:
    excitation = CrossSection(
        kind="EXCITATION",
        species="G",
        name="exc",
        threshold=0.3,
        data=np.array([[0.3, 0.0], [0.6, 3e-20], [10.0, 3e-20]]),
    )
    mixture = Mixture([Gas("G", 1.0, [elastic(sigma=3e-20, mass_ratio=1e-4), excitation])], N=3.2e22)
    fine = PMSolver(mixture, eps_max_eV=6.0, d_eps_eV=0.005, n_theta=12)
    edges = graded_energy_grid(6.0, 0.005, 0.5)
    graded = PMSolver(mixture, energy_grid=edges, n_theta=12)
    assert graded.mesh.n_eps < 0.5 * fine.mesh.n_eps
    a = solve(fine, 20.0, tol=1e-7)
    b = solve(graded, 20.0, tol=1e-7)
    assert a.converged and b.converged
    assert b.mean_energy == pytest.approx(a.mean_energy, rel=0.02)
    assert b.drift_velocity == pytest.approx(a.drift_velocity, rel=0.02)
    assert b.extra["dt"] > a.extra["dt"]


def inelastic(mt_ratio: float | None, data_scale: float = 1.0) -> CrossSection:
    table = np.array([[0.05, 0.0], [0.1, 1e-19], [10.0, 1e-19]])
    table[:, 1] *= data_scale
    mt = None if mt_ratio is None else np.column_stack([table[:, 0], mt_ratio * table[:, 1]])
    return CrossSection(
        kind="EXCITATION", species="G", name="exc", threshold=0.05, data=table, mt_data=mt
    )


def aniso_solver(section: CrossSection) -> PMSolver:
    mixture = Mixture([Gas("G", 1.0, [elastic(sigma=1e-20, mass_ratio=1e-4), section])], N=3.2e22)
    return PMSolver(mixture, eps_max_eV=3.0, d_eps_eV=0.01, n_theta=16, superelastic=False)


def test_equal_momentum_transfer_is_isotropic() -> None:
    plain = solve(aniso_solver(inelastic(None)), 10.0, max_steps=3000, tol=0.0)
    same = solve(aniso_solver(inelastic(1.0)), 10.0, max_steps=3000, tol=0.0)
    np.testing.assert_array_equal(plain.n, same.n)


def test_forward_scattering_lies_between_isotropic_limits() -> None:
    # ICS を等方に使うと運動量移行を過大評価し、MT を使うとエネルギー損失を過小評価する
    ics = solve(aniso_solver(inelastic(None)), 10.0)
    mt = solve(aniso_solver(inelastic(None, data_scale=0.01)), 10.0)
    aniso = solve(aniso_solver(inelastic(0.01)), 10.0)
    assert ics.converged and mt.converged and aniso.converged
    assert ics.drift_velocity < aniso.drift_velocity
    assert aniso.mean_energy < mt.mean_energy
    # 運動量移行が等しいので、ドリフト速度は ICS 版より MT 版に近い
    distance = lambda a, b: abs(np.log(a.drift_velocity / b.drift_velocity))  # noqa: E731
    assert distance(aniso, mt) < distance(aniso, ics)
