"""0.2.0で加えた物理モデル（超弾性、気体温度、異方散乱、非一様格子）の確認。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boltzpmp import CrossSection, Gas, Mixture, PMSolver, graded_energy_grid, parse_lxcat
from boltzpmp.constants import AMU, E_CHARGE, M_E

K_B_EV = 1.380649e-23 / E_CHARGE
THERMAL_MEAN = 1.5 * K_B_EV * 300.0  # 3/2 kT (eV)
GAMMA = np.sqrt(2.0 * E_CHARGE / M_E)  # 速度 = γ √ε
ARGON = Path(__file__).resolve().parents[2] / "examples" / "data" / "Ar_IST-Lisbon_LXCat.txt"

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
    # upwind の数値拡散（刻みと電場に比例）は電子を温めるので、中心差分で確かめる（ξの探索は陽解法だけ）
    result = solve(heated, 0.01, tol=1e-6, init_T_eV=0.035, scheme="blending", method="explicit")
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


def test_thermal_exchange_does_not_scatter_electrons() -> None:
    # 熱運動によるエネルギー交換は向きを変えない。0.5.0 までは跳びのたびに等方に再注入し、刻みの2乗に反比例
    # する余分な運動量移行でドリフト速度を下げていた（この気体の 2 meV 刻みで約半分）。熱平衡の分布の
    # ドリフト速度は、一定の断面積なら w = 2γ(E/N)/(3√π σ √kT)
    mixture = Mixture([Gas("G", 1.0, [elastic()])], N=3.2e22, T_K=300.0)
    expected = 2.0 * GAMMA * 0.01e-21 / (3.0 * np.sqrt(np.pi) * 1e-19 * np.sqrt(K_B_EV * 300.0))
    for d_eps in (0.004, 0.001):
        result = PMSolver(mixture, eps_max_eV=0.5, d_eps_eV=d_eps, n_theta=16).solve_dc(0.01)
        assert result.converged
        assert result.drift_velocity == pytest.approx(expected, rel=0.005), d_eps


def davydov(section: CrossSection, mass_ratio: float, en_td: float) -> tuple[float, float]:
    """弾性衝突だけの二項近似の定常解（Davydov 分布）の平均エネルギー (eV) と換算移動度 (1/(V m s))。"""
    kt = K_B_EV * 300.0
    eps = np.linspace(1e-7, 5.0, 200_001)
    sigma = section.sigma(eps)
    # f ∝ exp(−∫ dε / (kT + (M/6m)(E/N)²/(ε σ²)))
    scale = kt + (en_td * 1e-21) ** 2 / (6.0 * mass_ratio * eps * sigma**2)
    exponent = np.concatenate(([0.0], np.cumsum(0.5 * (1 / scale[1:] + 1 / scale[:-1]) * np.diff(eps))))
    f = np.exp(-exponent)
    norm = np.trapezoid(np.sqrt(eps) * f, eps)
    mean = np.trapezoid(eps**1.5 * f, eps) / norm
    mobility = GAMMA / 3.0 * np.trapezoid(eps / sigma * f / scale, eps) / norm
    return float(mean), float(mobility)


def test_low_field_argon_converges_to_two_term_solution() -> None:
    # 熱平衡に近い低い E/N（Ar の 0.0025 Td、非等方成分は約 0.2%）では Davydov 分布がほぼ厳密で、格子を
    # 細かくするとそれに近づく（0.5.0 までは細かくするほど離れ、1200 セルで移動度が 25% 小さかった）
    sections = parse_lxcat(ARGON)
    mixture = Mixture([Gas("Ar", 1.0, sections, mass_amu=39.948)], p_Pa=133.0, T_K=300.0)
    mean, mobility = davydov(sections[0], M_E / (39.948 * AMU), 0.0025)
    for cells, tolerance in ((300, 0.01), (1200, 0.002)):
        solver = PMSolver(mixture, eps_max_eV=1.0, d_eps_eV=1.0 / cells, n_theta=32)
        result = solver.solve_dc(0.0025)
        assert result.converged
        assert result.mean_energy == pytest.approx(mean, rel=tolerance), cells
        assert result.drift_velocity / 0.0025e-21 == pytest.approx(mobility, rel=tolerance), cells


EFFECTIVE_SET = """
EFFECTIVE
G
 1e-4
-----
 0.0 5e-20
 20.0 5e-20
 100.0 3e-20
-----
EXCITATION
G -> G*
 5.0
-----
 5.0 0.0
 10.0 1e-20
 100.0 1e-20
-----
IONIZATION
G -> G^+
 12.0
-----
 12.0 0.0
 30.0 1e-20
 100.0 1e-20
-----
"""


def test_effective_is_elastic_minus_inelastic() -> None:
    # EFFECTIVE（全運動量移行断面積）から非弾性断面積を引いた ELASTIC と同じ結果になる（BOLSIG+ と同じ）
    sections = parse_lxcat(EFFECTIVE_SET)
    effective, inelastic = sections[0], sections[1:]
    energy = np.unique(np.concatenate([section.data[:, 0] for section in sections]))
    table = np.column_stack(
        [energy, effective.sigma(energy) - sum(section.sigma(energy) for section in inelastic)]
    )
    elastic_set = CrossSection(
        kind="ELASTIC", species="G", name=effective.name, mass_ratio=effective.mass_ratio, data=table
    )
    results = []
    for first in (effective, elastic_set):
        mixture = Mixture([Gas("G", 1.0, [first, *inelastic])], N=3.2e22, T_K=300.0)
        solver = PMSolver(mixture, eps_max_eV=40.0, d_eps_eV=0.1, n_theta=12)
        results.append(solver.solve_dc(50.0))
    converted, reference = results
    assert converted.converged and reference.converged
    assert converted.mean_energy == pytest.approx(reference.mean_energy, rel=1e-9)
    assert converted.drift_velocity == pytest.approx(reference.drift_velocity, rel=1e-9)
    for key, value in reference.rate_coefficients.items():
        assert converted.rate_coefficients[key] == pytest.approx(value, rel=1e-9), key


def test_effective_below_inelastic_warns() -> None:
    sections = parse_lxcat(EFFECTIVE_SET.replace(" 100.0 3e-20", " 100.0 1e-20"))
    mixture = Mixture([Gas("G", 1.0, sections)], N=3.2e22)
    with pytest.warns(UserWarning, match="EFFECTIVE .* is negative"):
        PMSolver(mixture, eps_max_eV=40.0, d_eps_eV=0.1, n_theta=12)


def test_limiter_scheme_removes_upwind_heating() -> None:
    mixture = Mixture([Gas("G", 1.0, [elastic()])], N=3.2e22, T_K=300.0)
    solver = PMSolver(mixture, eps_max_eV=0.3, d_eps_eV=0.005, n_theta=8)
    eps = solver.mesh.eps_c
    weight = np.sqrt(eps) * np.exp(-eps / (K_B_EV * 300.0))
    target = float(np.sum(eps * weight) / np.sum(weight))
    upwind = solve(solver, 0.01, tol=1e-6, init_T_eV=0.035)
    limiter = solve(solver, 0.01, tol=1e-6, init_T_eV=0.035, scheme="limiter")
    assert limiter.converged
    assert np.isnan(limiter.xi_used)
    # upwind の数値拡散は刻みに比例して電子を温める（ここでは約6%）
    assert upwind.mean_energy / target - 1.0 > 0.03
    assert abs(limiter.mean_energy / target - 1.0) < 0.01


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
    a = solve(fine, 20.0, tol=1e-7, method="explicit")
    b = solve(graded, 20.0, tol=1e-7, method="explicit")
    assert a.converged and b.converged
    assert b.mean_energy == pytest.approx(a.mean_energy, rel=0.02)
    assert b.drift_velocity == pytest.approx(a.drift_velocity, rel=0.02)
    # 高エネルギー側の刻みが広いので、陽解法の時間刻みが大きくなる
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
