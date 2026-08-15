from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boltzpmp import CrossSection, Gas, Mixture, PMSolver, load_argon


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = np.load(ROOT / "reference" / "core_reference.npz")
SOLVER_REFERENCE = np.load(ROOT / "reference" / "solver_reference.npz")
METADATA = json.loads((ROOT / "reference" / "reference.json").read_text(encoding="utf-8"))


def reference_solver() -> PMSolver:
    sigma = METADATA["mixture"]["hard_sphere_sigma"]
    mass_ratio = METADATA["mixture"]["hard_sphere_mass_ratio"]
    threshold = METADATA["mixture"]["ionization_threshold_eV"]
    elastic = CrossSection(
        kind="ELASTIC",
        species="HS",
        name="HS elastic",
        mass_ratio=mass_ratio,
        data=np.array([[0.0, sigma], [100.0, sigma]]),
    )
    ionization = CrossSection(
        kind="IONIZATION",
        species="HS",
        name="HS ionization",
        threshold=threshold,
        data=np.array(
            [
                [0.0, 0.0],
                [threshold, 0.0],
                [threshold + 1e-6, sigma / 10.0],
                [100.0, sigma / 10.0],
            ]
        ),
    )
    mixture = Mixture(
        [Gas("HS", 1.0, [elastic, ionization])],
        N=METADATA["mixture"]["number_density"],
    )
    mesh = METADATA["mesh"]
    return PMSolver(
        mixture,
        eps_max_eV=mesh["eps_max_eV"],
        d_eps_eV=mesh["d_eps_eV"],
        n_theta=mesh["n_theta"],
    )


def csr_matvec(prefix: str, vector: np.ndarray) -> np.ndarray:
    data = REFERENCE[f"{prefix}_data"]
    indices = REFERENCE[f"{prefix}_indices"]
    indptr = REFERENCE[f"{prefix}_indptr"]
    shape = REFERENCE[f"{prefix}_shape"]
    result = np.zeros(int(shape[0]))
    for row in range(int(shape[0])):
        start, end = int(indptr[row]), int(indptr[row + 1])
        result[row] = np.dot(data[start:end], vector[indices[start:end]])
    return result


@pytest.fixture(scope="module")
def solver() -> PMSolver:
    return reference_solver()


def test_mesh_matches_python_reference(solver: PMSolver) -> None:
    mesh = solver._core_solver.mesh_data()
    mappings = {
        "eps_b": "eps_b",
        "eps_c": "eps_c",
        "v_b": "v_b",
        "v_c": "v_c",
        "theta_b": "theta_b",
        "theta_c": "theta_c",
        "V": "volume",
        "S_plus_eps": "s_plus_eps",
        "S_minus_eps": "s_minus_eps",
        "S_plus_theta": "s_plus_theta",
        "S_minus_theta": "s_minus_theta",
        "w_theta": "w_theta",
    }
    for rust_name, reference_name in mappings.items():
        actual = np.asarray(mesh[rust_name])
        expected = REFERENCE[reference_name]
        if expected.ndim == 2:
            actual = actual.reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("xi", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("sign", [-1, 1])
def test_advection_apply_matches_csr_reference(
    solver: PMSolver, xi: float, sign: int
) -> None:
    initial = REFERENCE["n_initial"]
    label = (
        f"advection_xi_{xi:.1f}_sign_{sign:+d}"
        .replace(".", "p")
        .replace("+", "plus")
        .replace("-", "minus")
    )
    expected = csr_matvec(label, initial)
    actual = np.asarray(solver._core_solver.advection_apply(initial.tolist(), xi, sign))
    scale = max(np.max(np.abs(expected)), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * scale)


def test_collision_apply_matches_reference(solver: PMSolver) -> None:
    actual = np.asarray(
        solver._core_solver.collision_apply(REFERENCE["n_initial"].tolist())
    )
    expected = REFERENCE["collision_apply"]
    scale = max(np.max(np.abs(expected)), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * scale)


def test_fixed_steps_match_reference(solver: PMSolver) -> None:
    config = METADATA["fixed_step"]
    actual = np.asarray(
        solver._core_solver.fixed_steps(
            REFERENCE["n_initial"].tolist(),
            config["acceleration"],
            config["dt"],
            config["xi"],
            1,
            config["steps"],
        )
    )
    assert np.sum(np.abs(actual - REFERENCE["fixed_step_n"])) <= 1e-10


@pytest.mark.filterwarnings("ignore:EEPF at eps_max.*")
def test_dc_matches_reference(solver: PMSolver) -> None:
    config = METADATA["dc"]
    result = solver.solve_dc(
        EN_Td=config["EN_Td"],
        scheme=config["scheme"],
        tol=1e-5,
        max_steps=200_000,
        check_every=100,
        init_n=REFERENCE["n_initial"],
    )
    assert result.converged == config["converged"]
    assert result.n_steps == config["n_steps"]
    assert result.xi_used == config["xi_used"]
    np.testing.assert_allclose(result.n, SOLVER_REFERENCE["dc_n"], rtol=1e-8, atol=1e-12)
    assert result.mean_energy == pytest.approx(config["mean_energy"], rel=1e-8)
    assert result.drift_velocity == pytest.approx(config["drift_velocity"], rel=1e-8)


def test_rf_matches_reference(solver: PMSolver) -> None:
    config = METADATA["rf"]
    result = solver.solve_rf(
        EN_rms_Td=config["EN_rms_Td"],
        freq_Hz=config["freq_Hz"],
        scheme=config["scheme"],
        xi=0.0,
        cycles_max=config["cycles"],
        tol=0.0,
        n_store=32,
        init_n=REFERENCE["n_initial"],
    )
    assert result.extra["steps_per_cycle"] == config["steps_per_cycle"]
    np.testing.assert_allclose(result.n, SOLVER_REFERENCE["rf_n"], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(
        result.mean_energy_t,
        SOLVER_REFERENCE["rf_mean_energy"],
        rtol=1e-8,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.drift_velocity_t,
        SOLVER_REFERENCE["rf_drift_velocity"],
        rtol=1e-8,
        atol=1e-8,
    )


def test_bundled_argon_loads() -> None:
    mixture = load_argon()
    assert len(mixture.gases) == 2
    assert len(mixture.processes()) == 5
