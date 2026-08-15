"""現行boltzpmからRust移植用の決定的な基準値を生成する。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse

import boltzpm
from boltzpm.crosssections import CrossSection, Gas, Mixture
from boltzpm.mesh import VelocityMesh
from boltzpm.output import compute_swarm
from boltzpm.propagators import build_advection_matrix, build_collision_operator
from boltzpm.solver import PMSolver


HS_SIGMA = 1.0e-19
HS_MASS_RATIO = 1.36e-5


def hard_sphere_mixture(number_density: float = 3.5e22) -> Mixture:
    """移植中も外部データに依存しない決定的な試験用混合気体を作る。"""
    elastic = CrossSection(
        kind="ELASTIC",
        species="HS",
        name="HS elastic",
        threshold=0.0,
        mass_ratio=HS_MASS_RATIO,
        data=np.array([[0.0, HS_SIGMA], [100.0, HS_SIGMA]]),
    )
    ionization = CrossSection(
        kind="IONIZATION",
        species="HS",
        name="HS ionization",
        threshold=5.0,
        data=np.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [5.000001, HS_SIGMA / 10.0],
                [100.0, HS_SIGMA / 10.0],
            ]
        ),
    )
    return Mixture(
        [Gas(name="HS", fraction=1.0, cross_sections=[elastic, ionization])],
        N=number_density,
    )


def csr_parts(prefix: str, matrix: sparse.csr_matrix) -> dict[str, np.ndarray]:
    """CSRを言語非依存で復元できる三配列へ分解する。"""
    matrix = matrix.tocsr()
    return {
        f"{prefix}_data": matrix.data,
        f"{prefix}_indices": matrix.indices.astype(np.int64),
        f"{prefix}_indptr": matrix.indptr.astype(np.int64),
        f"{prefix}_shape": np.asarray(matrix.shape, dtype=np.int64),
    }


def fixed_step_reference(solver: PMSolver, n0: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """収束判定に依存しない100ステップ基準を作る。"""
    # 固定ステップ基準は、安定性が保証されるupwindを使う。ブレンディング
    # 係数0.5と1.0は別途、演算子係数そのものをfixtureへ保存して比較する。
    xi = 0.0
    acceleration = 2.5e15
    dt = min(solver._auto_dt(acceleration), 1.0e-12)
    advection = solver._get_B(xi, sign=1)
    n = n0.copy()
    for _ in range(100):
        n_new = n + dt * (
            acceleration * (advection @ n) + solver.collision_op.apply(n)
        )
        if n_new.min() < -1.0e-14 * n_new.max():
            raise RuntimeError("fixed-step reference unexpectedly became negative")
        n = n_new / n_new.sum()
    swarm = compute_swarm(n, solver.mesh, solver.mixture)
    return n, {
        "xi": xi,
        "acceleration": acceleration,
        "dt": dt,
        "steps": 100,
        "mean_energy": swarm["mean_energy"],
        "drift_velocity": swarm["drift_velocity"],
        "reduced_ionization_frequency": swarm["reduced_ionization_frequency"],
    }


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = VelocityMesh(eps_max_eV=8.0, d_eps_eV=0.2, n_theta=16)
    mixture = hard_sphere_mixture()
    solver = PMSolver(mixture, eps_max_eV=8.0, d_eps_eV=0.2, n_theta=16)

    rng = np.random.default_rng(20260815)
    n0 = rng.random(mesh.n_cells)
    n0 /= n0.sum()

    arrays: dict[str, np.ndarray] = {
        "eps_b": mesh.eps_b,
        "eps_c": mesh.eps_c,
        "v_b": mesh.v_b,
        "v_c": mesh.v_c,
        "theta_b": mesh.theta_b,
        "theta_c": mesh.theta_c,
        "volume": mesh.V,
        "s_plus_eps": mesh.S_plus_eps,
        "s_minus_eps": mesh.S_minus_eps,
        "s_plus_theta": mesh.S_plus_theta,
        "s_minus_theta": mesh.S_minus_theta,
        "w_theta": mesh.w_theta,
        "n_initial": n0,
        "collision_nu_total": solver.collision_op.nu_total,
        "collision_apply": solver.collision_op.apply(n0),
    }
    arrays.update(csr_parts("collision_energy", solver.collision_op.M_e))
    for xi in (0.0, 0.5, 1.0):
        for sign in (-1, 1):
            label = f"advection_xi_{xi:.1f}_sign_{sign:+d}".replace(".", "p").replace("+", "plus").replace("-", "minus")
            arrays.update(csr_parts(label, build_advection_matrix(mesh, xi, sign)))

    fixed_n, fixed_meta = fixed_step_reference(solver, n0)
    arrays["fixed_step_n"] = fixed_n
    np.savez_compressed(output_dir / "core_reference.npz", **arrays)

    dc = solver.solve_dc(
        EN_Td=10.0,
        scheme="upwind",
        tol=1.0e-5,
        max_steps=200_000,
        check_every=100,
        init_n=n0,
    )
    rf = solver.solve_rf(
        EN_rms_Td=10.0,
        freq_Hz=13.56e6,
        scheme="upwind",
        xi=0.0,
        cycles_max=3,
        tol=0.0,
        n_store=32,
        init_n=n0,
    )
    np.savez_compressed(
        output_dir / "solver_reference.npz",
        dc_n=dc.n,
        dc_eedf=dc.eedf,
        dc_eepf=dc.eepf,
        rf_n=rf.n,
        rf_eedf=rf.eedf,
        rf_eepf=rf.eepf,
        rf_time=rf.time_grid,
        rf_field=rf.E_t,
        rf_mean_energy=rf.mean_energy_t,
        rf_drift_velocity=rf.drift_velocity_t,
        rf_nu_ion_over_n=rf.reduced_ionization_frequency_t,
    )

    metadata = {
        "schema_version": 1,
        "generator": "benchmarks/generate_reference.py",
        "python": sys.version,
        "platform": platform.platform(),
        "boltzpm": boltzpm.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mesh": {
            "eps_max_eV": mesh.eps_max_eV,
            "d_eps_eV": mesh.d_eps_eV,
            "n_eps": mesh.n_eps,
            "n_theta": mesh.n_theta,
            "n_cells": mesh.n_cells,
        },
        "mixture": {
            "number_density": mixture.N,
            "hard_sphere_sigma": HS_SIGMA,
            "hard_sphere_mass_ratio": HS_MASS_RATIO,
            "ionization_threshold_eV": 5.0,
        },
        "fixed_step": fixed_meta,
        "dc": {
            "EN_Td": 10.0,
            "scheme": "upwind",
            "xi_used": dc.xi_used,
            "converged": dc.converged,
            "n_steps": dc.n_steps,
            "mean_energy": dc.mean_energy,
            "drift_velocity": dc.drift_velocity,
            "reduced_ionization_frequency": dc.reduced_ionization_frequency,
        },
        "rf": {
            "EN_rms_Td": 10.0,
            "freq_Hz": 13.56e6,
            "scheme": "upwind",
            "cycles": 3,
            "steps_per_cycle": rf.extra["steps_per_cycle"],
            "mean_energy_rms": rf.mean_energy_rms,
            "drift_velocity_rms": rf.drift_velocity_rms,
            "nu_ion_rms_over_N": rf.nu_ion_rms_over_N,
        },
    }
    (output_dir / "reference.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reference"))
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
