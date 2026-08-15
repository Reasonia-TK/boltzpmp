"""Rustコアの固定ステップ性能を旧Python版と同条件で測定する。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from boltzpm2 import PMSolver, load_argon
from boltzpm2.constants import E_CHARGE, M_E, TOWNSEND


CASES = {
    "small": {"eps_max_eV": 6.0, "d_eps_eV": 0.03, "n_theta": 30, "steps": 20_000},
    "representative": {"eps_max_eV": 8.0, "d_eps_eV": 8.0 / 600.0, "n_theta": 90, "steps": 5_000},
}


def run_case(name: str, repeats: int, parallel: bool | None) -> dict:
    config = CASES[name]
    mixture = load_argon()
    solver = PMSolver(
        mixture,
        eps_max_eV=config["eps_max_eV"],
        d_eps_eV=config["d_eps_eV"],
        n_theta=config["n_theta"],
        parallel=parallel,
    )
    initial = np.asarray(solver._core_solver.initial_maxwell(1.0))
    electric_field = 0.33 * TOWNSEND * mixture.N
    acceleration = E_CHARGE * electric_field / M_E
    dt = solver._core_solver.auto_dt(acceleration)

    durations: list[float] = []
    final_state: np.ndarray | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        final_state = np.asarray(
            solver._core_solver.fixed_steps(
                initial.tolist(), acceleration, dt, 0.0, 1, config["steps"]
            )
        )
        durations.append(time.perf_counter() - start)

    assert final_state is not None
    median = statistics.median(durations)
    cells = solver.mesh.n_cells
    return {
        "case": name,
        **config,
        "n_eps": solver.mesh.n_eps,
        "n_cells": cells,
        "repeats": repeats,
        "parallel": solver.parallel,
        "durations_seconds": durations,
        "median_seconds": median,
        "steps_per_second": config["steps"] / median,
        "cell_updates_per_second": config["steps"] * cells / median,
        "final_sum": float(final_state.sum()),
        "final_min": float(final_state.min()),
        "final_max": float(final_state.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--parallel", choices=["auto", "true", "false"], default="auto"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reference/baseline_rust.json")
    )
    args = parser.parse_args()
    names = list(CASES) if args.case == "all" else [args.case]
    parallel = {"auto": None, "true": True, "false": False}[args.parallel]
    result = {
        "schema_version": 1,
        "implementation": "boltzpm2-rust-release",
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "cases": [run_case(name, args.repeats, parallel) for name in names],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
