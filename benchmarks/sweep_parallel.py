"""独立Rust計算をPythonスレッドで並列実行したときのスケーリング測定。"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from boltzpm2 import PMSolver, load_argon
from boltzpm2.constants import E_CHARGE, M_E, TOWNSEND


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument(
        "--output", type=Path, default=Path("reference/sweep_parallel.json")
    )
    args = parser.parse_args()

    mixture = load_argon()
    solver = PMSolver(
        mixture,
        eps_max_eV=8.0,
        d_eps_eV=8.0 / 600.0,
        n_theta=90,
        parallel=False,
    )
    initial = np.asarray(solver._core_solver.initial_maxwell(1.0)).tolist()
    acceleration = E_CHARGE * (0.33 * TOWNSEND * mixture.N) / M_E
    dt = solver._core_solver.auto_dt(acceleration)

    def run(_: int) -> float:
        state = solver._core_solver.fixed_steps(
            initial, acceleration, dt, 0.0, 1, args.steps
        )
        return float(sum(state))

    start = time.perf_counter()
    sequential_sums = [run(index) for index in range(args.jobs)]
    sequential_seconds = time.perf_counter() - start

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        parallel_sums = list(executor.map(run, range(args.jobs)))
    parallel_seconds = time.perf_counter() - start

    result = {
        "schema_version": 1,
        "jobs": args.jobs,
        "workers": args.workers,
        "steps_per_job": args.steps,
        "n_cells": solver.mesh.n_cells,
        "sequential_seconds": sequential_seconds,
        "parallel_seconds": parallel_seconds,
        "speedup": sequential_seconds / parallel_seconds,
        "sequential_sums": sequential_sums,
        "parallel_sums": parallel_sums,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
