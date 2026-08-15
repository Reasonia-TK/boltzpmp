"""boltzpm2移植前後で同じ条件を測る固定ステップベンチマーク。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from boltzpm import load_argon
from boltzpm.solver import PMSolver


CASES = {
    "small": {"eps_max_eV": 6.0, "d_eps_eV": 0.03, "n_theta": 30, "steps": 20_000},
    "representative": {"eps_max_eV": 8.0, "d_eps_eV": 8.0 / 600.0, "n_theta": 90, "steps": 5_000},
}


def run_case(name: str, repeats: int) -> dict:
    config = CASES[name]
    mixture = load_argon()
    solver = PMSolver(
        mixture,
        eps_max_eV=config["eps_max_eV"],
        d_eps_eV=config["d_eps_eV"],
        n_theta=config["n_theta"],
    )
    n0 = solver._init_n("maxwell", 1.0)
    field_townsend = 0.33
    number_density = mixture.N
    electric_field = field_townsend * 1.0e-21 * number_density
    acceleration = 1.602176634e-19 * electric_field / 9.1093837015e-31
    dt = solver._auto_dt(acceleration)

    durations = []
    final_n = None
    for _ in range(repeats):
        start = time.perf_counter()
        final_n, _, _, went_negative = solver._march_dc(
            n0,
            acceleration,
            dt,
            0.0,
            0.0,
            config["steps"],
            config["steps"] + 1,
        )
        durations.append(time.perf_counter() - start)
        if went_negative:
            raise RuntimeError(f"{name}: upwind reference became negative")

    median = statistics.median(durations)
    cells = solver.mesh.n_cells
    return {
        "case": name,
        **config,
        "n_eps": solver.mesh.n_eps,
        "n_cells": cells,
        "repeats": repeats,
        "durations_seconds": durations,
        "median_seconds": median,
        "steps_per_second": config["steps"] / median,
        "cell_updates_per_second": config["steps"] * cells / median,
        "final_sum": float(final_n.sum()),
        "final_min": float(final_n.min()),
        "final_max": float(final_n.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reference/baseline.json"))
    args = parser.parse_args()

    names = list(CASES) if args.case == "all" else [args.case]
    result = {
        "schema_version": 1,
        "implementation": "boltzpm-python",
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "cases": [run_case(name, args.repeats) for name in names],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
