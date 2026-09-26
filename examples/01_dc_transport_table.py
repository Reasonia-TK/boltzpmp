"""同梱の Ar で、DC の輸送係数と速度係数の表を作る。

- `solve_dc_many` で複数の E/N を Rust のスレッドで並列に解く（既定は陰解法）。
- 平均エネルギー、ドリフト速度、換算移動度 μN、換算電離係数 α/N、過程ごとの速度係数を CSV に書く。
- `--plot` を付けると、EEPF と輸送係数の図も保存する（matplotlib が必要）。

使い方:
    python examples/01_dc_transport_table.py
    python examples/01_dc_transport_table.py --plot --out examples/output
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import boltzpmp as bp
from boltzpmp.constants import TOWNSEND

FIELDS_TD = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--plot", action="store_true", help="図を保存する（matplotlib が必要）")
    parser.add_argument("--workers", type=int, default=None, help="並列に解くスレッド数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot:
        require_matplotlib()
    args.out.mkdir(parents=True, exist_ok=True)
    mixture = bp.load_argon()
    # 低エネルギーは細かく、上は速度の刻みが一定になるように広げる格子
    solver = bp.PMSolver(
        mixture, energy_grid=bp.graded_energy_grid(120.0, 0.02, 2.0), n_theta=16
    )
    print(f"boltzpmp {bp.__version__}: {solver.mesh.n_eps} energy cells x {solver.mesh.n_theta} angles")

    start = time.perf_counter()
    results = solver.solve_dc_many(FIELDS_TD, max_workers=args.workers)
    elapsed = time.perf_counter() - start
    print(f"solved {len(results)} E/N values in {elapsed:.2f} s")

    keys = sorted(results[0].rate_coefficients)
    path = args.out / "argon_dc_transport.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["E/N (Td)", "mean energy (eV)", "drift velocity (m/s)", "mu*N (1/(V m s))",
             "alpha/N (m^2)", "iterations", *[f"k[{key}] (m^3/s)" for key in keys]]
        )
        print(f"{'E/N':>7} {'<eps>':>9} {'w':>10} {'alpha/N':>10} {'it':>4}")
        for en, result in zip(FIELDS_TD, results, strict=True):
            if not result.converged:
                raise RuntimeError(f"{en} Td did not converge in {result.n_steps} iterations")
            field = en * TOWNSEND * mixture.N
            mobility_n = result.drift_velocity / field * mixture.N
            writer.writerow(
                [en, result.mean_energy, result.drift_velocity, mobility_n, result.alpha_over_N,
                 result.n_steps, *[result.rate_coefficients[key] for key in keys]]
            )
            print(f"{en:7.1f} {result.mean_energy:9.4f} {result.drift_velocity:10.4g} "
                  f"{result.alpha_over_N:10.3g} {result.n_steps:4d}")
    print(f"wrote {path}")

    if args.plot:
        plot(results, args.out)


def require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise SystemExit("--plot には matplotlib が必要です（uv pip install matplotlib）") from error


def plot(results: list[bp.SwarmResult], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for en, result in zip(FIELDS_TD, results, strict=True):
        if en in (1.0, 10.0, 100.0, 300.0):
            left.semilogy(result.energy_grid, result.eepf, label=f"{en:g} Td")
    left.set(xlabel="energy (eV)", ylabel="EEPF (eV$^{-3/2}$)", xlim=(0, 60), ylim=(1e-12, 10))
    left.legend()
    right.loglog(FIELDS_TD, [r.mean_energy for r in results], "o-", label="mean energy (eV)")
    right.loglog(FIELDS_TD, [r.drift_velocity / 1e4 for r in results], "s-",
                 label="drift velocity (10$^4$ m/s)")
    right.set(xlabel="E/N (Td)")
    right.legend()
    fig.tight_layout()
    path = out / "argon_dc_transport.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
