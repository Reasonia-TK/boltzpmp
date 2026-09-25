"""RFの陰解法と陽解法の比較（VALIDATION.md の 0.4.0 の表）。

使い方: uv run --extra test python benchmarks/rf_implicit.py [--explicit-cycles 3000]
陽解法は既定の初期状態（Maxwell分布）から、周期ごとの平均エネルギー波形の変化が1e-6未満になるまで進める。
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

import boltzpmp as bp
from boltzpmp import CrossSection, Gas, Mixture, PMSolver


def synthetic() -> Mixture:
    table = lambda values: np.array(values, dtype=float)  # noqa: E731
    sections = [
        CrossSection(kind="ELASTIC", species="G", name="el", mass_ratio=1e-4,
                     data=table([(0.0, 3e-20), (100.0, 3e-20)])),
        CrossSection(kind="EXCITATION", species="G", name="exc", threshold=0.3,
                     data=table([(0.3, 0.0), (0.6, 3e-20), (40.0, 3e-20)])),
        CrossSection(kind="IONIZATION", species="G", name="ion", threshold=4.0,
                     data=table([(4.0, 0.0), (6.0, 2e-20), (40.0, 2e-20)])),
    ]
    return Mixture([Gas("G", 1.0, sections)], N=3.2e22, T_K=300.0)


def cases():
    argon = bp.load_argon()
    ar_mesh = {"eps_max_eV": 40.0, "d_eps_eV": 0.2, "n_theta": 16}
    yield "合成気体 60 Td、50 MHz", PMSolver(synthetic(), eps_max_eV=30.0, d_eps_eV=0.1, n_theta=8), 60.0, 5e7
    yield "同梱Ar 100 Td、13.56 MHz", PMSolver(argon, **ar_mesh), 100.0, 13.56e6
    yield "同梱Ar 10 Td、13.56 MHz", PMSolver(argon, **ar_mesh), 10.0, 13.56e6
    low = bp.Mixture(argon.gases, p_Pa=10.0, T_K=300.0)
    yield "同梱Ar 10 Td、13.56 MHz、10 Pa", PMSolver(low, **ar_mesh), 10.0, 13.56e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explicit-cycles", type=int, default=3000)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    print("| 条件 | 陽解法（周期 × 段、時間） | 陰解法（周期 × 段、時間） | ⟨ε⟩ の比 | w の比 |")
    print("|---|---:|---:|---:|---:|")
    for label, solver, en, freq in cases():
        t0 = time.time()
        implicit = solver.solve_rf(en, freq, n_store=64)
        t_implicit = time.time() - t0
        t0 = time.time()
        explicit = solver.solve_rf(en, freq, n_store=64, method="explicit", tol=1e-6,
                                   cycles_max=args.explicit_cycles)
        t_explicit = time.time() - t0
        mark = "" if explicit.converged else "（未収束）"
        print(f"| {label} | {explicit.extra['n_cycles']} × {explicit.extra['steps_per_cycle']}、"
              f"{t_explicit:.1f} s{mark} | {implicit.extra['n_cycles']} × {implicit.extra['steps_per_cycle']}、"
              f"{t_implicit:.1f} s | {implicit.mean_energy_rms / explicit.mean_energy_rms:.5f} | "
              f"{implicit.drift_velocity_rms / explicit.drift_velocity_rms:.5f} |", flush=True)


if __name__ == "__main__":
    main()
