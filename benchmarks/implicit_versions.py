"""陰解法の反復回数と計算時間（VALIDATION.md の 0.5.0 の表）。

同じスクリプトを 0.4.0 と 0.5.0 の環境で実行して比べる。HF の断面積ファイル（xsecsim の BOLSIG+ 形式）の
場所は `--hf-dir` で与える（なければ HF の条件は飛ばす）。

使い方: uv run --extra test python benchmarks/implicit_versions.py [--hf-dir DIR] [--rf]
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

import boltzpmp as bp
from boltzpmp import CrossSection, Gas, Mixture, PMSolver


def elastic_gas() -> Mixture:
    section = CrossSection(kind="ELASTIC", species="G", name="el", mass_ratio=1e-3,
                           data=np.array([[0.0, 1e-19], [100.0, 1e-19]]))
    return Mixture([Gas("G", 1.0, [section])], N=3.2e22, T_K=300.0)


def dc_cases(hf_dir: Path | None):
    argon = bp.load_argon()
    yield "弾性だけの気体（m/M = 1e-3）、0.01 Td", PMSolver(elastic_gas(), eps_max_eV=0.5, d_eps_eV=0.002, n_theta=8), 0.01
    ar90 = PMSolver(argon, eps_max_eV=25.0, d_eps_eV=0.2, n_theta=90)
    yield "同梱Ar 10 Td（0.2 eV刻み、n_theta = 90）", ar90, 10.0
    yield "同梱Ar 100 Td（同上）", ar90, 100.0
    fine = PMSolver(argon, eps_max_eV=15.0, d_eps_eV=0.02, n_theta=16)
    yield "同梱Ar 1 Td（0.02 eV刻み、n_theta = 16）", fine, 1.0
    if hf_dir is None:
        return
    for tag, name in (("MT版", "HF_xsecsim_bolsig-rot_rot-mt_J0-6.txt"),
                      ("異方散乱版", "HF_xsecsim_bolsig-rot_rot-aniso_J0-6.txt")):
        sections = bp.parse_lxcat(hf_dir / name)
        mixture = bp.Mixture([bp.Gas("HF", 1.0, sections, mass_amu=20.006)], p_Pa=133.0, T_K=300.0)
        solver = PMSolver(mixture, energy_grid=bp.graded_energy_grid(20.0, 0.0025, 0.5), n_theta=16)
        for en in (1.0, 10.0, 30.0, 50.0):
            yield f"HF {tag} {en:g} Td（{solver.mesh.n_eps}セル、n_theta = 16）", solver, en


def rf_cases():
    argon = bp.load_argon()
    mesh = {"eps_max_eV": 40.0, "d_eps_eV": 0.2, "n_theta": 16}
    for pressure in (133.0, 10.0, 1.0):
        mixture = bp.Mixture(argon.gases, p_Pa=pressure, T_K=300.0)
        yield f"同梱Ar 10 Td、13.56 MHz、{pressure:g} Pa", PMSolver(mixture, **mesh), 10.0, 13.56e6
    yield "同梱Ar 100 Td、13.56 MHz、133 Pa", PMSolver(bp.Mixture(argon.gases, p_Pa=133.0, T_K=300.0), **mesh), 100.0, 13.56e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path)
    ap.add_argument("--rf", action="store_true", help="RF の条件も実行する")
    ap.add_argument("--rf-cycles", type=int, default=200)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    print(f"boltzpmp {bp.__version__}")
    print("| DC の条件 | 反復 | 時間 | 収束 | ⟨ε⟩ (eV) | w (m/s) |")
    print("|---|---:|---:|---|---:|---:|")
    for label, solver, en in dc_cases(args.hf_dir):
        t0 = time.time()
        result = solver.solve_dc(en, max_steps=20_000)
        elapsed = time.time() - t0
        print(f"| {label} | {result.n_steps} | {elapsed:.2f} s | {result.converged} | "
              f"{result.mean_energy:.6g} | {result.drift_velocity:.6g} |", flush=True)
    if not args.rf:
        return
    print("| RF の条件 | 周期 | 時間 | 収束 | 最後の残差 | ⟨ε⟩ の実効値 | w の実効値 |")
    print("|---|---:|---:|---|---:|---:|---:|")
    for label, solver, en, freq in rf_cases():
        t0 = time.time()
        result = solver.solve_rf(en, freq, n_store=64, cycles_max=args.rf_cycles)
        elapsed = time.time() - t0
        residual = result.extra["cycle_residuals"][-1]
        print(f"| {label} | {result.extra['n_cycles']} | {elapsed:.1f} s | {result.converged} | "
              f"{residual:.1e} | {result.mean_energy_rms:.6g} | {result.drift_velocity_rms:.6g} |",
              flush=True)


if __name__ == "__main__":
    main()
