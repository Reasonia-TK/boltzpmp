"""分子気体の模型: 回転励起と逆過程、気体温度、異方散乱、付着。

断面積は、この例のために作った架空の分子 M（質量 28 u）のもの（実在の気体の値ではない）。LXCat の書式で、
次を含む。

- ELASTIC と ROTATION の表の3列目（運動量移行断面積）: 積分断面積との比から、前方に偏った角度分布
  （遮蔽 Rutherford 型）を作って異方散乱として扱う。
- ROTATION ブロック（J = 0〜3）: 準位の占有を気体温度の Boltzmann 分布で求め、逆過程（超弾性衝突）も入れる。
- 振動励起（2準位系として逆過程も入れる）、電子励起、電離、解離性付着。

次の4つを比べる（DC、既定の陰解法）。

1. 既定（超弾性衝突と気体温度の効果あり、異方散乱）
2. 超弾性衝突なし（低い E/N で電子が気体温度より冷える）
3. 気体温度 1000 K
4. 3列目を捨てて等方散乱にしたもの（運動量移行ではなく積分断面積で散乱させることになり、ドリフト速度が小さくなる）

最後に、13.56 MHz の RF の周期平均の付着周波数も求める。

使い方:
    python examples/04_molecular_gas.py
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np

import boltzpmp as bp

LXCAT = """
ELASTIC
M
 1.96e-5
PROCESS: E + M -> E + M, Elastic
-----
 0.0    1.0e-19  1.0e-19
 0.1    9.0e-20  8.0e-20
 1.0    1.0e-19  7.0e-20
 5.0    1.2e-19  6.0e-20
 20.0   8.0e-20  3.0e-20
-----
ROTATION
M
 0.0    1.0
 0.002  3.0
PROCESS: E + M(J=0) -> E + M(J=1), Rotation
-----
 0.002  0.0      0.0
 0.01   4.0e-19  2.0e-20
 0.1    2.0e-19  6.0e-21
 1.0    6.0e-20  1.5e-21
 20.0   5.0e-21  1.0e-22
-----
ROTATION
M
 0.002  3.0
 0.006  5.0
PROCESS: E + M(J=1) -> E + M(J=2), Rotation
-----
 0.004  0.0      0.0
 0.01   3.0e-19  1.5e-20
 0.1    1.5e-19  4.5e-21
 1.0    4.5e-20  1.1e-21
 20.0   4.0e-21  8.0e-23
-----
ROTATION
M
 0.006  5.0
 0.012  7.0
PROCESS: E + M(J=2) -> E + M(J=3), Rotation
-----
 0.006  0.0      0.0
 0.01   2.5e-19  1.2e-20
 0.1    1.3e-19  4.0e-21
 1.0    4.0e-20  1.0e-21
 20.0   3.5e-21  7.0e-23
-----
EXCITATION
M -> M(v=1)
 0.29  1.0
PROCESS: E + M -> E + M(v=1), Vibrational
-----
 0.29   0.0
 1.0    1.0e-20
 2.5    4.0e-20
 4.0    1.0e-20
 20.0   1.0e-21
-----
EXCITATION
M -> M*
 6.5
PROCESS: E + M -> E + M*, Excitation
-----
 6.5    0.0
 10.0   5.0e-21
 20.0   1.0e-20
-----
IONIZATION
M -> M^+
 12.0
PROCESS: E + M -> E + E + M^+, Ionization
-----
 12.0   0.0
 20.0   1.0e-20
-----
ATTACHMENT
M -> M^-
PROCESS: E + M -> M^-, Attachment
-----
 1.0    0.0
 2.0    2.0e-22
 3.0    0.0
-----
"""

FIELDS_TD = (0.1, 1.0, 10.0, 50.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pressure", type=float, default=133.0, help="気体の圧力 (Pa)")
    return parser.parse_args()


def mixture(sections: list[bp.CrossSection], pressure: float, temperature: float) -> bp.Mixture:
    return bp.Mixture([bp.Gas("M", 1.0, sections, mass_amu=28.0)], p_Pa=pressure, T_K=temperature)


def isotropic(sections: list[bp.CrossSection]) -> list[bp.CrossSection]:
    """表の3列目（運動量移行断面積）を捨てた断面積（等方散乱として扱われる）。"""
    return [dataclasses.replace(section, mt_data=None) for section in sections]


def main() -> None:
    args = parse_args()
    sections = bp.parse_lxcat(LXCAT)
    grid = bp.graded_energy_grid(30.0, 0.002, 0.2)
    variants = {
        "default": bp.PMSolver(mixture(sections, args.pressure, 300.0), energy_grid=grid, n_theta=12),
        "no superelastic": bp.PMSolver(mixture(sections, args.pressure, 300.0), energy_grid=grid,
                                       n_theta=12, superelastic=False),
        "gas at 1000 K": bp.PMSolver(mixture(sections, args.pressure, 1000.0), energy_grid=grid,
                                     n_theta=12),
        "isotropic": bp.PMSolver(mixture(isotropic(sections), args.pressure, 300.0), energy_grid=grid,
                                 n_theta=12),
    }
    default = variants["default"]
    print(f"boltzpmp {bp.__version__}: {default.mesh.n_eps} energy cells x {default.mesh.n_theta} angles")
    print("processes (including the reverse processes built from the level populations):")
    for process in default.processes():
        print(f"  {process['name']:45s} anisotropic={process['anisotropic']}")

    kt = 1.380649e-23 * 300.0 / 1.602176634e-19
    print(f"\nmean energy (eV); 3/2 kT at 300 K = {1.5 * kt:.4f} eV")
    print(f"{'E/N (Td)':>9} " + " ".join(f"{name:>16s}" for name in variants))
    drift = {}
    for en in FIELDS_TD:
        row = []
        for name, solver in variants.items():
            result = solver.solve_dc(en)
            if not result.converged:
                raise RuntimeError(f"{name}, {en} Td: not converged ({result.extra['two_term_error']})")
            row.append(result.mean_energy)
            drift.setdefault(name, []).append(result.drift_velocity)
        print(f"{en:9.1f} " + " ".join(f"{value:16.4f}" for value in row))
    print("\ndrift velocity (m/s)")
    for index, en in enumerate(FIELDS_TD):
        print(f"{en:9.1f} " + " ".join(f"{drift[name][index]:16.4g}" for name in variants))

    # RF は1周期に256段を解くので、粗い格子にする（50 Td では 10 meV より下の構造は効かない）
    coarse = bp.PMSolver(mixture(sections, args.pressure, 300.0),
                         energy_grid=bp.graded_energy_grid(30.0, 0.01, 0.2), n_theta=8)
    start = time.perf_counter()
    rf = coarse.solve_rf(50.0, 13.56e6, n_store=64)
    elapsed = time.perf_counter() - start
    dc = coarse.solve_dc(50.0)
    print(f"\nRF 13.56 MHz, E_rms/N = 50 Td ({rf.extra['n_cycles']} cycles, {elapsed:.1f} s, "
          f"converged={rf.converged}):")
    print(f"  cycle-averaged mean energy {rf.mean_energy_avg:.4f} eV (DC at 50 Td: {dc.mean_energy:.4f} eV)")
    print(f"  cycle-averaged attachment frequency / N {rf.reduced_attachment_frequency_avg:.3e} m^3/s "
          f"(DC: {dc.reduced_attachment_frequency:.3e} m^3/s)")
    rotation = [key for key in rf.rate_coefficients_avg if "Rotation" in key and "superelastic" not in key]
    for key in rotation[:2]:
        print(f"  {key}: k_avg = {rf.rate_coefficients_avg[key]:.3e} m^3/s")
    if np.isnan(rf.mean_energy_avg):
        raise RuntimeError("the RF result has no cycle average")


if __name__ == "__main__":
    main()
