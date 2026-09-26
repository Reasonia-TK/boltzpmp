"""RF の周波数を変えて、準静的な応答から実効電場の極限への移り変わりを見る。

電場の実効値 E_rms/N を固定して、周波数を 100 kHz から 1 GHz まで変える（`solve_rf_many` で並列に解く）。
ω がエネルギー緩和の周波数よりずっと大きい高周波では、分布が1周期にわずかしか変わらないので、収束に
百周期以上かかる（1 GHz で約180周期。0.5.0 の既知の限界）。

- 低い周波数（エネルギー緩和より遅い）では、分布は各時刻の電場の DC 解に追従する。平均エネルギーは周期の中で
  大きく変わり、その範囲は E_peak = √2 E_rms の DC 解と 0 の間になる。
- 周波数を上げると分布の変調は小さくなり、周期平均の平均エネルギーは E_rms の DC 解に近づく。
- さらに上げて ω が運動量移行の周波数 ν_m を超えると、実効電場 E_eff = E_rms ν_m/√(ν_m² + ω²) が下がり、
  平均エネルギーも下がる。ドリフト速度の位相の遅れは 180° から 90° に近づく（慣性で電場に追従しなくなる）。

使い方:
    python examples/03_rf_frequency_scan.py
    python examples/03_rf_frequency_scan.py --plot --out examples/output
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import boltzpmp as bp

FREQUENCIES_HZ = (1e5, 1e6, 1e7, 13.56e6, 1e8, 3e8, 1e9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--en-rms", type=float, default=20.0, help="E/N の実効値 (Td)")
    parser.add_argument("--pressure", type=float, default=133.0, help="気体の圧力 (Pa)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--plot", action="store_true", help="図を保存する（matplotlib が必要）")
    parser.add_argument("--workers", type=int, default=None, help="並列に解くスレッド数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot:
        require_matplotlib()
    args.out.mkdir(parents=True, exist_ok=True)
    argon = bp.load_argon()
    mixture = bp.Mixture(argon.gases, p_Pa=args.pressure, T_K=300.0)
    solver = bp.PMSolver(mixture, eps_max_eV=40.0, d_eps_eV=0.1, n_theta=16)

    rms_dc = solver.solve_dc(args.en_rms)
    peak_dc = solver.solve_dc(np.sqrt(2.0) * args.en_rms)
    print(f"boltzpmp {bp.__version__}: DC mean energy at E_rms {rms_dc.mean_energy:.4f} eV, "
          f"at E_peak {peak_dc.mean_energy:.4f} eV")

    start = time.perf_counter()
    results = solver.solve_rf_many(
        [args.en_rms] * len(FREQUENCIES_HZ), FREQUENCIES_HZ, n_store=64, cycles_max=400,
        max_workers=args.workers,
    )
    elapsed = time.perf_counter() - start
    print(f"solved {len(results)} frequencies in {elapsed:.1f} s")

    path = args.out / "argon_rf_frequency_scan.csv"
    rows = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency (Hz)", "cycles", "converged", "mean energy avg (eV)",
                         "mean energy min (eV)", "mean energy max (eV)", "drift velocity rms (m/s)",
                         "drift phase delay (deg)", "ionization frequency/N avg (m^3/s)"])
        print(f"{'f (Hz)':>9} {'cycles':>6} {'<e>avg':>8} {'<e>min':>8} {'<e>max':>8} "
              f"{'w_rms':>9} {'phase':>7}")
        for frequency, result in zip(FREQUENCIES_HZ, results, strict=True):
            row = [frequency, result.extra["n_cycles"], result.converged, result.mean_energy_avg,
                   result.mean_energy_t.min(), result.mean_energy_t.max(), result.drift_velocity_rms,
                   np.degrees(result.phase_delay_W) % 360.0, result.reduced_ionization_frequency_avg]
            writer.writerow(row)
            rows.append(row)
            print(f"{frequency:9.3g} {row[1]:6d} {row[3]:8.4f} {row[4]:8.4f} {row[5]:8.4f} "
                  f"{row[6]:9.4g} {row[7]:7.1f}")
    print(f"wrote {path}")
    not_converged = [f for f, r in zip(FREQUENCIES_HZ, results, strict=True) if not r.converged]
    if not_converged:
        print(f"warning: not converged at {not_converged} Hz (increase cycles_max)")

    if args.plot:
        plot(rows, rms_dc.mean_energy, peak_dc.mean_energy, args.out)


def require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise SystemExit("--plot には matplotlib が必要です（uv pip install matplotlib）") from error


def plot(rows: list[list], rms_dc: float, peak_dc: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = np.array([row[0] for row in rows])
    fig, (energy, phase) = plt.subplots(1, 2, figsize=(10, 4))
    energy.semilogx(frequency, [row[3] for row in rows], "o-", label="cycle average")
    energy.fill_between(frequency, [row[4] for row in rows], [row[5] for row in rows], alpha=0.3,
                        label="min - max over the cycle")
    energy.axhline(rms_dc, color="k", ls="--", label="DC at E$_{rms}$")
    energy.axhline(peak_dc, color="gray", ls=":", label="DC at E$_{peak}$")
    energy.set(xlabel="frequency (Hz)", ylabel="mean energy (eV)")
    energy.legend(fontsize=8)
    phase.semilogx(frequency, [row[7] for row in rows], "s-")
    phase.set(xlabel="frequency (Hz)", ylabel="drift velocity phase delay (deg)")
    fig.tight_layout()
    path = out / "argon_rf_frequency_scan.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
