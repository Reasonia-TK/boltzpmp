"""同梱の Ar で、13.56 MHz の RF 周期定常解の波形と周期平均を求める。

- `solve_rf`（既定は陰解法）で周期定常解を求め、平均エネルギー、ドリフト速度、電離周波数の波形を CSV に書く。
- 電場が最大の時刻の速度係数（`rate_coefficients`）と、周期平均（`rate_coefficients_avg`）を比べる。
  プラズマの流体モデルに渡す速度係数は周期平均を使う。
- `--plot` を付けると、波形と、保存点ごとの EEDF（`eedf_t`）の図も保存する（matplotlib が必要）。

使い方:
    python examples/02_rf_waveforms.py
    python examples/02_rf_waveforms.py --en-rms 50 --pressure 50 --plot
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import boltzpmp as bp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--en-rms", type=float, default=30.0, help="E/N の実効値 (Td)")
    parser.add_argument("--frequency", type=float, default=13.56e6, help="周波数 (Hz)")
    parser.add_argument("--pressure", type=float, default=133.0, help="気体の圧力 (Pa)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--plot", action="store_true", help="図を保存する（matplotlib が必要）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot:
        require_matplotlib()
    args.out.mkdir(parents=True, exist_ok=True)
    argon = bp.load_argon()
    mixture = bp.Mixture(argon.gases, p_Pa=args.pressure, T_K=300.0)
    solver = bp.PMSolver(mixture, eps_max_eV=60.0, d_eps_eV=0.2, n_theta=16)

    start = time.perf_counter()
    result = solver.solve_rf(args.en_rms, args.frequency, n_store=64)
    elapsed = time.perf_counter() - start
    if not result.converged:
        raise RuntimeError(
            f"not converged in {result.extra['n_cycles']} cycles "
            f"(last residual {result.extra['cycle_residuals'][-1]:.1e})"
        )
    print(f"boltzpmp {bp.__version__}: converged in {result.extra['n_cycles']} cycles "
          f"x {result.extra['steps_per_cycle']} steps ({elapsed:.1f} s)")
    print(f"mean energy: cycle average {result.mean_energy_avg:.4f} eV, "
          f"min {result.mean_energy_t.min():.4f}, max {result.mean_energy_t.max():.4f}")
    print(f"drift velocity: rms {result.drift_velocity_rms:.4g} m/s, "
          f"phase delay {np.degrees(result.phase_delay_W):.1f} deg")
    print(f"ionization frequency / N: cycle average {result.reduced_ionization_frequency_avg:.3e} m^3/s, "
          f"rms of the waveform {result.nu_ion_rms_over_N:.3e} m^3/s")
    print(f"{'process':44s} {'k at max field':>14s} {'k averaged':>12s}")
    for key, average in result.rate_coefficients_avg.items():
        print(f"{key:44s} {result.rate_coefficients[key]:14.3e} {average:12.3e}")

    period = 1.0 / args.frequency
    path = args.out / "argon_rf_waveforms.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t/T", "E (V/m)", "mean energy (eV)", "drift velocity (m/s)",
                         "ionization frequency/N (m^3/s)"])
        for row in zip(result.time_grid / period, result.E_t, result.mean_energy_t,
                       result.drift_velocity_t, result.reduced_ionization_frequency_t, strict=True):
            writer.writerow(row)
    print(f"wrote {path}")

    if args.plot:
        plot(result, period, args.out)


def require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise SystemExit("--plot には matplotlib が必要です（uv pip install matplotlib）") from error


def plot(result: bp.SwarmResultRF, period: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase = result.time_grid / period
    order = np.argsort(phase)
    fig, (waves, eedf) = plt.subplots(1, 2, figsize=(11, 4))
    field = result.E_t / np.abs(result.E_t).max()
    waves.plot(phase[order], field[order], "k--", label="E / E$_{peak}$")
    waves.plot(phase[order], result.mean_energy_t[order] / result.mean_energy_avg, label="⟨ε⟩ / ⟨ε⟩$_{avg}$")
    waves.plot(phase[order], result.drift_velocity_t[order] / np.abs(result.drift_velocity_t).max(),
               label="w / w$_{max}$")
    waves.set(xlabel="t / T", title="waveforms")
    waves.legend(fontsize=8)
    for index in np.linspace(0, len(phase) - 1, 5, dtype=int)[:-1]:
        sample = order[index]
        eedf.semilogy(result.energy_grid, result.eedf_t[sample] / np.sqrt(result.energy_grid),
                      label=f"t/T = {phase[sample]:.2f}")
    eedf.semilogy(result.energy_grid, result.eepf_avg, "k", lw=2, label="cycle average")
    eedf.set(xlabel="energy (eV)", ylabel="EEPF (eV$^{-3/2}$)", xlim=(0, 40), ylim=(1e-10, 1))
    eedf.legend(fontsize=8)
    fig.tight_layout()
    path = out / "argon_rf_waveforms.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
