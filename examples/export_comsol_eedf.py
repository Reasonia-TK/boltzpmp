"""COMSOL Plasmaインターフェース向けEEDFテーブルを出力する。

COMSOLのPlasmaインターフェースで外部EEDFを使う場合は、Interpolation関数を
2引数で作成し、次の3列を持つSpreadsheet形式のテキストを読み込む。

1. 電子エネルギー (eV。COMSOLの単位欄ではV)
2. 平均電子エネルギー (eV。COMSOLの単位欄ではV)
3. COMSOLのEEDF f(epsilon) (eV^(-3/2)。単位欄ではV^(-3/2))

boltzpmpの ``result.eedf`` はエネルギー確率密度 (eV^(-1)) である。一方、
COMSOLが要求する分布関数は ``result.eepf = result.eedf / sqrt(epsilon)``
に対応するため、このサンプルでは ``result.eepf`` を出力する。

公式資料:
https://www.comsol.com/blogs/the-boltzmann-equation-two-term-approximation-interface
https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_fileformats.53.04.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boltzpmp as bp
import numpy as np

DEFAULT_FIELDS_TD = (5.0, 10.0, 20.0, 50.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COMSOL Plasma向けの2引数EEDF補間テーブルを作成します。"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comsol_eedf_argon.txt"),
        help="出力するCOMSOL Spreadsheet形式ファイル",
    )
    parser.add_argument(
        "--lxcat",
        type=Path,
        help="任意のLXCat断面積ファイル。省略時は同梱の純Arデータを使用",
    )
    parser.add_argument(
        "--fields",
        type=float,
        nargs="+",
        default=list(DEFAULT_FIELDS_TD),
        metavar="TD",
        help="計算する換算電場。2点以上を指定",
    )
    parser.add_argument("--pressure-pa", type=float, default=133.0)
    parser.add_argument("--gas-temperature-k", type=float, default=273.0)
    parser.add_argument("--eps-max-ev", type=float, default=60.0)
    parser.add_argument("--d-eps-ev", type=float, default=0.5)
    parser.add_argument("--n-theta", type=int, default=48)
    parser.add_argument("--scheme", choices=("blending", "upwind"), default="blending")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--check-every", type=int, default=200)
    parser.add_argument("--max-tail-ratio", type=float, default=1e-6)
    return parser.parse_args()


def validate_fields(fields_td: list[float]) -> list[float]:
    fields = [float(value) for value in fields_td]
    if len(fields) < 2:
        raise ValueError("COMSOLの2引数補間には2点以上の換算電場が必要です")
    if not all(np.isfinite(value) and value > 0.0 for value in fields):
        raise ValueError("換算電場は有限な正値で指定してください")
    if len(set(fields)) != len(fields):
        raise ValueError("換算電場に重複があります")
    return fields


def build_argon_mixture(args: argparse.Namespace) -> bp.Mixture:
    if args.lxcat is None:
        return bp.load_argon(
            metastable_fraction=0.0,
            p_Pa=args.pressure_pa,
            T_K=args.gas_temperature_k,
        )

    source = args.lxcat.resolve(strict=True)
    gas = bp.Gas(
        name="Ar",
        fraction=1.0,
        cross_sections=bp.parse_lxcat(source),
        mass_amu=39.948,
    )
    return bp.Mixture(
        [gas],
        p_Pa=args.pressure_pa,
        T_K=args.gas_temperature_k,
    )


def solve_eedf_sweep(
    solver: bp.PMSolver,
    fields_td: list[float],
    *,
    scheme: str,
    tol: float,
    max_steps: int,
    check_every: int,
    max_tail_ratio: float,
) -> list[bp.SwarmResult]:
    """高電場側からwarm startし、収束済み結果を平均エネルギー順で返す。"""
    results: list[bp.SwarmResult] = []
    initial_state: np.ndarray | None = None
    for field_td in sorted(fields_td, reverse=True):
        result = solver.solve_dc(
            EN_Td=field_td,
            scheme=scheme,
            tol=tol,
            max_steps=max_steps,
            check_every=check_every,
            init_n=initial_state,
        )
        if not result.converged:
            raise RuntimeError(
                f"{field_td:g} Tdが{result.n_steps}ステップで収束しませんでした"
            )
        tail_ratio = float(result.extra["eepf_tail_ratio"])
        if tail_ratio > max_tail_ratio:
            raise RuntimeError(
                f"{field_td:g} TdのEEDF末端比が{tail_ratio:.3e}です。"
                "--eps-max-evを増やしてください"
            )
        results.append(result)
        initial_state = result.n
    return sorted(results, key=lambda result: result.mean_energy)


def trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    widths = np.diff(coordinates)
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * widths))


def as_comsol_curve(result: bp.SwarmResult) -> tuple[np.ndarray, np.ndarray, float]:
    """セル中心のEEPFをCOMSOL補間用の端点付き曲線へ変換する。"""
    energy = np.asarray(result.energy_grid, dtype=float)
    eedf = np.maximum(np.asarray(result.eepf, dtype=float), 0.0)
    if energy.ndim != 1 or eedf.shape != energy.shape or energy.size < 2:
        raise ValueError("EEDFのエネルギーグリッドが不正です")
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(eedf)):
        raise ValueError("EEDFに非有限値があります")
    if not np.all(np.diff(energy) > 0.0):
        raise ValueError("エネルギーグリッドは単調増加である必要があります")

    eps_max = float(result.mesh.eps_max_eV)
    left_slope = (eedf[1] - eedf[0]) / (energy[1] - energy[0])
    right_slope = (eedf[-1] - eedf[-2]) / (energy[-1] - energy[-2])
    left_value = max(0.0, float(eedf[0] - left_slope * energy[0]))
    right_value = max(
        0.0,
        float(eedf[-1] + right_slope * (eps_max - energy[-1])),
    )
    export_energy = np.concatenate(([0.0], energy, [eps_max]))
    export_eedf = np.concatenate(([left_value], eedf, [right_value]))

    weight = np.sqrt(export_energy) * export_eedf
    normalization = trapezoid(weight, export_energy)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("COMSOL EEDFの規格化係数が正ではありません")
    export_eedf /= normalization
    exported_mean_energy = trapezoid(
        export_energy**1.5 * export_eedf,
        export_energy,
    )
    return export_energy, export_eedf, exported_mean_energy


def write_comsol_spreadsheet(
    results: list[bp.SwarmResult],
    output: Path,
) -> list[tuple[float, float]]:
    """COMSOLの2引数Interpolationで読める3列Spreadsheetを書き出す。"""
    rows: list[np.ndarray] = []
    summaries: list[tuple[float, float]] = []
    for result in results:
        energy, eedf, mean_energy = as_comsol_curve(result)
        rows.append(
            np.column_stack(
                (
                    energy,
                    np.full_like(energy, mean_energy),
                    eedf,
                )
            )
        )
        summaries.append((float(result.extra["EN_Td"]), mean_energy))

    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "COMSOL Spreadsheet data for a two-argument EEDF interpolation function\n"
        "Argument units in COMSOL: V, V; function unit: V^(-3/2)\n"
        "electron_energy_eV\tmean_electron_energy_eV\teedf_eV^-3/2"
    )
    np.savetxt(
        output,
        np.vstack(rows),
        fmt="%.17e",
        delimiter="\t",
        header=header,
        comments="% ",
        encoding="utf-8",
    )
    return summaries


def main() -> None:
    args = parse_args()
    fields_td = validate_fields(args.fields)
    mixture = build_argon_mixture(args)
    solver = bp.PMSolver(
        mixture,
        eps_max_eV=args.eps_max_ev,
        d_eps_eV=args.d_eps_ev,
        n_theta=args.n_theta,
    )
    results = solve_eedf_sweep(
        solver,
        fields_td,
        scheme=args.scheme,
        tol=args.tol,
        max_steps=args.max_steps,
        check_every=args.check_every,
        max_tail_ratio=args.max_tail_ratio,
    )
    output = args.output.resolve()
    summaries = write_comsol_spreadsheet(results, output)

    print(f"wrote {output}")
    for field_td, mean_energy in summaries:
        print(f"  E/N={field_td:g} Td, mean energy={mean_energy:.8g} eV")
    print("COMSOL: Interpolation関数をFile/Spreadsheet/2 argumentsで作成します。")
    print("引数単位をV, V、関数単位をV^(-3/2)に設定してください。")


if __name__ == "__main__":
    main()
