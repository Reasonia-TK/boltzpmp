"""LXCat断面積を使い、Python参照実装とRust実装のDC結果を比較する。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FIELDS_UPWIND_TD = (100.0, 50.0, 10.0, 1.0)
FIELDS_BLENDING_TD = (100.0, 10.0)
MESH = {"eps_max_eV": 60.0, "d_eps_eV": 0.5, "n_theta": 48}
SOLVE = {"tol": 1e-5, "max_steps": 1_000_000, "check_every": 200}
GATES = {
    "distribution_l1": 1e-5,
    "eedf_relative_l1": 1e-5,
    "scalar_relative_error": 1e-5,
    "rate_relative_error": 1e-5,
    "normalization_error": 1e-12,
    "eepf_tail_ratio": 1e-6,
    "negative_state_relative": 1e-14,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cross_section", type=Path)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def relative_error(actual: float, reference: float) -> float:
    return abs(actual - reference) / max(abs(actual), abs(reference), 1e-300)


def relative_l1(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.sum(np.abs(actual - reference))
        / max(float(np.sum(np.abs(reference))), 1e-300)
    )


def disambiguate_process_names(cross_sections: list[Any]) -> None:
    """重複ラベルの各反応速度係数を個別に検証できる名前へ変える。"""
    totals = Counter(cross_section.name for cross_section in cross_sections)
    seen: Counter[str] = Counter()
    for cross_section in cross_sections:
        original = cross_section.name
        seen[original] += 1
        if totals[original] > 1:
            cross_section.name = f"{original} [{seen[original]}]"


def parser_signature(cross_sections: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "kind": cross_section.kind,
            "species": cross_section.species,
            "name": cross_section.name,
            "threshold_eV": float(cross_section.threshold),
            "mass_ratio": (
                None
                if cross_section.mass_ratio is None
                else float(cross_section.mass_ratio)
            ),
            "rows": int(len(cross_section.data)),
            "energy_min_eV": float(cross_section.data[0, 0]),
            "energy_max_eV": float(cross_section.data[-1, 0]),
            "data_sha256": hashlib.sha256(
                np.asarray(cross_section.data, dtype="<f8").tobytes()
            ).hexdigest(),
        }
        for cross_section in cross_sections
    ]


def build_mixture(module: Any, source: Path) -> tuple[Any, list[Any]]:
    cross_sections = module.parse_lxcat(source)
    disambiguate_process_names(cross_sections)
    gas = module.Gas("Ar", 1.0, cross_sections, mass_amu=39.948)
    mixture = module.Mixture([gas], p_Pa=133.0, T_K=273.0)
    return mixture, cross_sections


def run_case(
    module: Any,
    mixture: Any,
    en_td: float,
    scheme: str,
    initial_state: np.ndarray | None,
) -> tuple[Any, float]:
    solver = module.PMSolver(mixture, **MESH)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        result = solver.solve_dc(
            EN_Td=en_td,
            scheme=scheme,
            init_n=initial_state,
            **SOLVE,
        )
    elapsed = time.perf_counter() - started
    result.extra["validation_warning_count"] = len(captured)
    return result, elapsed


def run_engine(module: Any, mixture: Any) -> dict[str, tuple[Any, float]]:
    results: dict[str, tuple[Any, float]] = {}
    initial_state: np.ndarray | None = None
    for en_td in FIELDS_UPWIND_TD:
        key = f"upwind_{en_td:g}Td"
        result, elapsed = run_case(module, mixture, en_td, "upwind", initial_state)
        results[key] = (result, elapsed)
        initial_state = result.n
    for en_td in FIELDS_BLENDING_TD:
        key = f"blending_{en_td:g}Td"
        results[key] = run_case(module, mixture, en_td, "blending", None)
    return results


def result_summary(result: Any, elapsed: float) -> dict[str, Any]:
    return {
        "converged": bool(result.converged),
        "n_steps": int(result.n_steps),
        "xi_used": float(result.xi_used),
        "seconds": elapsed,
        "mean_energy_eV": float(result.mean_energy),
        "drift_velocity_m_per_s": float(result.drift_velocity),
        "reduced_ionization_frequency_m3_per_s": float(
            result.reduced_ionization_frequency
        ),
        "rate_coefficients_m3_per_s": {
            key: float(value) for key, value in result.rate_coefficients.items()
        },
        "normalization_error": abs(float(np.sum(result.n)) - 1.0),
        "minimum_state": float(np.min(result.n)),
        "maximum_state": float(np.max(result.n)),
        "eepf_tail_ratio": float(result.extra["eepf_tail_ratio"]),
        "warning_count": int(result.extra["validation_warning_count"]),
    }


def compare_case(reference: Any, actual: Any) -> dict[str, Any]:
    scalar_errors = {
        "mean_energy": relative_error(actual.mean_energy, reference.mean_energy),
        "drift_velocity": relative_error(
            actual.drift_velocity, reference.drift_velocity
        ),
        "reduced_ionization_frequency": relative_error(
            actual.reduced_ionization_frequency,
            reference.reduced_ionization_frequency,
        ),
    }
    rate_keys_match = actual.rate_coefficients.keys() == reference.rate_coefficients.keys()
    rate_errors = {
        key: relative_error(actual.rate_coefficients[key], value)
        for key, value in reference.rate_coefficients.items()
        if key in actual.rate_coefficients
    }
    return {
        "energy_grid_exact": bool(np.array_equal(actual.energy_grid, reference.energy_grid)),
        "rate_keys_match": rate_keys_match,
        "steps_match": actual.n_steps == reference.n_steps,
        "xi_absolute_error": abs(float(actual.xi_used) - float(reference.xi_used)),
        "distribution_l1": float(np.sum(np.abs(actual.n - reference.n))),
        "eedf_relative_l1": relative_l1(actual.eedf, reference.eedf),
        "eepf_relative_l1": relative_l1(actual.eepf, reference.eepf),
        "scalar_relative_errors": scalar_errors,
        "max_scalar_relative_error": max(scalar_errors.values()),
        "rate_relative_errors": rate_errors,
        "max_rate_relative_error": max(rate_errors.values(), default=0.0),
    }


def main() -> None:
    args = parse_args()
    source = args.cross_section.resolve(strict=True)
    reference_source = args.reference_source.resolve(strict=True)
    sys.path.insert(0, str(reference_source))

    import boltzpm as reference_module
    import boltzpmp as rust_module

    reference_mixture, reference_cross_sections = build_mixture(
        reference_module, source
    )
    rust_mixture, rust_cross_sections = build_mixture(rust_module, source)
    reference_parser = parser_signature(reference_cross_sections)
    rust_parser = parser_signature(rust_cross_sections)
    parser_match = reference_parser == rust_parser

    reference_results = run_engine(reference_module, reference_mixture)
    rust_results = run_engine(rust_module, rust_mixture)
    cases: dict[str, Any] = {}
    for key in reference_results:
        reference_result, reference_seconds = reference_results[key]
        rust_result, rust_seconds = rust_results[key]
        cases[key] = {
            "reference": result_summary(reference_result, reference_seconds),
            "rust": result_summary(rust_result, rust_seconds),
            "comparison": compare_case(reference_result, rust_result),
            "speedup": reference_seconds / max(rust_seconds, 1e-300),
        }

    upwind_rust = [
        rust_results[f"upwind_{value:g}Td"][0]
        for value in reversed(FIELDS_UPWIND_TD)
    ]
    monotonic = {
        "mean_energy": all(
            left.mean_energy < right.mean_energy
            for left, right in zip(upwind_rust, upwind_rust[1:])
        ),
        "drift_velocity": all(
            left.drift_velocity < right.drift_velocity
            for left, right in zip(upwind_rust, upwind_rust[1:])
        ),
        "ionization_frequency": all(
            left.reduced_ionization_frequency
            <= right.reduced_ionization_frequency
            for left, right in zip(upwind_rust, upwind_rust[1:])
        ),
    }

    all_results = [
        summary
        for case in cases.values()
        for summary in (case["reference"], case["rust"])
    ]
    comparisons = [case["comparison"] for case in cases.values()]
    checks = {
        "parser_match": parser_match,
        "all_cases_converged": all(item["converged"] for item in all_results),
        "negative_state_gate": all(
            item["minimum_state"]
            >= -GATES["negative_state_relative"] * item["maximum_state"]
            for item in all_results
        ),
        "normalization_gate": all(
            item["normalization_error"] <= GATES["normalization_error"]
            for item in all_results
        ),
        "tail_gate": all(
            item["eepf_tail_ratio"] <= GATES["eepf_tail_ratio"]
            for item in all_results
        ),
        "distribution_gate": all(
            item["distribution_l1"] <= GATES["distribution_l1"]
            for item in comparisons
        ),
        "eedf_gate": all(
            item["eedf_relative_l1"] <= GATES["eedf_relative_l1"]
            for item in comparisons
        ),
        "scalar_gate": all(
            item["max_scalar_relative_error"] <= GATES["scalar_relative_error"]
            for item in comparisons
        ),
        "rate_gate": all(
            item["rate_keys_match"]
            and item["max_rate_relative_error"] <= GATES["rate_relative_error"]
            for item in comparisons
        ),
        "monotonic_field_response": all(monotonic.values()),
    }
    passed = all(checks.values())

    text = source.read_text(encoding="utf-8-sig")
    source_reference = next(
        (
            line.removeprefix("- ").strip()
            for line in text.splitlines()
            if line.startswith("- Morgan database")
        ),
        "Morgan database, LXCat",
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "source": {
            "filename": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "reference": source_reference,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "reference_package": importlib.metadata.version("boltzpm"),
            "rust_package": importlib.metadata.version("boltzpmp"),
        },
        "conditions": {
            "gas": "Ar",
            "fraction": 1.0,
            "p_Pa": 133.0,
            "T_K": 273.0,
            "mesh": MESH,
            "solve": SOLVE,
            "upwind_fields_Td": FIELDS_UPWIND_TD,
            "blending_fields_Td": FIELDS_BLENDING_TD,
        },
        "gates": GATES,
        "parser": {
            "match": parser_match,
            "process_count": len(rust_parser),
            "processes": rust_parser,
        },
        "monotonic": monotonic,
        "checks": checks,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
