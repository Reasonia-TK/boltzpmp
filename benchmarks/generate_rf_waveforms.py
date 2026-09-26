"""陽解法RFの波形の基準（`reference/rf_waveform_reference.npz`）を作る。

0.5.0 で保存点を周期全体に等間隔に取るように変えたので、`tests/parity/test_reference.py` の RF の波形は
旧Python版の基準ではなくこの基準と比べる（最終状態は引き続き旧Python版の基準と比べる）。

使い方: uv run --extra test python benchmarks/generate_rf_waveforms.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "parity"))

from test_reference import METADATA, REFERENCE, reference_solver  # noqa: E402


def main() -> None:
    config = METADATA["rf"]
    result = reference_solver().solve_rf(
        EN_rms_Td=config["EN_rms_Td"],
        freq_Hz=config["freq_Hz"],
        scheme=config["scheme"],
        xi=0.0,
        cycles_max=config["cycles"],
        tol=0.0,
        n_store=32,
        init_n=REFERENCE["n_initial"],
        method="explicit",
    )
    output = ROOT / "reference" / "rf_waveform_reference.npz"
    np.savez_compressed(
        output,
        time=result.time_grid,
        field=result.E_t,
        mean_energy_t=result.mean_energy_t,
        drift_velocity_t=result.drift_velocity_t,
        reduced_ionization_frequency_t=result.reduced_ionization_frequency_t,
    )
    print(json.dumps({"output": str(output), "samples": len(result.time_grid)}))


if __name__ == "__main__":
    main()
