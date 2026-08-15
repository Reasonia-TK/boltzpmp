"""独立した計算点を安全に並列実行する補助API。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from .output import SwarmResult
from .solver import PMSolver


def solve_dc_sweep(
    solver: PMSolver,
    EN_Td_values: Iterable[float],
    *,
    max_workers: int | None = None,
    **solve_kwargs: Any,
) -> list[SwarmResult]:
    """複数のDC換算電場を入力順のまま並列計算する。

    同じ ``PMSolver`` の数値コアは読み取り専用であり、各計算は独立した状態・作業
    バッファを持つ。Rust計算中はPythonインタープリタからdetachされるため、CPU
    バウンドな計算も複数スレッドで同時に進行できる。
    """
    values = [float(value) for value in EN_Td_values]
    if not values:
        return []
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive or None")

    def solve(value: float) -> SwarmResult:
        return solver.solve_dc(value, **solve_kwargs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(solve, values))
