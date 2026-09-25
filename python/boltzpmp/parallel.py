"""独立した計算点を並列実行する補助API。"""

from __future__ import annotations

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

    計算はRustコアのスレッドプールで行い、その間Pythonインタープリタは解放される。
    各計算は独立した状態・作業バッファを持つので、結果は逐次計算と一致する。
    """
    return solver.solve_dc_many(EN_Td_values, max_workers=max_workers, **solve_kwargs)
