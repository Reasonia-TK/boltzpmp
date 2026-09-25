"""Python APIへ公開する速度空間メッシュ（計算はRustコア）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core


class VelocityMesh:
    """軸対称 `(energy, theta)` メッシュ。

    `VelocityMesh(eps_max_eV, d_eps_eV, n_theta)` は一定エネルギー幅、
    `VelocityMesh.from_edges(edges_eV, n_theta)` は任意のエネルギー境界（0 eVから）の格子。
    非一様格子では `d_eps_eV` は None で、セル幅は `d_eps` にある。
    """

    def __init__(self, eps_max_eV: float, d_eps_eV: float, n_theta: int = 90) -> None:
        self._assign(_core.mesh_data(float(eps_max_eV), float(d_eps_eV), int(n_theta)))

    @classmethod
    def from_edges(cls, edges_eV, n_theta: int = 90) -> VelocityMesh:
        edges = np.asarray(edges_eV, dtype=float).ravel()
        return cls._from_data(_core.mesh_data_from_edges(edges.tolist(), int(n_theta)))

    @classmethod
    def _from_data(cls, data: dict[str, Any]) -> VelocityMesh:
        mesh = cls.__new__(cls)
        mesh._assign(data)
        return mesh

    def _assign(self, data: dict[str, Any]) -> None:
        self.eps_max_eV = float(data["eps_max_eV"])
        self.d_eps_eV = None if data["d_eps_eV"] is None else float(data["d_eps_eV"])
        self.n_eps = int(data["n_eps"])
        self.n_theta = int(data["n_theta"])
        self.n_cells = int(data["n_cells"])
        self.shape = (self.n_eps, self.n_theta)
        self.d_theta = float(data["d_theta"])
        for name in ("eps_b", "eps_c", "d_eps", "v_b", "v_c", "theta_b", "theta_c", "w_theta"):
            setattr(self, name, np.asarray(data[name], dtype=float))
        for name in ("V", "S_plus_eps", "S_minus_eps", "S_plus_theta", "S_minus_theta"):
            setattr(self, name, np.asarray(data[name], dtype=float).reshape(self.shape))

    @property
    def is_uniform(self) -> bool:
        return self.d_eps_eV is not None

    def idx(self, i, j):
        return np.asarray(i) * self.n_theta + np.asarray(j)

    def unravel(self, k):
        k = np.asarray(k)
        return k // self.n_theta, k % self.n_theta


def graded_energy_grid(eps_max_eV: float, d_eps_min_eV: float, eps_uniform_eV: float) -> np.ndarray:
    """低エネルギー側を細かくしたエネルギー境界（`PMSolver(energy_grid=...)`用）。

    `eps_uniform_eV` までは `d_eps_min_eV` の一様刻み、その上は刻みを `sqrt(eps)` に比例して広げる。
    回転・振動のしきい値が小さい分子では、一様格子より少ないセル数と大きな時間刻みで同じ精度が出る。
    """
    return np.asarray(
        _core.graded_energy_edges(float(eps_max_eV), float(d_eps_min_eV), float(eps_uniform_eV)),
        dtype=float,
    )
