"""Python APIへ公開する速度空間メッシュ。"""

from __future__ import annotations

import numpy as np

from .constants import speed_from_ev


class VelocityMesh:
    """一定エネルギー幅の軸対称 `(energy, theta)` メッシュ。"""

    def __init__(self, eps_max_eV: float, d_eps_eV: float, n_theta: int = 90) -> None:
        self.eps_max_eV = float(eps_max_eV)
        self.d_eps_eV = float(d_eps_eV)
        self.n_theta = int(n_theta)
        self.n_eps = int(round(self.eps_max_eV / self.d_eps_eV))
        if self.n_eps < 1:
            raise ValueError("eps_max_eV / d_eps_eV must be >= 1")
        if self.n_theta < 1:
            raise ValueError("n_theta must be >= 1")

        self.eps_b = np.arange(self.n_eps + 1, dtype=float) * self.d_eps_eV
        self.eps_c = (np.arange(self.n_eps, dtype=float) + 0.5) * self.d_eps_eV
        self.v_b = speed_from_ev(self.eps_b)
        self.v_c = speed_from_ev(self.eps_c)
        self.d_theta = np.pi / self.n_theta
        self.theta_b = np.arange(self.n_theta + 1, dtype=float) * self.d_theta
        self.theta_c = (np.arange(self.n_theta, dtype=float) + 0.5) * self.d_theta

        dv3 = self.v_b[1:] ** 3 - self.v_b[:-1] ** 3
        dcos = np.cos(self.theta_b[:-1]) - np.cos(self.theta_b[1:])
        self.V = (2.0 / 3.0) * np.pi * np.outer(dv3, dcos)

        sin2_lo = np.sin(self.theta_b[:-1]) ** 2
        sin2_hi = np.sin(self.theta_b[1:]) ** 2
        max_sin2 = np.maximum(sin2_lo, sin2_hi)
        straddles = (self.theta_b[:-1] <= np.pi / 2) & (
            self.theta_b[1:] >= np.pi / 2
        )
        max_sin2 = np.where(straddles, 1.0, max_sin2)
        sin2_diff = max_sin2 - np.minimum(sin2_lo, sin2_hi)
        self.S_plus_eps = np.pi * np.outer(self.v_b[1:] ** 2, sin2_diff)
        self.S_minus_eps = np.pi * np.outer(self.v_b[:-1] ** 2, sin2_diff)
        dv2 = self.v_b[1:] ** 2 - self.v_b[:-1] ** 2
        self.S_plus_theta = np.pi * np.outer(dv2, sin2_hi)
        self.S_minus_theta = np.pi * np.outer(dv2, sin2_lo)
        self.w_theta = dcos / 2.0
        self.n_cells = self.n_eps * self.n_theta
        self.shape = (self.n_eps, self.n_theta)

    def idx(self, i, j):
        return np.asarray(i) * self.n_theta + np.asarray(j)

    def unravel(self, k):
        k = np.asarray(k)
        return k // self.n_theta, k % self.n_theta
