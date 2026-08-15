"""物理定数。内部はSI単位系を使用する。"""

from __future__ import annotations

import numpy as np

E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
K_B = 1.380649e-23
AMU = 1.66053906660e-27
TOWNSEND = 1.0e-21


def speed_from_ev(eps_ev):
    return np.sqrt(2.0 * E_CHARGE * np.asarray(eps_ev) / M_E)
