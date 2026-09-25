"""断面積データ、LXCat parser、組成データ構造。

読み込み、検証、補間、混合気体の検証はRustコアが行い、ここではPythonのデータ構造へ詰め替えるだけとする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import _core

KINDS = ("ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATION")


@dataclass
class CrossSection:
    """1つの衝突過程の断面積。

    `data` は (エネルギー eV, 断面積 m²) の表。`mt_data` を与えると運動量移行断面積として
    角度分布の異方性に使い、`data` は積分断面積とみなす。ROTATION では `lower_state`、
    `upper_state` に (基底状態からのエネルギー eV, 統計重み) を与える。
    """

    kind: str
    species: str
    name: str
    threshold: float = 0.0
    mass_ratio: float | None = None
    data: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    comment: str = ""
    weight_ratio: float | None = None
    lower_state: tuple[float, float] | None = None
    upper_state: tuple[float, float] | None = None
    mt_data: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown cross-section kind: {self.kind!r}")
        self.data = _table(self.data, "cross-section data")
        if self.mt_data is not None:
            self.mt_data = _table(self.mt_data, "momentum-transfer data")
        if self.kind == "ROTATION":
            if self.lower_state is None or self.upper_state is None:
                raise ValueError("ROTATION needs lower_state and upper_state")
            self.lower_state = (float(self.lower_state[0]), float(self.lower_state[1]))
            self.upper_state = (float(self.upper_state[0]), float(self.upper_state[1]))
            self.threshold = self.upper_state[0] - self.lower_state[0]
        _core.validate_cross_section(self.to_core())

    @property
    def target(self) -> str:
        """反応式の左辺。"""
        return self.species.replace("<->", "->").split("->")[0].strip()

    @property
    def product(self) -> str | None:
        """反応式の右辺。矢印がなければ None。"""
        parts = self.species.replace("<->", "->").split("->", 1)
        return parts[1].strip() if len(parts) == 2 else None

    def sigma(self, eps_ev) -> np.ndarray:
        """しきい値未満を0とした断面積（範囲外は下側0、上側は最後の値）。"""
        return self._interp(self.data, eps_ev)

    def momentum_transfer(self, eps_ev) -> np.ndarray | None:
        return None if self.mt_data is None else self._interp(self.mt_data, eps_ev)

    def _interp(self, table: np.ndarray, eps_ev) -> np.ndarray:
        eps = np.asarray(eps_ev, dtype=float)
        values = _core.interp_sigma(
            table[:, 0].tolist(), table[:, 1].tolist(), float(self.threshold), eps.ravel().tolist()
        )
        return np.asarray(values, dtype=float).reshape(eps.shape)

    def to_core(self) -> dict[str, Any]:
        """Rustコアへ渡す辞書。"""
        return {
            "kind": self.kind,
            "species": self.species,
            "name": self.name,
            "threshold": float(self.threshold),
            "mass_ratio": None if self.mass_ratio is None else float(self.mass_ratio),
            "weight_ratio": None if self.weight_ratio is None else float(self.weight_ratio),
            "lower_state": self.lower_state,
            "upper_state": self.upper_state,
            "energy": self.data[:, 0].tolist(),
            "sigma": self.data[:, 1].tolist(),
            "mt_energy": None if self.mt_data is None else self.mt_data[:, 0].tolist(),
            "mt": None if self.mt_data is None else self.mt_data[:, 1].tolist(),
            "comment": self.comment,
        }


def _table(values, label: str) -> np.ndarray:
    table = np.asarray(values, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(f"{label} must have shape (n, 2)")
    if len(table) == 0:
        raise ValueError(f"{label} must not be empty")
    return table


@dataclass
class Gas:
    name: str
    fraction: float
    cross_sections: list[CrossSection] = field(default_factory=list)
    mass_amu: float | None = None

    def mass_ratio(self, cross_section: CrossSection) -> float:
        if cross_section.mass_ratio is not None:
            return cross_section.mass_ratio
        if self.mass_amu is not None:
            return _core.mass_ratio_from_amu(float(self.mass_amu))
        raise ValueError(f"no mass ratio available for elastic process of {self.name}")

    def to_core(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fraction": float(self.fraction),
            "mass_amu": None if self.mass_amu is None else float(self.mass_amu),
            "cross_sections": [section.to_core() for section in self.cross_sections],
        }


class Mixture:
    """混合気体。

    `T_K` は気体温度（弾性衝突による加熱と、励起準位の占有に使う）。`T_exc_K` と
    `transition_energy_eV` はBOLSIG+の Excitation temperature と Transition energy に対応し、
    `transition_energy_eV` より上の準位の占有を `T_exc_K`（既定は `T_K`）で決める。
    """

    def __init__(
        self,
        gases: list[Gas],
        p_Pa: float | None = None,
        T_K: float = 300.0,
        N: float | None = None,
        *,
        T_exc_K: float | None = None,
        transition_energy_eV: float = 0.0,
    ) -> None:
        self.gases = list(gases)
        self.T_K = float(T_K)
        self.p_Pa = p_Pa
        self.T_exc_K = None if T_exc_K is None else float(T_exc_K)
        self.transition_energy_eV = float(transition_energy_eV)
        _core.validate_fractions([float(gas.fraction) for gas in self.gases])
        self._N = _core.number_density(
            None if p_Pa is None else float(p_Pa),
            self.T_K,
            None if N is None else float(N),
        )

    @property
    def N(self) -> float:
        return self._N

    def processes(self) -> list[tuple[Gas, CrossSection]]:
        return [
            (gas, cross_section)
            for gas in self.gases
            for cross_section in gas.cross_sections
        ]


def _from_core(item: dict[str, Any]) -> CrossSection:
    energy = np.asarray(item["energy"], dtype=float)
    mt = item["mt"]
    return CrossSection(
        kind=item["kind"],
        species=item["species"],
        name=item["name"],
        threshold=item["threshold"],
        mass_ratio=item["mass_ratio"],
        data=np.column_stack([energy, np.asarray(item["sigma"], dtype=float)]),
        comment=item["comment"],
        weight_ratio=item["weight_ratio"],
        lower_state=item["lower_state"],
        upper_state=item["upper_state"],
        mt_data=None if mt is None else np.column_stack([energy, np.asarray(mt, dtype=float)]),
    )


def parse_lxcat(source: str | Path) -> list[CrossSection]:
    """LXCat形式のパス、または生テキストを読み込む。

    3行目の「しきい値 統計重み比」、反応式の `<->`、ROTATION ブロック、3列目の運動量移行断面積に
    対応する。書式の誤りは行番号付きの ValueError になる。
    """
    if isinstance(source, Path):
        items = _core.parse_lxcat_file(str(source))
    else:
        raw = str(source)
        is_path = False
        if "\n" not in raw:
            try:
                is_path = Path(raw).exists()
            except OSError:
                is_path = False
        items = _core.parse_lxcat_file(raw) if is_path else _core.parse_lxcat_text(raw)
    return [_from_core(item) for item in items]


def load_argon(
    metastable_fraction: float = 1e-4,
    p_Pa: float = 133.0,
    T_K: float = 273.0,
    N: float | None = None,
) -> Mixture:
    """同梱する近似Ar/Ar*断面積から混合気体を作る。"""
    if not 0.0 <= metastable_fraction <= 1.0:
        raise ValueError("metastable_fraction must be in [0, 1]")
    data_dir = Path(__file__).parent / "data"
    gases = [
        Gas(
            "Ar",
            1.0 - metastable_fraction,
            parse_lxcat(data_dir / "Ar.txt"),
            mass_amu=39.948,
        )
    ]
    if metastable_fraction != 0.0:
        gases.append(
            Gas(
                "Ar*",
                metastable_fraction,
                parse_lxcat(data_dir / "Ar_star.txt"),
                mass_amu=39.948,
            )
        )
    return Mixture(gases, N=N, p_Pa=None if N is not None else p_Pa, T_K=T_K)
