"""断面積データ、LXCat parser、組成データ構造。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import AMU, K_B, M_E

KINDS = ("ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT")


@dataclass
class CrossSection:
    kind: str
    species: str
    name: str
    threshold: float = 0.0
    mass_ratio: float | None = None
    data: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    comment: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown cross-section kind: {self.kind!r}")
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2 or self.data.shape[1] != 2:
            raise ValueError("cross-section data must have shape (n, 2)")
        if len(self.data) == 0:
            raise ValueError("cross-section data must not be empty")

    def sigma(self, eps_ev) -> np.ndarray:
        eps = np.asarray(eps_ev, dtype=float)
        energy, values = self.data[:, 0], self.data[:, 1]
        result = np.interp(eps, energy, values, left=0.0, right=values[-1])
        if self.threshold > 0.0:
            result = np.where(eps < self.threshold, 0.0, result)
        return result


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
            return M_E / (self.mass_amu * AMU)
        raise ValueError(f"no mass ratio available for elastic process of {self.name}")


class Mixture:
    def __init__(
        self,
        gases: list[Gas],
        p_Pa: float | None = None,
        T_K: float = 300.0,
        N: float | None = None,
    ) -> None:
        self.gases = list(gases)
        self.T_K = float(T_K)
        self.p_Pa = p_Pa
        total = sum(gas.fraction for gas in self.gases)
        if not np.isclose(total, 1.0, rtol=1e-6):
            raise ValueError(f"mole fractions sum to {total}, expected 1")
        if N is not None:
            self._N = float(N)
        elif p_Pa is not None:
            self._N = float(p_Pa) / (K_B * self.T_K)
        else:
            raise ValueError("give either N or p_Pa (with T_K)")
        if not np.isfinite(self._N) or self._N <= 0.0:
            raise ValueError("number density must be finite and positive")

    @property
    def N(self) -> float:
        return self._N

    def processes(self) -> list[tuple[Gas, CrossSection]]:
        return [
            (gas, cross_section)
            for gas in self.gases
            for cross_section in gas.cross_sections
        ]


_DASH_RE = re.compile(r"^-{5,}$")


def _is_dashed(line: str) -> bool:
    return bool(_DASH_RE.match(line.strip()))


def _try_float(token: str) -> float | None:
    try:
        return float(token.strip())
    except ValueError:
        return None


def parse_lxcat(source: str | Path) -> list[CrossSection]:
    """LXCat形式のパス、または生テキストを読み込む。"""
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8-sig")
    else:
        raw = str(source)
        is_path = False
        if "\n" not in raw:
            try:
                is_path = Path(raw).exists()
            except OSError:
                is_path = False
        text = Path(raw).read_text(encoding="utf-8-sig") if is_path else raw.lstrip("\ufeff")

    lines = text.splitlines()
    results: list[CrossSection] = []
    index = 0
    while index < len(lines):
        kind = lines[index].strip()
        if kind not in KINDS:
            index += 1
            continue
        index += 1
        if index >= len(lines):
            break
        species = lines[index].strip()
        index += 1

        threshold = 0.0
        mass_ratio: float | None = None
        if index < len(lines):
            candidate = _try_float(lines[index])
            if candidate is not None:
                if kind in ("ELASTIC", "EFFECTIVE"):
                    mass_ratio = candidate
                else:
                    threshold = candidate
                index += 1

        name: str | None = None
        comments: list[str] = []
        while index < len(lines) and not _is_dashed(lines[index]):
            metadata = lines[index].strip()
            if metadata.upper().startswith("PROCESS:"):
                name = metadata.split(":", 1)[1].strip()
            elif metadata.upper().startswith("COMMENT:"):
                comments.append(metadata.split(":", 1)[1].strip())
            index += 1
        if index >= len(lines):
            break
        index += 1

        rows: list[tuple[float, float]] = []
        while index < len(lines) and not _is_dashed(lines[index]):
            parts = lines[index].split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
            index += 1
        if index < len(lines):
            index += 1
        if rows:
            results.append(
                CrossSection(
                    kind=kind,
                    species=species,
                    name=name or f"{species} {kind.lower()}",
                    threshold=threshold,
                    mass_ratio=mass_ratio,
                    data=np.asarray(rows, dtype=float),
                    comment="\n".join(comments),
                )
            )
    return results


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
