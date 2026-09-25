from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boltzpmp import CrossSection, parse_lxcat

TEXT = """header
EXCITATION
A -> B
 1.000000e-1  3.000000e+0
PROCESS: E + A -> E + B, Excitation
-----
 0.1 0.0
 1.0 2.0e-20
 5.0 1.0e-20
-----
ROTATION
HF
 0.0 1.0
 5.126e-3 3.0
PROCESS: rot 0-1
-----
 5.126e-3 0.0 0.0
 1.0 1.0e-18 2.0e-20
-----
EXCITATION
A <-> A*
 11.5
-----
 11.5 0.0
 20.0 1.0e-21
-----
"""


def test_line_three_threshold_with_weight_ratio() -> None:
    # boltzpmp 0.1.3 では2数の行を読めず、しきい値が黙って0になっていた
    sections = parse_lxcat(TEXT)
    assert len(sections) == 3
    excitation = sections[0]
    assert excitation.threshold == pytest.approx(0.1)
    assert excitation.weight_ratio == pytest.approx(3.0)
    assert excitation.target == "A"
    assert excitation.product == "B"
    assert excitation.name == "E + A -> E + B, Excitation"


def test_rotation_block_and_momentum_transfer_column() -> None:
    rotation = parse_lxcat(TEXT)[1]
    assert rotation.kind == "ROTATION"
    assert rotation.lower_state == (0.0, 1.0)
    assert rotation.upper_state == (5.126e-3, 3.0)
    assert rotation.threshold == pytest.approx(5.126e-3)
    assert rotation.mt_data is not None
    np.testing.assert_array_equal(rotation.mt_data[:, 1], [0.0, 2.0e-20])
    assert rotation.momentum_transfer(1.0) == pytest.approx(2.0e-20)
    assert rotation.sigma(1.0) == pytest.approx(1.0e-18)


def test_reversible_arrow_is_kept_in_species() -> None:
    reversible = parse_lxcat(TEXT)[2]
    assert reversible.species == "A <-> A*"
    assert reversible.target == "A"
    assert reversible.product == "A*"


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("EXCITATION\nAr\nPROCESS: x\n-----\n 12 0\n-----\n", 3),
        ("ATTACHMENT\nX\n-----\n 1 0\n 2 abc\n-----\n", 5),
        ("ROTATION\nX\n 0 1\n 0.01\n-----\n 0.01 0\n-----\n", 4),
    ],
)
def test_malformed_blocks_report_line_numbers(text: str, line: int) -> None:
    with pytest.raises(ValueError, match=f"line {line}"):
        parse_lxcat(text)


def test_sigma_matches_numpy_interp() -> None:
    data = np.array([[0.0, 0.0], [1.0, 2.0], [1.0, 4.0], [3.0, 8.0], [7.5, 1.0]])
    section = CrossSection(kind="EXCITATION", species="A", name="x", threshold=0.5, data=data)
    eps = np.linspace(-1.0, 9.0, 401)
    expected = np.interp(eps, data[:, 0], data[:, 1], left=0.0, right=data[-1, 1])
    expected = np.where(eps < 0.5, 0.0, expected)
    # x86-64 では演算順序まで同じなので完全に一致する。numpy が積和を融合演算（FMA）で計算する
    # 環境（macOS arm64 など）では最後の1桁だけ異なるので、丸め誤差の範囲で比べる
    np.testing.assert_allclose(section.sigma(eps), expected, rtol=4 * np.finfo(float).eps, atol=0.0)


def test_invalid_tables_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        CrossSection(kind="ELASTIC", species="A", name="x", data=[[1.0, 1.0], [0.5, 1.0]])
    with pytest.raises(ValueError, match="non-negative"):
        CrossSection(kind="ELASTIC", species="A", name="x", data=[[0.0, -1.0]])
    with pytest.raises(ValueError, match="ROTATION"):
        CrossSection(kind="ROTATION", species="A", name="x", data=[[0.0, 1.0]])


def test_bundled_argon_parses_as_before() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "python" / "boltzpmp" / "data"
    sections = parse_lxcat(data_dir / "Ar.txt")
    assert [s.kind for s in sections] == ["ELASTIC", "EXCITATION", "EXCITATION", "IONIZATION"]
    assert [s.threshold for s in sections[1:]] == pytest.approx([11.55, 12.9, 15.76])
    assert sections[0].mass_ratio == pytest.approx(1.371e-5)
