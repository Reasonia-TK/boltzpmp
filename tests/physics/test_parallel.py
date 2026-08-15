from __future__ import annotations

import numpy as np
import pytest

from boltzpmp import CrossSection, Gas, Mixture, PMSolver, solve_dc_sweep


def solver() -> PMSolver:
    cross_section = CrossSection(
        kind="ELASTIC",
        species="HS",
        name="elastic",
        mass_ratio=1.36e-5,
        data=np.array([[0.0, 1e-19], [100.0, 1e-19]]),
    )
    mixture = Mixture([Gas("HS", 1.0, [cross_section])], N=3.5e22)
    return PMSolver(mixture, eps_max_eV=4.0, d_eps_eV=0.1, n_theta=20)


@pytest.mark.filterwarnings("ignore:EEPF at eps_max.*")
def test_dc_sweep_preserves_input_order_and_results() -> None:
    instance = solver()
    fields = [2.0, 5.0, 10.0]
    kwargs = {
        "scheme": "upwind",
        "tol": 0.0,
        "max_steps": 100,
        "check_every": 101,
    }
    sequential = [instance.solve_dc(value, **kwargs) for value in fields]
    parallel = solve_dc_sweep(instance, fields, max_workers=3, **kwargs)
    for expected, actual in zip(sequential, parallel, strict=True):
        np.testing.assert_allclose(actual.n, expected.n, rtol=0.0, atol=0.0)
        assert actual.extra["EN_Td"] == expected.extra["EN_Td"]


def test_dc_sweep_accepts_empty_input() -> None:
    assert solve_dc_sweep(solver(), []) == []
