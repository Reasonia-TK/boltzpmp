from typing import Any

class CoreSolver:
    def __init__(
        self,
        eps_max_ev: float,
        d_eps_ev: float,
        n_theta: int,
        safety: float,
        parallel: bool,
        number_density: float,
        kinds: list[str],
        gas_names: list[str],
        process_names: list[str],
        fractions: list[float],
        thresholds_ev: list[float],
        mass_ratios: list[float],
        sigma_rows: list[list[float]],
    ) -> None: ...
    def mesh_data(self) -> dict[str, Any]: ...
    def initial_maxwell(self, temperature_ev: float) -> list[float]: ...
    def auto_dt(self, acceleration: float) -> float: ...
    def advection_apply(self, state: list[float], xi: float, sign: int) -> list[float]: ...
    def collision_apply(self, state: list[float]) -> list[float]: ...
    def fixed_steps(
        self,
        state: list[float],
        acceleration: float,
        dt: float,
        xi: float,
        sign: int,
        steps: int,
    ) -> list[float]: ...
    def solve_dc(self, *args: Any) -> dict[str, Any]: ...
    def solve_rf(self, *args: Any) -> dict[str, Any]: ...
