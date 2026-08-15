from __future__ import annotations

import boltzpm2 as bp


def main() -> None:
    mixture = bp.load_argon()
    solver = bp.PMSolver(
        mixture,
        eps_max_eV=25.0,
        d_eps_eV=0.2,
        n_theta=90,
    )
    result = solver.solve_dc(EN_Td=10.0)
    print(f"converged={result.converged}, steps={result.n_steps}")
    print(f"mean energy = {result.mean_energy:.6g} eV")
    print(f"drift velocity = {result.drift_velocity:.6g} m/s")
    print(f"xi = {result.xi_used:.2f}")


if __name__ == "__main__":
    main()
