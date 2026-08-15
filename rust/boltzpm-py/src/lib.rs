use ::boltzpm_core::{
    CoreSolver, DcOptions, DcResult, ProcessKind, ProcessSpec, RfOptions, RfResult, SolverError,
    SwarmScalars,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[pyclass(name = "CoreSolver")]
struct PyCoreSolver {
    inner: CoreSolver,
}

#[pymethods]
impl PyCoreSolver {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        eps_max_ev: f64,
        d_eps_ev: f64,
        n_theta: usize,
        safety: f64,
        parallel: bool,
        number_density: f64,
        kinds: Vec<String>,
        gas_names: Vec<String>,
        process_names: Vec<String>,
        fractions: Vec<f64>,
        thresholds_ev: Vec<f64>,
        mass_ratios: Vec<f64>,
        sigma_rows: Vec<Vec<f64>>,
    ) -> PyResult<Self> {
        let count = kinds.len();
        for (label, actual) in [
            ("gas_names", gas_names.len()),
            ("process_names", process_names.len()),
            ("fractions", fractions.len()),
            ("thresholds_ev", thresholds_ev.len()),
            ("mass_ratios", mass_ratios.len()),
            ("sigma_rows", sigma_rows.len()),
        ] {
            if actual != count {
                return Err(PyValueError::new_err(format!(
                    "{label} has length {actual}, expected {count}"
                )));
            }
        }
        let mut processes = Vec::with_capacity(count);
        for index in 0..count {
            processes.push(ProcessSpec {
                gas_name: gas_names[index].clone(),
                name: process_names[index].clone(),
                kind: ProcessKind::parse(&kinds[index]).map_err(PyValueError::new_err)?,
                fraction: fractions[index],
                threshold_ev: thresholds_ev[index],
                mass_ratio: mass_ratios[index],
                sigma: sigma_rows[index].clone(),
            });
        }
        let inner = CoreSolver::new(
            eps_max_ev,
            d_eps_ev,
            n_theta,
            safety,
            parallel,
            number_density,
            processes,
        )
        .map_err(to_python_error)?;
        Ok(Self { inner })
    }

    fn mesh_data(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let mesh = &self.inner.mesh;
        let dict = PyDict::new(py);
        dict.set_item("eps_max_eV", mesh.eps_max_ev)?;
        dict.set_item("d_eps_eV", mesh.d_eps_ev)?;
        dict.set_item("n_eps", mesh.n_eps)?;
        dict.set_item("n_theta", mesh.n_theta)?;
        dict.set_item("n_cells", mesh.n_cells)?;
        dict.set_item("d_theta", mesh.d_theta)?;
        dict.set_item("eps_b", mesh.eps_b.clone())?;
        dict.set_item("eps_c", mesh.eps_c.clone())?;
        dict.set_item("v_b", mesh.v_b.clone())?;
        dict.set_item("v_c", mesh.v_c.clone())?;
        dict.set_item("theta_b", mesh.theta_b.clone())?;
        dict.set_item("theta_c", mesh.theta_c.clone())?;
        dict.set_item("V", mesh.volume.clone())?;
        dict.set_item("S_plus_eps", mesh.s_plus_eps.clone())?;
        dict.set_item("S_minus_eps", mesh.s_minus_eps.clone())?;
        dict.set_item("S_plus_theta", mesh.s_plus_theta.clone())?;
        dict.set_item("S_minus_theta", mesh.s_minus_theta.clone())?;
        dict.set_item("w_theta", mesh.w_theta.clone())?;
        Ok(dict.unbind())
    }

    fn initial_maxwell(&self, temperature_ev: f64) -> PyResult<Vec<f64>> {
        self.inner
            .initial_maxwell(temperature_ev)
            .map_err(to_python_error)
    }

    fn auto_dt(&self, acceleration: f64) -> PyResult<f64> {
        self.inner.auto_dt(acceleration).map_err(to_python_error)
    }

    fn advection_apply(&self, state: Vec<f64>, xi: f64, sign: i8) -> PyResult<Vec<f64>> {
        self.inner
            .advection_apply(&state, xi, sign)
            .map_err(to_python_error)
    }

    fn collision_apply(&self, state: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.collision_apply(&state).map_err(to_python_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn fixed_steps(
        &self,
        py: Python<'_>,
        state: Vec<f64>,
        acceleration: f64,
        dt: f64,
        xi: f64,
        sign: i8,
        steps: usize,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            self.inner
                .fixed_steps(state, acceleration, dt, xi, sign, steps)
        })
        .map_err(to_python_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_dc(
        &self,
        py: Python<'_>,
        en_td: f64,
        scheme: String,
        xi_or_nan: f64,
        tol: f64,
        max_steps: usize,
        check_every: usize,
        initial_temperature_ev: f64,
        dt_or_nan: f64,
        initial_state: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let options = DcOptions {
            en_td,
            scheme,
            xi: finite_option(xi_or_nan),
            tol,
            max_steps,
            check_every,
            dt: finite_option(dt_or_nan),
            initial_state: nonempty_option(initial_state),
            initial_temperature_ev,
        };
        let result = py
            .detach(|| self.inner.solve_dc(options))
            .map_err(to_python_error)?;
        dc_result_to_dict(py, result)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_rf(
        &self,
        py: Python<'_>,
        en_rms_td: f64,
        frequency_hz: f64,
        scheme: String,
        xi_or_nan: f64,
        cycles_max: usize,
        tol: f64,
        steps_per_cycle_or_zero: usize,
        initial_temperature_ev: f64,
        n_store: usize,
        dt_or_nan: f64,
        initial_state: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let options = RfOptions {
            en_rms_td,
            frequency_hz,
            scheme,
            xi: finite_option(xi_or_nan),
            cycles_max,
            tol,
            steps_per_cycle: (steps_per_cycle_or_zero > 0).then_some(steps_per_cycle_or_zero),
            n_store,
            dt: finite_option(dt_or_nan),
            initial_state: nonempty_option(initial_state),
            initial_temperature_ev,
        };
        let result = py
            .detach(|| self.inner.solve_rf(options))
            .map_err(to_python_error)?;
        rf_result_to_dict(py, result)
    }
}

fn finite_option(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn nonempty_option(value: Vec<f64>) -> Option<Vec<f64>> {
    (!value.is_empty()).then_some(value)
}

fn to_python_error(error: SolverError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn add_swarm(dict: &Bound<'_, PyDict>, swarm: SwarmScalars) -> PyResult<()> {
    dict.set_item("eedf", swarm.eedf)?;
    dict.set_item("eepf", swarm.eepf)?;
    dict.set_item("mean_energy", swarm.mean_energy)?;
    dict.set_item("drift_velocity", swarm.drift_velocity)?;
    dict.set_item(
        "reduced_ionization_frequency",
        swarm.reduced_ionization_frequency,
    )?;
    dict.set_item("rate_coefficients", swarm.rate_coefficients)?;
    Ok(())
}

fn dc_result_to_dict(py: Python<'_>, result: DcResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    add_swarm(&dict, result.swarm)?;
    dict.set_item("n", result.state)?;
    dict.set_item("xi_used", result.xi_used)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("n_steps", result.n_steps)?;
    dict.set_item("dt", result.dt)?;
    dict.set_item("acceleration", result.acceleration)?;
    Ok(dict.unbind())
}

fn rf_result_to_dict(py: Python<'_>, result: RfResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    add_swarm(&dict, result.swarm_at_max_field)?;
    dict.set_item("n", result.state)?;
    dict.set_item("xi_used", result.xi_used)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("n_cycles", result.n_cycles)?;
    dict.set_item("steps_per_cycle", result.steps_per_cycle)?;
    dict.set_item("dt", result.dt)?;
    dict.set_item("time", result.time)?;
    dict.set_item("field", result.field)?;
    dict.set_item("mean_energy_t", result.mean_energy)?;
    dict.set_item("drift_velocity_t", result.drift_velocity)?;
    dict.set_item(
        "reduced_ionization_frequency_t",
        result.reduced_ionization_frequency,
    )?;
    dict.set_item("phase_delay_energy", result.phase_delay_energy)?;
    dict.set_item("phase_delay_W", result.phase_delay_drift)?;
    dict.set_item("mean_energy_rms", result.mean_energy_rms)?;
    dict.set_item("drift_velocity_rms", result.drift_velocity_rms)?;
    dict.set_item("nu_ion_rms_over_N", result.ionization_rms_over_n)?;
    Ok(dict.unbind())
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCoreSolver>()?;
    Ok(())
}
