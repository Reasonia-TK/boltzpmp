use std::path::PathBuf;

use ::boltzpmp_core::{
    CoreSolver, CrossSection, DcMethod, DcOptions, DcResult, Gas, Kind, LevelState, Mixture,
    ModelOptions, ProcessKind, ProcessSpec, RfOptions, RfResult, SolveMethod, SolverError,
    SwarmScalars, Table, VelocityMesh, graded_edges, lxcat, mixture,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

/// Python側の`CrossSection`を辞書にしたもの。
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct CrossSectionInput {
    kind: String,
    species: String,
    name: String,
    threshold: f64,
    mass_ratio: Option<f64>,
    weight_ratio: Option<f64>,
    lower_state: Option<(f64, f64)>,
    upper_state: Option<(f64, f64)>,
    energy: Vec<f64>,
    sigma: Vec<f64>,
    mt_energy: Option<Vec<f64>>,
    mt: Option<Vec<f64>>,
    comment: String,
}

impl CrossSectionInput {
    fn into_core(self) -> Result<CrossSection, String> {
        let state = |value: Option<(f64, f64)>| {
            value.map(|(energy_ev, weight)| LevelState { energy_ev, weight })
        };
        let label = self.name.clone();
        let table = Table::new(self.energy, self.sigma).map_err(|err| format!("{label}: {err}"))?;
        let momentum_transfer = match (self.mt_energy, self.mt) {
            (Some(energy), Some(values)) => Some(
                Table::new(energy, values)
                    .map_err(|err| format!("{label}: momentum transfer: {err}"))?,
            ),
            (None, None) => None,
            _ => {
                return Err(format!(
                    "{label}: momentum transfer needs both energies and values"
                ));
            }
        };
        let section = CrossSection {
            kind: Kind::parse(&self.kind)?,
            species: self.species,
            name: self.name,
            threshold_ev: self.threshold,
            mass_ratio: self.mass_ratio,
            weight_ratio: self.weight_ratio,
            lower_state: state(self.lower_state),
            upper_state: state(self.upper_state),
            table,
            momentum_transfer,
            comment: self.comment,
        };
        section.validate()?;
        Ok(section)
    }
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct GasInput {
    name: String,
    fraction: f64,
    mass_amu: Option<f64>,
    cross_sections: Vec<CrossSectionInput>,
}

impl GasInput {
    fn into_core(self) -> Result<Gas, String> {
        let cross_sections = self
            .cross_sections
            .into_iter()
            .map(CrossSectionInput::into_core)
            .collect::<Result<_, _>>()?;
        Ok(Gas {
            name: self.name,
            fraction: self.fraction,
            cross_sections,
            mass_amu: self.mass_amu,
        })
    }
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn to_python_error(error: SolverError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn section_to_dict<'py>(py: Python<'py>, section: &CrossSection) -> PyResult<Bound<'py, PyDict>> {
    let state = |value: Option<LevelState>| value.map(|s| (s.energy_ev, s.weight));
    let dict = PyDict::new(py);
    dict.set_item("kind", section.kind.keyword())?;
    dict.set_item("species", &section.species)?;
    dict.set_item("name", &section.name)?;
    dict.set_item("threshold", section.threshold_ev)?;
    dict.set_item("mass_ratio", section.mass_ratio)?;
    dict.set_item("weight_ratio", section.weight_ratio)?;
    dict.set_item("lower_state", state(section.lower_state))?;
    dict.set_item("upper_state", state(section.upper_state))?;
    dict.set_item("energy", section.table.energy.clone())?;
    dict.set_item("sigma", section.table.values.clone())?;
    dict.set_item(
        "mt",
        section
            .momentum_transfer
            .as_ref()
            .map(|table| table.values.clone()),
    )?;
    dict.set_item("comment", &section.comment)?;
    Ok(dict)
}

fn sections_to_list(py: Python<'_>, sections: Vec<CrossSection>) -> PyResult<Vec<Py<PyDict>>> {
    sections
        .iter()
        .map(|section| section_to_dict(py, section).map(Bound::unbind))
        .collect()
}

/// LXCat形式のテキストを読み、断面積ごとの辞書のリストを返す。
#[pyfunction]
fn parse_lxcat_text(py: Python<'_>, text: &str) -> PyResult<Vec<Py<PyDict>>> {
    let sections = lxcat::parse_str(text).map_err(|err| value_error(err.to_string()))?;
    sections_to_list(py, sections)
}

/// LXCat形式のファイルを読む（UTF-8、読めなければLatin-1）。
#[pyfunction]
fn parse_lxcat_file(py: Python<'_>, path: PathBuf) -> PyResult<Vec<Py<PyDict>>> {
    let sections = lxcat::parse_file(&path).map_err(value_error)?;
    sections_to_list(py, sections)
}

/// 断面積を検証する（`CrossSection`の生成時に呼ぶ）。
#[pyfunction]
fn validate_cross_section(section: CrossSectionInput) -> PyResult<()> {
    section.into_core().map(|_| ()).map_err(value_error)
}

/// しきい値未満を0とした区分線形補間（範囲外は下側0、上側は最後の値）。
#[pyfunction]
fn interp_sigma(
    energy: Vec<f64>,
    values: Vec<f64>,
    threshold: f64,
    eps: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let table = Table::new(energy, values).map_err(value_error)?;
    Ok(eps
        .into_iter()
        .map(|e| {
            if threshold > 0.0 && e < threshold {
                0.0
            } else {
                table.at(e)
            }
        })
        .collect())
}

#[pyfunction]
fn validate_fractions(fractions: Vec<f64>) -> PyResult<f64> {
    mixture::validate_fractions(fractions).map_err(value_error)
}

#[pyfunction]
#[pyo3(signature = (pressure_pa, temperature_k, number_density))]
fn number_density(
    pressure_pa: Option<f64>,
    temperature_k: f64,
    number_density: Option<f64>,
) -> PyResult<f64> {
    mixture::number_density(pressure_pa, temperature_k, number_density).map_err(value_error)
}

#[pyfunction]
fn mass_ratio_from_amu(mass_amu: f64) -> PyResult<f64> {
    mixture::mass_ratio_from_amu(mass_amu).map_err(value_error)
}

fn mesh_to_dict(py: Python<'_>, mesh: &VelocityMesh) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("eps_max_eV", mesh.eps_max_ev)?;
    dict.set_item("d_eps_eV", mesh.d_eps_ev)?;
    dict.set_item("n_eps", mesh.n_eps)?;
    dict.set_item("n_theta", mesh.n_theta)?;
    dict.set_item("n_cells", mesh.n_cells)?;
    dict.set_item("d_theta", mesh.d_theta)?;
    dict.set_item("eps_b", mesh.eps_b.clone())?;
    dict.set_item("eps_c", mesh.eps_c.clone())?;
    dict.set_item("d_eps", mesh.d_eps.clone())?;
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

#[pyfunction]
fn mesh_data(
    py: Python<'_>,
    eps_max_ev: f64,
    d_eps_ev: f64,
    n_theta: usize,
) -> PyResult<Py<PyDict>> {
    let mesh = VelocityMesh::new(eps_max_ev, d_eps_ev, n_theta).map_err(value_error)?;
    mesh_to_dict(py, &mesh)
}

#[pyfunction]
fn mesh_data_from_edges(
    py: Python<'_>,
    edges_ev: Vec<f64>,
    n_theta: usize,
) -> PyResult<Py<PyDict>> {
    let mesh = VelocityMesh::from_edges(&edges_ev, n_theta).map_err(value_error)?;
    mesh_to_dict(py, &mesh)
}

/// 低エネルギー側を細かくしたエネルギー境界。
#[pyfunction]
fn graded_energy_edges(
    eps_max_ev: f64,
    d_eps_min_ev: f64,
    eps_uniform_ev: f64,
) -> PyResult<Vec<f64>> {
    graded_edges(eps_max_ev, d_eps_min_ev, eps_uniform_ev).map_err(value_error)
}

#[pyclass(name = "CoreSolver")]
struct PyCoreSolver {
    inner: CoreSolver,
}

#[pymethods]
impl PyCoreSolver {
    /// 低水準の生成。セル中心の断面積を直接与える（等方散乱・冷たい気体）。
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
                return Err(value_error(format!(
                    "{label} has length {actual}, expected {count}"
                )));
            }
        }
        let mut processes = Vec::with_capacity(count);
        for index in 0..count {
            processes.push(ProcessSpec::new(
                gas_names[index].clone(),
                process_names[index].clone(),
                ProcessKind::parse(&kinds[index]).map_err(value_error)?,
                fractions[index],
                thresholds_ev[index],
                mass_ratios[index],
                sigma_rows[index].clone(),
            ));
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

    /// 混合気体から生成する。`energy_edges`を与えると非一様格子を使う。
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        gases, number_density, temperature_k, excitation_temperature_k, transition_energy_ev,
        energy_edges, eps_max_ev, d_eps_ev, n_theta, safety, parallel, superelastic, gas_heating
    ))]
    fn from_mixture(
        gases: Vec<GasInput>,
        number_density: f64,
        temperature_k: f64,
        excitation_temperature_k: Option<f64>,
        transition_energy_ev: f64,
        energy_edges: Option<Vec<f64>>,
        eps_max_ev: f64,
        d_eps_ev: f64,
        n_theta: usize,
        safety: f64,
        parallel: bool,
        superelastic: bool,
        gas_heating: bool,
    ) -> PyResult<Self> {
        let gases = gases
            .into_iter()
            .map(GasInput::into_core)
            .collect::<Result<Vec<_>, _>>()
            .map_err(value_error)?;
        let mixture = Mixture::new(
            gases,
            number_density,
            temperature_k,
            excitation_temperature_k,
            transition_energy_ev,
        )
        .map_err(value_error)?;
        let mesh = match energy_edges {
            Some(edges) => VelocityMesh::from_edges(&edges, n_theta),
            None => VelocityMesh::new(eps_max_ev, d_eps_ev, n_theta),
        }
        .map_err(value_error)?;
        let options = ModelOptions {
            superelastic,
            gas_heating,
        };
        let inner = CoreSolver::from_mixture(&mixture, mesh, safety, parallel, options)
            .map_err(to_python_error)?;
        Ok(Self { inner })
    }

    fn mesh_data(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        mesh_to_dict(py, &self.inner.mesh)
    }

    /// 組み立てた衝突過程の一覧（確認用）。
    fn processes(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.inner
            .processes
            .iter()
            .map(|process| {
                let dict = PyDict::new(py);
                dict.set_item("gas", &process.gas_name)?;
                dict.set_item("name", &process.name)?;
                dict.set_item("kind", format!("{:?}", process.kind).to_ascii_uppercase())?;
                dict.set_item("fraction", process.fraction)?;
                dict.set_item("threshold", process.threshold_ev)?;
                dict.set_item("anisotropic", process.sigma_mt.is_some())?;
                dict.set_item("gas_temperature_eV", process.gas_temperature_ev)?;
                Ok(dict.unbind())
            })
            .collect()
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
    #[pyo3(signature = (
        en_td, scheme, xi_or_nan, tol, max_steps, check_every, initial_temperature_ev,
        dt_or_nan, initial_state, method = "explicit"
    ))]
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
        method: &str,
    ) -> PyResult<Py<PyDict>> {
        let options = dc_options(
            en_td,
            scheme,
            xi_or_nan,
            tol,
            max_steps,
            check_every,
            initial_temperature_ev,
            dt_or_nan,
            initial_state,
            DcMethod::parse(method).map_err(to_python_error)?,
        );
        let result = py
            .detach(|| self.inner.solve_dc(options))
            .map_err(to_python_error)?;
        dc_result_to_dict(py, result)
    }

    /// 複数のE/NをRayonで並列に解く。結果は入力順で、最初の失敗は例外として送出する。
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        en_values, scheme, xi_or_nan, tol, max_steps, check_every, initial_temperature_ev,
        dt_or_nan, initial_state, max_workers, method = "explicit"
    ))]
    fn solve_dc_many(
        &self,
        py: Python<'_>,
        en_values: Vec<f64>,
        scheme: String,
        xi_or_nan: f64,
        tol: f64,
        max_steps: usize,
        check_every: usize,
        initial_temperature_ev: f64,
        dt_or_nan: f64,
        initial_state: Vec<f64>,
        max_workers: Option<usize>,
        method: &str,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let method = DcMethod::parse(method).map_err(to_python_error)?;
        let options: Vec<DcOptions> = en_values
            .into_iter()
            .map(|en_td| {
                dc_options(
                    en_td,
                    scheme.clone(),
                    xi_or_nan,
                    tol,
                    max_steps,
                    check_every,
                    initial_temperature_ev,
                    dt_or_nan,
                    initial_state.clone(),
                    method,
                )
            })
            .collect();
        let results = py
            .detach(|| self.inner.solve_dc_many(options, max_workers))
            .map_err(to_python_error)?;
        results
            .into_iter()
            .map(|result| dc_result_to_dict(py, result.map_err(to_python_error)?))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        en_rms_td, frequency_hz, scheme, xi_or_nan, cycles_max, tol, steps_per_cycle_or_zero,
        initial_temperature_ev, n_store, dt_or_nan, initial_state, method = "explicit"
    ))]
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
        method: &str,
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
            method: SolveMethod::parse(method).map_err(to_python_error)?,
        };
        let result = py
            .detach(|| self.inner.solve_rf(options))
            .map_err(to_python_error)?;
        rf_result_to_dict(py, result)
    }
}

#[allow(clippy::too_many_arguments)]
fn dc_options(
    en_td: f64,
    scheme: String,
    xi_or_nan: f64,
    tol: f64,
    max_steps: usize,
    check_every: usize,
    initial_temperature_ev: f64,
    dt_or_nan: f64,
    initial_state: Vec<f64>,
    method: DcMethod,
) -> DcOptions {
    DcOptions {
        en_td,
        scheme,
        xi: finite_option(xi_or_nan),
        tol,
        max_steps,
        check_every,
        dt: finite_option(dt_or_nan),
        initial_state: nonempty_option(initial_state),
        initial_temperature_ev,
        method,
    }
}

fn finite_option(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn nonempty_option(value: Vec<f64>) -> Option<Vec<f64>> {
    (!value.is_empty()).then_some(value)
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
    dict.set_item(
        "reduced_attachment_frequency",
        swarm.reduced_attachment_frequency,
    )?;
    dict.set_item("rate_coefficients", swarm.rate_coefficients)?;
    dict.set_item("fractions", swarm.fractions)?;
    dict.set_item("eepf_tail_ratio", swarm.eepf_tail_ratio)?;
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
    dict.set_item("inner_iterations", result.inner_iterations)?;
    dict.set_item("cycle_residuals", result.cycle_residuals)?;
    Ok(dict.unbind())
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCoreSolver>()?;
    module.add_function(wrap_pyfunction!(parse_lxcat_text, module)?)?;
    module.add_function(wrap_pyfunction!(parse_lxcat_file, module)?)?;
    module.add_function(wrap_pyfunction!(validate_cross_section, module)?)?;
    module.add_function(wrap_pyfunction!(interp_sigma, module)?)?;
    module.add_function(wrap_pyfunction!(validate_fractions, module)?)?;
    module.add_function(wrap_pyfunction!(number_density, module)?)?;
    module.add_function(wrap_pyfunction!(mass_ratio_from_amu, module)?)?;
    module.add_function(wrap_pyfunction!(mesh_data, module)?)?;
    module.add_function(wrap_pyfunction!(mesh_data_from_edges, module)?)?;
    module.add_function(wrap_pyfunction!(graded_energy_edges, module)?)?;
    Ok(())
}
