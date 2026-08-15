use std::f64::consts::PI;

use rayon::prelude::*;
use thiserror::Error;

use crate::{
    AdvectionOperator, CollisionOperator, E_CHARGE, M_E, ProcessSpec, TOWNSEND, VelocityMesh,
    output::{SwarmScalars, cheap_scalars, compute_swarm, reduced_ionization_frequency},
};

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("state became negative at step {step} with xi={xi}")]
    NegativeState { step: usize, xi: f64 },
    #[error("state normalization failed at step {step}")]
    Normalization { step: usize },
}

#[derive(Clone, Debug)]
pub struct DcOptions {
    pub en_td: f64,
    pub scheme: String,
    pub xi: Option<f64>,
    pub tol: f64,
    pub max_steps: usize,
    pub check_every: usize,
    pub dt: Option<f64>,
    pub initial_state: Option<Vec<f64>>,
    pub initial_temperature_ev: f64,
}

#[derive(Clone, Debug)]
pub struct DcResult {
    pub state: Vec<f64>,
    pub swarm: SwarmScalars,
    pub xi_used: f64,
    pub converged: bool,
    pub n_steps: usize,
    pub dt: f64,
    pub acceleration: f64,
}

#[derive(Clone, Debug)]
pub struct RfOptions {
    pub en_rms_td: f64,
    pub frequency_hz: f64,
    pub scheme: String,
    pub xi: Option<f64>,
    pub cycles_max: usize,
    pub tol: f64,
    pub steps_per_cycle: Option<usize>,
    pub n_store: usize,
    pub dt: Option<f64>,
    pub initial_state: Option<Vec<f64>>,
    pub initial_temperature_ev: f64,
}

#[derive(Clone, Debug)]
pub struct RfResult {
    pub state: Vec<f64>,
    pub swarm_at_max_field: SwarmScalars,
    pub xi_used: f64,
    pub converged: bool,
    pub n_cycles: usize,
    pub steps_per_cycle: usize,
    pub dt: f64,
    pub time: Vec<f64>,
    pub field: Vec<f64>,
    pub mean_energy: Vec<f64>,
    pub drift_velocity: Vec<f64>,
    pub reduced_ionization_frequency: Vec<f64>,
    pub phase_delay_energy: f64,
    pub phase_delay_drift: f64,
    pub mean_energy_rms: f64,
    pub drift_velocity_rms: f64,
    pub ionization_rms_over_n: f64,
}

#[derive(Clone, Debug)]
pub struct CoreSolver {
    pub mesh: VelocityMesh,
    pub number_density: f64,
    pub safety: f64,
    pub parallel: bool,
    pub processes: Vec<ProcessSpec>,
    pub collision: CollisionOperator,
}

struct MarchResult {
    state: Vec<f64>,
    converged: bool,
    steps: usize,
    negative: bool,
}

struct Workspace {
    advection: Vec<f64>,
    collision: Vec<f64>,
    next: Vec<f64>,
    energy_sum: Vec<f64>,
    reinject: Vec<f64>,
    edge_flux: Vec<f64>,
}

impl Workspace {
    fn new(mesh: &VelocityMesh, edge_count: usize) -> Self {
        Self {
            advection: vec![0.0; mesh.n_cells],
            collision: vec![0.0; mesh.n_cells],
            next: vec![0.0; mesh.n_cells],
            energy_sum: vec![0.0; mesh.n_eps],
            reinject: vec![0.0; mesh.n_eps],
            edge_flux: vec![0.0; edge_count],
        }
    }
}

impl CoreSolver {
    pub fn new(
        eps_max_ev: f64,
        d_eps_ev: f64,
        n_theta: usize,
        safety: f64,
        parallel: bool,
        number_density: f64,
        processes: Vec<ProcessSpec>,
    ) -> Result<Self, SolverError> {
        if !number_density.is_finite() || number_density <= 0.0 {
            return Err(SolverError::InvalidInput(
                "number density must be finite and positive".into(),
            ));
        }
        if !safety.is_finite() || safety <= 0.0 || safety >= 1.0 {
            return Err(SolverError::InvalidInput("safety must be in (0, 1)".into()));
        }
        let mesh =
            VelocityMesh::new(eps_max_ev, d_eps_ev, n_theta).map_err(SolverError::InvalidInput)?;
        let collision = CollisionOperator::new(&mesh, number_density, processes.clone())
            .map_err(SolverError::InvalidInput)?;
        Ok(Self {
            mesh,
            number_density,
            safety,
            parallel,
            processes,
            collision,
        })
    }

    pub fn initial_maxwell(&self, temperature_ev: f64) -> Result<Vec<f64>, SolverError> {
        if !temperature_ev.is_finite() || temperature_ev <= 0.0 {
            return Err(SolverError::InvalidInput(
                "initial temperature must be finite and positive".into(),
            ));
        }
        let mut state = vec![0.0; self.mesh.n_cells];
        for i in 0..self.mesh.n_eps {
            let energy_part =
                self.mesh.eps_c[i].sqrt() * (-self.mesh.eps_c[i] / temperature_ev).exp();
            for j in 0..self.mesh.n_theta {
                state[self.mesh.idx(i, j)] = energy_part * self.mesh.w_theta[j];
            }
        }
        normalize(&mut state, 0)?;
        Ok(state)
    }

    fn resolve_initial(
        &self,
        state: Option<Vec<f64>>,
        temperature_ev: f64,
    ) -> Result<Vec<f64>, SolverError> {
        if let Some(mut state) = state {
            if state.len() != self.mesh.n_cells {
                return Err(SolverError::InvalidInput(format!(
                    "initial state has length {}, expected {}",
                    state.len(),
                    self.mesh.n_cells
                )));
            }
            normalize(&mut state, 0)?;
            Ok(state)
        } else {
            self.initial_maxwell(temperature_ev)
        }
    }

    pub fn auto_dt(&self, acceleration: f64) -> Result<f64, SolverError> {
        let mut dt_adv = f64::INFINITY;
        if acceleration > 0.0 {
            for i in 0..self.mesh.n_eps {
                for j in 0..self.mesh.n_theta {
                    let k = self.mesh.idx(i, j);
                    let mut energy_out = if self.mesh.theta_c[j] < PI / 2.0 {
                        self.mesh.s_plus_eps[k]
                    } else {
                        self.mesh.s_minus_eps[k]
                    };
                    if i + 1 == self.mesh.n_eps && self.mesh.theta_c[j] < PI / 2.0 {
                        energy_out = 0.0;
                    }
                    let area = energy_out + self.mesh.s_minus_theta[k];
                    if area > 0.0 {
                        dt_adv = dt_adv.min(self.mesh.volume[k] / (acceleration * area));
                    }
                }
            }
        }
        let nu_max = self.collision.nu_total.iter().copied().fold(0.0, f64::max);
        let dt_collision = if nu_max > 0.0 {
            1.0 / nu_max
        } else {
            f64::INFINITY
        };
        let dt = self.safety * dt_adv.min(dt_collision);
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SolverError::InvalidInput(
                "cannot determine a finite positive time step".into(),
            ));
        }
        Ok(dt)
    }

    pub fn advection_apply(
        &self,
        state: &[f64],
        xi: f64,
        sign: i8,
    ) -> Result<Vec<f64>, SolverError> {
        validate_state(state, self.mesh.n_cells)?;
        let operator =
            AdvectionOperator::new(&self.mesh, xi, sign).map_err(SolverError::InvalidInput)?;
        let mut output = vec![0.0; self.mesh.n_cells];
        let mut edge_flux = vec![0.0; operator.edge_count()];
        operator.apply(state, &mut output, &mut edge_flux, self.parallel);
        Ok(output)
    }

    pub fn collision_apply(&self, state: &[f64]) -> Result<Vec<f64>, SolverError> {
        validate_state(state, self.mesh.n_cells)?;
        let mut output = vec![0.0; self.mesh.n_cells];
        let mut energy_sum = vec![0.0; self.mesh.n_eps];
        let mut reinject = vec![0.0; self.mesh.n_eps];
        self.collision.apply(
            state,
            &mut output,
            &mut energy_sum,
            &mut reinject,
            self.parallel,
        );
        Ok(output)
    }

    pub fn fixed_steps(
        &self,
        mut state: Vec<f64>,
        acceleration: f64,
        dt: f64,
        xi: f64,
        sign: i8,
        steps: usize,
    ) -> Result<Vec<f64>, SolverError> {
        validate_state(&state, self.mesh.n_cells)?;
        normalize(&mut state, 0)?;
        let operator =
            AdvectionOperator::new(&self.mesh, xi, sign).map_err(SolverError::InvalidInput)?;
        let mut workspace = Workspace::new(&self.mesh, operator.edge_count());
        for step in 1..=steps {
            if !self.step(
                &mut state,
                &operator,
                acceleration,
                dt,
                &mut workspace,
                step,
            )? {
                return Err(SolverError::NegativeState { step, xi });
            }
        }
        Ok(state)
    }

    pub fn solve_dc(&self, options: DcOptions) -> Result<DcResult, SolverError> {
        validate_iterations(options.max_steps, options.check_every)?;
        let initial =
            self.resolve_initial(options.initial_state, options.initial_temperature_ev)?;
        let electric_field = options.en_td * TOWNSEND * self.number_density;
        let acceleration = E_CHARGE * electric_field.abs() / M_E;
        let dt = options.dt.unwrap_or(self.auto_dt(acceleration)?);
        validate_dt(dt)?;

        let fixed_xi = if options.scheme == "upwind" {
            Some(0.0)
        } else if let Some(xi) = options.xi {
            Some(xi)
        } else if options.scheme == "blending" {
            None
        } else {
            return Err(SolverError::InvalidInput(format!(
                "unknown scheme: {}",
                options.scheme
            )));
        };

        let mut xi = fixed_xi.unwrap_or(1.0);
        loop {
            let march = self.march_dc(
                &initial,
                acceleration,
                dt,
                xi,
                options.tol,
                options.max_steps,
                options.check_every,
            )?;
            if !march.negative || fixed_xi.is_some() || xi <= 0.0 {
                if march.negative {
                    return Err(SolverError::NegativeState {
                        step: march.steps,
                        xi,
                    });
                }
                let swarm = compute_swarm(&march.state, &self.mesh, &self.processes);
                return Ok(DcResult {
                    state: march.state,
                    swarm,
                    xi_used: xi,
                    converged: march.converged,
                    n_steps: march.steps,
                    dt,
                    acceleration,
                });
            }
            xi = (xi - 0.02).max(0.0);
            if xi.abs() < 1.0e-12 {
                xi = 0.0;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn march_dc(
        &self,
        initial: &[f64],
        acceleration: f64,
        dt: f64,
        xi: f64,
        tol: f64,
        max_steps: usize,
        check_every: usize,
    ) -> Result<MarchResult, SolverError> {
        let operator =
            AdvectionOperator::new(&self.mesh, xi, 1).map_err(SolverError::InvalidInput)?;
        let mut state = initial.to_vec();
        normalize(&mut state, 0)?;
        let mut workspace = Workspace::new(&self.mesh, operator.edge_count());
        let mut previous_scalars: Option<(f64, f64)> = None;
        let mut previous_state: Option<Vec<f64>> = None;
        for step in 1..=max_steps {
            if !self.step(
                &mut state,
                &operator,
                acceleration,
                dt,
                &mut workspace,
                step,
            )? {
                return Ok(MarchResult {
                    state,
                    converged: false,
                    steps: step,
                    negative: true,
                });
            }
            if step % check_every == 0 {
                let scalars = cheap_scalars(&state, &self.mesh);
                if let (Some((previous_energy, previous_drift)), Some(previous)) =
                    (previous_scalars, previous_state.as_ref())
                {
                    let energy_delta =
                        (scalars.0 - previous_energy).abs() / scalars.0.abs().max(1.0e-300);
                    let drift_delta =
                        (scalars.1 - previous_drift).abs() / scalars.1.abs().max(1.0e-300);
                    let state_delta = state
                        .iter()
                        .zip(previous)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>();
                    if energy_delta < tol && drift_delta < tol && state_delta < tol {
                        return Ok(MarchResult {
                            state,
                            converged: true,
                            steps: step,
                            negative: false,
                        });
                    }
                }
                previous_scalars = Some(scalars);
                previous_state = Some(state.clone());
            }
        }
        Ok(MarchResult {
            state,
            converged: false,
            steps: max_steps,
            negative: false,
        })
    }

    fn step(
        &self,
        state: &mut Vec<f64>,
        advection: &AdvectionOperator,
        acceleration: f64,
        dt: f64,
        workspace: &mut Workspace,
        step: usize,
    ) -> Result<bool, SolverError> {
        advection.apply(
            state,
            &mut workspace.advection,
            &mut workspace.edge_flux,
            self.parallel,
        );
        self.collision.apply(
            state,
            &mut workspace.collision,
            &mut workspace.energy_sum,
            &mut workspace.reinject,
            self.parallel,
        );
        let mut max_value = f64::NEG_INFINITY;
        let mut min_value = f64::INFINITY;
        let mut total = 0.0;
        if self.parallel {
            (min_value, max_value, total) = workspace
                .next
                .par_iter_mut()
                .enumerate()
                .map(|(k, next)| {
                    let value = state[k]
                        + dt * (acceleration * workspace.advection[k] + workspace.collision[k]);
                    *next = value;
                    (value, value, value)
                })
                .reduce(
                    || (f64::INFINITY, f64::NEG_INFINITY, 0.0),
                    |a, b| (a.0.min(b.0), a.1.max(b.1), a.2 + b.2),
                );
        } else {
            for (k, state_value) in state.iter().copied().enumerate() {
                let value = state_value
                    + dt * (acceleration * workspace.advection[k] + workspace.collision[k]);
                workspace.next[k] = value;
                max_value = max_value.max(value);
                min_value = min_value.min(value);
                total += value;
            }
        }
        if min_value < -1.0e-14 * max_value {
            return Ok(false);
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(SolverError::Normalization { step });
        }
        let inv_total = 1.0 / total;
        if self.parallel {
            workspace
                .next
                .par_iter_mut()
                .for_each(|value| *value *= inv_total);
        } else {
            for value in &mut workspace.next {
                *value *= inv_total;
            }
        }
        std::mem::swap(state, &mut workspace.next);
        Ok(true)
    }

    pub fn solve_rf(&self, options: RfOptions) -> Result<RfResult, SolverError> {
        if !options.frequency_hz.is_finite() || options.frequency_hz <= 0.0 {
            return Err(SolverError::InvalidInput(
                "frequency_Hz must be finite and positive".into(),
            ));
        }
        if options.cycles_max == 0 || options.n_store == 0 {
            return Err(SolverError::InvalidInput(
                "cycles_max and n_store must be positive".into(),
            ));
        }
        let initial =
            self.resolve_initial(options.initial_state, options.initial_temperature_ev)?;
        let field_rms = options.en_rms_td * TOWNSEND * self.number_density;
        let field_peak = 2.0_f64.sqrt() * field_rms;
        let acceleration_peak = E_CHARGE * field_peak / M_E;
        let period = 1.0 / options.frequency_hz;
        let (steps_per_cycle, dt) = if let Some(dt) = options.dt {
            validate_dt(dt)?;
            let steps = options
                .steps_per_cycle
                .unwrap_or_else(|| (period / dt).ceil() as usize);
            (steps, dt)
        } else {
            let stable = self.auto_dt(acceleration_peak)?;
            let steps = options
                .steps_per_cycle
                .unwrap_or_else(|| (period / stable).ceil() as usize)
                .max(4);
            (steps, period / steps as f64)
        };
        if steps_per_cycle == 0 {
            return Err(SolverError::InvalidInput(
                "steps_per_cycle must be positive".into(),
            ));
        }
        let fixed_xi = if options.scheme == "upwind" {
            Some(0.0)
        } else if let Some(xi) = options.xi {
            Some(xi)
        } else if options.scheme == "blending" {
            None
        } else {
            return Err(SolverError::InvalidInput(format!(
                "unknown scheme: {}",
                options.scheme
            )));
        };

        let mut xi = fixed_xi.unwrap_or(1.0);
        loop {
            match self.run_rf_cycles(
                &initial,
                acceleration_peak,
                field_peak,
                options.frequency_hz,
                dt,
                steps_per_cycle,
                xi,
                options.tol,
                options.cycles_max,
                options.n_store,
            )? {
                RfMarch::Complete(result) => return Ok(*result),
                RfMarch::Negative { step } if fixed_xi.is_some() || xi <= 0.0 => {
                    return Err(SolverError::NegativeState { step, xi });
                }
                RfMarch::Negative { .. } => {
                    xi = (xi - 0.02).max(0.0);
                    if xi.abs() < 1.0e-12 {
                        xi = 0.0;
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rf_cycles(
        &self,
        initial: &[f64],
        acceleration_peak: f64,
        field_peak: f64,
        frequency_hz: f64,
        dt: f64,
        steps_per_cycle: usize,
        xi: f64,
        tol: f64,
        cycles_max: usize,
        n_store: usize,
    ) -> Result<RfMarch, SolverError> {
        let plus = AdvectionOperator::new(&self.mesh, xi, 1).map_err(SolverError::InvalidInput)?;
        let minus =
            AdvectionOperator::new(&self.mesh, xi, -1).map_err(SolverError::InvalidInput)?;
        let sample_stride = (steps_per_cycle / n_store).max(1);
        let sample_steps: Vec<_> = (0..steps_per_cycle)
            .step_by(sample_stride)
            .take(n_store)
            .collect();
        let time: Vec<_> = sample_steps.iter().map(|k| *k as f64 * dt).collect();
        let field: Vec<_> = time
            .iter()
            .map(|t| field_peak * (2.0 * PI * frequency_hz * t).cos())
            .collect();
        let index_max_field = field
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(index, _)| index)
            .unwrap_or(0);

        let mut state = initial.to_vec();
        normalize(&mut state, 0)?;
        let mut workspace = Workspace::new(&self.mesh, plus.edge_count().max(minus.edge_count()));
        let mut previous_energy: Option<Vec<f64>> = None;
        let mut state_at_max_field = state.clone();

        for cycle in 1..=cycles_max {
            let mut energy_wave = Vec::with_capacity(sample_steps.len());
            let mut drift_wave = Vec::with_capacity(sample_steps.len());
            let mut ionization_wave = Vec::with_capacity(sample_steps.len());
            let mut sample_index = 0;
            for k in 0..steps_per_cycle {
                let time_local = k as f64 * dt;
                let physical_acceleration =
                    -acceleration_peak * (2.0 * PI * frequency_hz * time_local).cos();
                let operator = if physical_acceleration >= 0.0 {
                    &plus
                } else {
                    &minus
                };
                let absolute_acceleration = physical_acceleration.abs();
                let total_step = (cycle - 1) * steps_per_cycle + k + 1;
                if !self.step(
                    &mut state,
                    operator,
                    absolute_acceleration,
                    dt,
                    &mut workspace,
                    total_step,
                )? {
                    return Ok(RfMarch::Negative { step: total_step });
                }
                if sample_index < sample_steps.len() && k == sample_steps[sample_index] {
                    let (energy, drift) = cheap_scalars(&state, &self.mesh);
                    energy_wave.push(energy);
                    drift_wave.push(drift);
                    ionization_wave.push(reduced_ionization_frequency(
                        &state,
                        &self.mesh,
                        &self.processes,
                    ));
                    if sample_index == index_max_field {
                        state_at_max_field.clone_from(&state);
                    }
                    sample_index += 1;
                }
            }

            let converged = previous_energy.as_ref().is_some_and(|previous| {
                let denominator = energy_wave
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0, f64::max)
                    .max(1.0e-300);
                energy_wave
                    .iter()
                    .zip(previous)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max)
                    / denominator
                    < tol
            });
            previous_energy = Some(energy_wave.clone());
            if converged || cycle == cycles_max {
                let mean_energy_rms = rms(&energy_wave);
                let drift_velocity_rms = rms(&drift_wave);
                let ionization_rms_over_n = rms(&ionization_wave);
                let phase_field_1 = dft_phase(&field, 1);
                let phase_drift_1 = dft_phase(&drift_wave, 1);
                let phase_energy_2 = dft_phase(&energy_wave, 2);
                let swarm_at_max_field =
                    compute_swarm(&state_at_max_field, &self.mesh, &self.processes);
                return Ok(RfMarch::Complete(Box::new(RfResult {
                    state,
                    swarm_at_max_field,
                    xi_used: xi,
                    converged,
                    n_cycles: cycle,
                    steps_per_cycle,
                    dt,
                    time,
                    field,
                    mean_energy: energy_wave,
                    drift_velocity: drift_wave,
                    reduced_ionization_frequency: ionization_wave,
                    phase_delay_energy: wrap_angle(phase_energy_2 - 2.0 * phase_field_1),
                    phase_delay_drift: wrap_angle(phase_drift_1 - phase_field_1),
                    mean_energy_rms,
                    drift_velocity_rms,
                    ionization_rms_over_n,
                })));
            }
        }
        unreachable!()
    }
}

enum RfMarch {
    Complete(Box<RfResult>),
    Negative { step: usize },
}

fn validate_state(state: &[f64], expected: usize) -> Result<(), SolverError> {
    if state.len() != expected {
        return Err(SolverError::InvalidInput(format!(
            "state has length {}, expected {expected}",
            state.len()
        )));
    }
    if state.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::InvalidInput(
            "state must contain only finite values".into(),
        ));
    }
    Ok(())
}

fn validate_iterations(max_steps: usize, check_every: usize) -> Result<(), SolverError> {
    if max_steps == 0 || check_every == 0 {
        return Err(SolverError::InvalidInput(
            "max_steps and check_every must be positive".into(),
        ));
    }
    Ok(())
}

fn validate_dt(dt: f64) -> Result<(), SolverError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(SolverError::InvalidInput(
            "dt must be finite and positive".into(),
        ));
    }
    Ok(())
}

fn normalize(state: &mut [f64], step: usize) -> Result<(), SolverError> {
    let total: f64 = state.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(SolverError::Normalization { step });
    }
    for value in state {
        *value /= total;
    }
    Ok(())
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn dft_phase(values: &[f64], harmonic: usize) -> f64 {
    if harmonic >= values.len() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mut real = 0.0;
    let mut imaginary = 0.0;
    for (index, value) in values.iter().enumerate() {
        let angle = -2.0 * PI * harmonic as f64 * index as f64 / n;
        real += value * angle.cos();
        imaginary += value * angle.sin();
    }
    imaginary.atan2(real)
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}
