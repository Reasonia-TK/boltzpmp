use std::f64::consts::PI;

use rayon::prelude::*;
use thiserror::Error;

use crate::{
    AdvectionOperator, AdvectionScheme, CollisionOperator, E_CHARGE, M_E, ProcessSpec, TOWNSEND,
    VelocityMesh,
    implicit::{
        self, DiffusionAcceleration, ImplicitProblem, OuterAnderson, StepBuffers, StepCoefficients,
        TimeStepper,
    },
    mixture::Mixture,
    output::{SwarmScalars, cheap_scalars, compute_swarm, reduced_ionization_frequency},
    processes::{ModelOptions, build_processes},
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

/// DC定常解とRFの周期定常解の求め方。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SolveMethod {
    /// 陽的な時間発展（プロパゲータ法）で定常になるまで進める。
    #[default]
    Explicit,
    /// 輸送スイープのソース反復とAnderson加速で解く（`implicit`モジュール）。DCは定常方程式を直接、
    /// RFは各段を陰的（BDF2）に解き、周期写像の不動点をAnderson加速で求める。
    Implicit,
}

/// 0.3 までの名前。
pub type DcMethod = SolveMethod;

impl SolveMethod {
    pub fn parse(value: &str) -> Result<Self, SolverError> {
        match value {
            "explicit" => Ok(Self::Explicit),
            "implicit" => Ok(Self::Implicit),
            _ => Err(SolverError::InvalidInput(format!(
                "unknown method: {value} (use 'explicit' or 'implicit')"
            ))),
        }
    }
}

/// 陰解法で使うAnderson加速の履歴の長さ。
const ANDERSON_DEPTH: usize = 10;
/// RFの陰解法で1周期の段数を指定しないときの最小の段数（保存点数の倍数に切り上げる）。
const RF_IMPLICIT_MIN_STEPS: usize = 256;
/// RFの陰解法の1段の反復の許容値（残差 ‖g − x‖₁）と反復回数の上限。
const RF_STEP_TOL: f64 = 1.0e-10;
const RF_STEP_MAX_ITERATIONS: usize = 1000;
/// RFの周期写像のAnderson加速の履歴の長さ。
const RF_CYCLE_DEPTH: usize = 10;
/// RFの陰解法の初期状態に使う、実効電場のDC解の許容値と反復回数の上限。
const RF_START_TOL: f64 = 1.0e-8;
const RF_START_MAX_ITERATIONS: usize = 2000;
/// 遅いモードの補正の直後に、周期の残差が補正前のこの倍を超えたら、補正を使うのをやめる。
const RF_NDA_GIVE_UP: f64 = 3.0;
/// 周期の残差の減りがこの比より鈍ったら（速いモードが消えたら）遅いモードを補正する。
const RF_STAGNATION: f64 = 0.7;
/// 遅いモードの補正の間に置く周期の数（補正で起きた過渡がおさまるのを待つ）。
const RF_CORRECTION_GAP: usize = 3;

#[derive(Clone, Debug)]
pub struct DcOptions {
    pub en_td: f64,
    pub scheme: String,
    pub xi: Option<f64>,
    /// 陽解法では判定間隔ごとの相対変化、陰解法では1反復の残差 ‖g(n) − n‖₁ の許容値。
    pub tol: f64,
    /// 陽解法ではステップ数、陰解法では反復回数の上限。
    pub max_steps: usize,
    pub check_every: usize,
    pub dt: Option<f64>,
    pub initial_state: Option<Vec<f64>>,
    pub initial_temperature_ev: f64,
    pub method: SolveMethod,
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
    /// 陰解法で二項近似の合成加速を使えなかったときの理由（使えたとき、陽解法では `None`）。
    pub two_term_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RfOptions {
    pub en_rms_td: f64,
    pub frequency_hz: f64,
    pub scheme: String,
    pub xi: Option<f64>,
    pub cycles_max: usize,
    /// 陽解法では周期ごとの平均エネルギー波形の最大相対変化、陰解法ではそれに加えて
    /// 1周期の写像の残差 ‖Φ(n) − n‖₁ の許容値。
    pub tol: f64,
    pub steps_per_cycle: Option<usize>,
    pub n_store: usize,
    pub dt: Option<f64>,
    /// 陰解法で `None` のときは、実効値の電場でのDC解から始める。
    pub initial_state: Option<Vec<f64>>,
    pub initial_temperature_ev: f64,
    pub method: SolveMethod,
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
    /// 最後の1周期で時間平均した分布の値（EEDF、平均エネルギー、速度係数など）。
    pub swarm_average: SwarmScalars,
    /// 保存点ごとのEEDF（eV⁻¹、`time` と同じ順）。
    pub eedf_t: Vec<Vec<f64>>,
    /// 陰解法で各段を解いた反復の合計（陽解法では0）。
    pub inner_iterations: usize,
    /// 陰解法の周期ごとの残差 ‖Φ(n) − n‖₁（陽解法では空）。
    pub cycle_residuals: Vec<f64>,
    /// 陰解法で二項近似の前処理を使えなかったときの理由（使えたとき、陽解法では `None`）。
    pub two_term_error: Option<String>,
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
        let mesh =
            VelocityMesh::new(eps_max_ev, d_eps_ev, n_theta).map_err(SolverError::InvalidInput)?;
        Self::with_mesh(mesh, number_density, safety, parallel, processes)
    }

    /// 組み立て済みのメッシュ上の衝突過程から作る。
    pub fn with_mesh(
        mesh: VelocityMesh,
        number_density: f64,
        safety: f64,
        parallel: bool,
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

    /// 混合気体から作る。断面積のセル中心値、超弾性衝突、気体温度の効果、異方散乱をここで組み立てる。
    pub fn from_mixture(
        mixture: &Mixture,
        mesh: VelocityMesh,
        safety: f64,
        parallel: bool,
        options: ModelOptions,
    ) -> Result<Self, SolverError> {
        let processes =
            build_processes(mixture, &mesh, options).map_err(SolverError::InvalidInput)?;
        Self::with_mesh(mesh, mixture.number_density, safety, parallel, processes)
    }

    pub fn initial_maxwell(&self, temperature_ev: f64) -> Result<Vec<f64>, SolverError> {
        if !temperature_ev.is_finite() || temperature_ev <= 0.0 {
            return Err(SolverError::InvalidInput(
                "initial temperature must be finite and positive".into(),
            ));
        }
        let mut state = vec![0.0; self.mesh.n_cells];
        for i in 0..self.mesh.n_eps {
            // 一様格子では幅が共通なので掛けない（0.1.3と同じ値にする）
            let width = if self.mesh.d_eps_ev.is_some() {
                1.0
            } else {
                self.mesh.d_eps[i]
            };
            let energy_part =
                width * self.mesh.eps_c[i].sqrt() * (-self.mesh.eps_c[i] / temperature_ev).exp();
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
        if options.method == SolveMethod::Implicit {
            return self.solve_dc_implicit(options);
        }
        let initial =
            self.resolve_initial(options.initial_state, options.initial_temperature_ev)?;
        let electric_field = options.en_td * TOWNSEND * self.number_density;
        let acceleration = E_CHARGE * electric_field.abs() / M_E;
        let dt = options.dt.unwrap_or(self.auto_dt(acceleration)?);
        validate_dt(dt)?;

        let (mut advection, searching) = choose_scheme(&options.scheme, options.xi)?;
        loop {
            let march = self.march_dc(
                &initial,
                acceleration,
                dt,
                advection,
                options.tol,
                options.max_steps,
                options.check_every,
            )?;
            let xi = xi_of(advection);
            if !march.negative || !searching || xi <= 0.0 {
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
                    two_term_error: None,
                });
            }
            advection = AdvectionScheme::Linear(lower_xi(xi));
        }
    }

    fn solve_dc_implicit(&self, options: DcOptions) -> Result<DcResult, SolverError> {
        // 初期状態を与えなければ、二項近似の定常解から始める（Maxwell 分布はその λ を決めるのに使う）
        let two_term_start = options.initial_state.is_none();
        let initial =
            self.resolve_initial(options.initial_state, options.initial_temperature_ev)?;
        let electric_field = options.en_td * TOWNSEND * self.number_density;
        let acceleration = E_CHARGE * electric_field.abs() / M_E;
        let (advection, searching) = choose_scheme(&options.scheme, options.xi)?;
        if searching {
            return Err(SolverError::InvalidInput(
                "scheme 'blending' searches xi by restarting the time march; with the implicit \
                 method use 'limiter', 'upwind' or a fixed xi"
                    .into(),
            ));
        }
        let problem = ImplicitProblem::new(
            &self.mesh,
            &self.collision,
            acceleration,
            advection,
            self.parallel,
        )
        .map_err(SolverError::InvalidInput)?;
        let outcome = implicit::solve(
            &problem,
            &initial,
            two_term_start,
            options.tol,
            options.max_steps,
            ANDERSON_DEPTH,
        )
        .map_err(SolverError::InvalidInput)?;
        let swarm = compute_swarm(&outcome.state, &self.mesh, &self.processes);
        Ok(DcResult {
            state: outcome.state,
            swarm,
            xi_used: xi_of(advection),
            converged: outcome.converged,
            n_steps: outcome.iterations,
            dt: f64::NAN,
            acceleration,
            two_term_error: outcome.acceleration_error,
        })
    }

    /// 独立なDC計算を並列に解く。結果は入力順。`max_workers`が`None`ならRayonの既定数。
    pub fn solve_dc_many(
        &self,
        options: Vec<DcOptions>,
        max_workers: Option<usize>,
    ) -> Result<Vec<Result<DcResult, SolverError>>, SolverError> {
        run_parallel(options, max_workers, |item| self.solve_dc(item))
    }

    /// 独立なRF計算を並列に解く。結果は入力順。`max_workers`が`None`ならRayonの既定数。
    pub fn solve_rf_many(
        &self,
        options: Vec<RfOptions>,
        max_workers: Option<usize>,
    ) -> Result<Vec<Result<RfResult, SolverError>>, SolverError> {
        run_parallel(options, max_workers, |item| self.solve_rf(item))
    }

    #[allow(clippy::too_many_arguments)]
    fn march_dc(
        &self,
        initial: &[f64],
        acceleration: f64,
        dt: f64,
        advection: AdvectionScheme,
        tol: f64,
        max_steps: usize,
        check_every: usize,
    ) -> Result<MarchResult, SolverError> {
        let operator = AdvectionOperator::with_scheme(&self.mesh, advection, 1)
            .map_err(SolverError::InvalidInput)?;
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
        if options.method == SolveMethod::Implicit {
            return self.solve_rf_implicit(options);
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
        let (mut advection, searching) = choose_scheme(&options.scheme, options.xi)?;
        loop {
            let xi = xi_of(advection);
            match self.run_rf_cycles(
                &initial,
                acceleration_peak,
                field_peak,
                options.frequency_hz,
                dt,
                steps_per_cycle,
                advection,
                options.tol,
                options.cycles_max,
                options.n_store,
            )? {
                RfMarch::Complete(result) => return Ok(*result),
                RfMarch::Negative { step } if !searching || xi <= 0.0 => {
                    return Err(SolverError::NegativeState { step, xi });
                }
                RfMarch::Negative { .. } => {
                    advection = AdvectionScheme::Linear(lower_xi(xi));
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
        advection: AdvectionScheme,
        tol: f64,
        cycles_max: usize,
        n_store: usize,
    ) -> Result<RfMarch, SolverError> {
        let xi = xi_of(advection);
        let plus = AdvectionOperator::with_scheme(&self.mesh, advection, 1)
            .map_err(SolverError::InvalidInput)?;
        let minus = AdvectionOperator::with_scheme(&self.mesh, advection, -1)
            .map_err(SolverError::InvalidInput)?;
        let sampling = RfSampling::new(steps_per_cycle, n_store, dt, field_peak, frequency_hz);

        let mut state = initial.to_vec();
        normalize(&mut state, 0)?;
        let mut workspace = Workspace::new(&self.mesh, plus.edge_count().max(minus.edge_count()));
        let mut previous_energy: Option<Vec<f64>> = None;

        for cycle in 1..=cycles_max {
            let mut recorder = CycleRecorder::new(sampling.time.len(), &state, 0);
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
                // 段 k の終わり（時刻 (k + 1)Δt）の状態
                recorder.record(&state, &sampling, k + 1, &self.mesh, &self.processes);
            }

            let converged = previous_energy.as_ref().is_some_and(|previous| {
                let denominator = recorder
                    .mean_energy
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0, f64::max)
                    .max(1.0e-300);
                recorder
                    .mean_energy
                    .iter()
                    .zip(previous)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max)
                    / denominator
                    < tol
            });
            previous_energy = Some(recorder.mean_energy.clone());
            if converged || cycle == cycles_max {
                return Ok(RfMarch::Complete(Box::new(periodic_result(
                    state,
                    recorder,
                    &sampling,
                    PeriodicRun {
                        xi_used: xi,
                        converged,
                        n_cycles: cycle,
                        steps_per_cycle,
                        dt,
                        inner_iterations: 0,
                        cycle_residuals: Vec::new(),
                        two_term_error: None,
                    },
                    &self.mesh,
                    &self.processes,
                ))));
            }
        }
        unreachable!()
    }

    /// RFの周期定常解を陰解法で求める。
    ///
    /// - 各段は BDF2（1段目と、右辺が負になる段は後退 Euler）で、`TimeStepper` の反復で解く。
    /// - 1周期の写像 Φ の不動点 n = Φ(n) を、非線形拡散加速（`implicit::DiffusionAcceleration`）で求める。
    ///   各周期の後に、周期平均の分布と流束から低次の定常解を求め、等方成分を置き換える。
    /// - 周期ごとの残差 ‖Φ(n) − n‖₁ と、低次の解と周期平均の分布の差がともに `tol` 未満で収束とする。
    fn solve_rf_implicit(&self, options: RfOptions) -> Result<RfResult, SolverError> {
        if options.en_rms_td.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(SolverError::InvalidInput(
                "the implicit RF method needs a positive EN_rms_Td".into(),
            ));
        }
        let field_rms = options.en_rms_td * TOWNSEND * self.number_density;
        let field_peak = 2.0_f64.sqrt() * field_rms;
        let acceleration_peak = E_CHARGE * field_peak / M_E;
        let period = 1.0 / options.frequency_hz;
        let steps = match (options.steps_per_cycle, options.dt) {
            (Some(steps), _) => steps,
            (None, Some(dt)) => {
                validate_dt(dt)?;
                (period / dt).ceil() as usize
            }
            (None, None) => options.n_store * RF_IMPLICIT_MIN_STEPS.div_ceil(options.n_store),
        };
        if steps < 2 {
            return Err(SolverError::InvalidInput(
                "the implicit RF method needs at least 2 steps per cycle".into(),
            ));
        }
        let dt = period / steps as f64;
        let (advection, searching) = choose_scheme(&options.scheme, options.xi)?;
        if searching {
            return Err(SolverError::InvalidInput(
                "scheme 'blending' searches xi by restarting the time march; with the implicit \
                 method use 'limiter', 'upwind' or a fixed xi"
                    .into(),
            ));
        }
        let effective = self.effective_field(
            field_rms,
            2.0 * PI * options.frequency_hz,
            advection,
            options.initial_temperature_ev,
        )?;
        let mut start = match options.initial_state {
            Some(state) => self.resolve_initial(Some(state), options.initial_temperature_ev)?,
            None => self.periodic_start(&effective.steady)?,
        };
        let stepper = TimeStepper::new(&self.mesh, &self.collision, advection, self.parallel)
            .map_err(SolverError::InvalidInput)?;
        let sampling =
            RfSampling::new(steps, options.n_store, dt, field_peak, options.frequency_hz);
        let cycle = RfCycle {
            stepper: &stepper,
            acceleration_peak,
            angular_frequency: 2.0 * PI * options.frequency_hz,
            dt,
            steps,
            sampling: &sampling,
        };
        let mut work = stepper.buffers();
        let mut meter = FluxMeter::new(&self.mesh, advection).map_err(SolverError::InvalidInput)?;
        let mut outer = OuterAnderson::new(RF_CYCLE_DEPTH);
        let mut next = vec![0.0; start.len()];
        let mut inner_iterations = 0;
        let mut cycle_residuals = Vec::new();
        let mut two_term_error = effective.error.clone();
        // 遅いモードの補正を使うか。補正の直後に残差がはっきり増えたら、それ以降は使わない（低圧で
        // 非等方成分の過渡が数周期残る条件では、流束から合わせた低次の演算子が周期ごとに振れるため）
        let mut accelerate = effective.acceleration.is_some();
        let mut previous_residual = f64::INFINITY;
        let mut corrected_from: Option<f64> = None;
        let mut since_correction = 0usize;
        for n_cycles in 1..=options.cycles_max {
            let run = self.implicit_cycle(&cycle, &start, &mut work, &mut meter)?;
            inner_iterations += run.iterations;
            let residual = implicit::l1_distance(&run.end_state, &start);
            cycle_residuals.push(residual);
            if corrected_from
                .take()
                .is_some_and(|before| residual > RF_NDA_GIVE_UP * before)
            {
                accelerate = false;
            }
            // 速いモードが消えて残差の減りが鈍ったら、遅いモードの補正の出番
            let stagnating = residual > RF_STAGNATION * previous_residual;
            previous_residual = residual;
            since_correction += 1;
            // 遅いモードの補正（使わないときも、誤差の見積もりに使う）
            let mut candidate = run.end_state.clone();
            let estimate = match &effective.acceleration {
                Some(acceleration) => {
                    let steps = run.recorder.steps.max(1) as f64;
                    let average: Vec<f64> =
                        run.recorder.sum.iter().map(|value| value / steps).collect();
                    let flux: Vec<f64> = run
                        .recorder
                        .face_flux
                        .iter()
                        .map(|value| value / steps)
                        .collect();
                    let mut target = run.end_state.clone();
                    match acceleration.apply(
                        &self.mesh,
                        &self.collision,
                        &average,
                        &flux,
                        &start,
                        &mut target,
                    ) {
                        Ok(change) => {
                            if accelerate && stagnating && since_correction >= RF_CORRECTION_GAP {
                                candidate = target;
                                since_correction = 0;
                                corrected_from = Some(residual);
                                outer.reset();
                            }
                            change
                        }
                        Err(error) => {
                            two_term_error.get_or_insert(error);
                            accelerate = false;
                            0.0
                        }
                    }
                }
                None => 0.0,
            };
            let converged = n_cycles > 1 && residual.max(estimate) < options.tol;
            if converged || n_cycles == options.cycles_max {
                return Ok(periodic_result(
                    run.end_state,
                    run.recorder,
                    &sampling,
                    PeriodicRun {
                        xi_used: xi_of(advection),
                        converged,
                        n_cycles,
                        steps_per_cycle: steps,
                        dt,
                        inner_iterations,
                        cycle_residuals,
                        two_term_error,
                    },
                    &self.mesh,
                    &self.processes,
                ));
            }
            if since_correction == 0 {
                // 補正した状態から Anderson 加速の履歴を始め直す
                start = candidate;
            } else {
                outer
                    .next(&start, &candidate, &mut next)
                    .map_err(SolverError::InvalidInput)?;
                std::mem::swap(&mut start, &mut next);
            }
        }
        unreachable!()
    }

    /// 実効電場の問題とその定常解（RFの陰解法の既定の初期状態と、非線形拡散加速の拡散係数）。
    ///
    /// 実効値の電場のDC問題に、エネルギーを変えない等方散乱 ω²/ν（ν は全衝突周波数）を加えたもの。
    /// 高周波では定常解が時間平均の分布になり（二項近似の実効電場の関係）、低周波では実効値の電場での
    /// DC解に近づく。定常解は収束していなくても使う。
    fn effective_field(
        &self,
        field_rms: f64,
        angular_frequency: f64,
        advection: AdvectionScheme,
        temperature_ev: f64,
    ) -> Result<EffectiveField, SolverError> {
        let period = 2.0 * PI / angular_frequency;
        let maxwell = self.initial_maxwell(temperature_ev)?;
        // ν ≪ ω のセルでは等方化がほぼ完全なので、上限を ω の 1e3 倍にしておく
        let isotropic = self
            .collision
            .nu_total
            .iter()
            .map(|nu| angular_frequency * angular_frequency / nu.max(1.0e-3 * angular_frequency))
            .collect();
        let problem = ImplicitProblem::new(
            &self.mesh,
            &self.collision,
            E_CHARGE * field_rms / M_E,
            advection,
            self.parallel,
        )
        .and_then(|problem| problem.with_isotropic_scattering(isotropic))
        .map_err(SolverError::InvalidInput)?;
        let outcome = implicit::solve(
            &problem,
            &maxwell,
            true,
            RF_START_TOL,
            RF_START_MAX_ITERATIONS,
            ANDERSON_DEPTH,
        )
        .map_err(SolverError::InvalidInput)?;
        // 二項近似を作れなかった（帯が大きすぎるなど）ときは、非線形拡散加速も使わない
        let acceleration = outcome
            .acceleration_error
            .is_none()
            .then(|| DiffusionAcceleration::new(&problem, period));
        Ok(EffectiveField {
            steady: outcome.state,
            acceleration,
            error: outcome.acceleration_error,
        })
    }

    /// 実効電場の定常解から、周期の始め（t = 0、E = E_peak）の状態を作る。
    ///
    /// 二項近似では、実効電場の解の非等方成分は f₁,eff = −a_rms ν/(ν² + ω²) ∂f₀/∂v（+θ 向きに加速）で、
    /// RF（加速度 −a_peak cos ωt）の t = 0 の値 f₁(0) = +a_peak ν/(ν² + ω²) ∂f₀/∂v はその −√2 倍になる
    /// （どの周波数でも同じ）。そこで、θ について奇の成分を −√2 倍する（負になった値は0にして規格化）。
    /// 非等方成分は低圧ではゆっくり（1周期に exp(−ν_m T) 倍）しか減らないので、向きの逆な初期状態から
    /// 始めると、その過渡が長く残って前処理を乱す。
    fn periodic_start(&self, steady: &[f64]) -> Result<Vec<f64>, SolverError> {
        let mut start: Vec<f64> = (0..steady.len())
            .map(|k| {
                let mirrored = steady[self.mesh.mirror_idx(k)];
                let even = 0.5 * (steady[k] + mirrored);
                let odd = 0.5 * (steady[k] - mirrored);
                even - 2.0_f64.sqrt() * odd
            })
            .collect();
        implicit::clip_and_normalize(&mut start).map_err(SolverError::InvalidInput)?;
        Ok(start)
    }

    /// `start` から1周期を陰的に進め、保存点の値と、周期平均の分布と面の流束を集める。
    fn implicit_cycle(
        &self,
        cycle: &RfCycle<'_>,
        start: &[f64],
        work: &mut StepBuffers,
        meter: &mut FluxMeter,
    ) -> Result<CycleRun, SolverError> {
        let n = start.len();
        let mut previous = start.to_vec();
        let mut current = start.to_vec();
        let mut history = vec![0.0; n];
        let mut guess = vec![0.0; n];
        let mut recorder = CycleRecorder::new(
            cycle.sampling.time.len(),
            start,
            self.mesh.n_eps.saturating_sub(1),
        );
        let mut iterations = 0;
        for k in 0..cycle.steps {
            let time_end = (k + 1) as f64 * cycle.dt;
            let acceleration =
                -cycle.acceleration_peak * (cycle.angular_frequency * time_end).cos();
            let bdf2 = k > 0 && bdf2_history(&current, &previous, cycle.dt, &mut history);
            let inverse_dt = if bdf2 {
                1.5 / cycle.dt
            } else {
                for (h, value) in history.iter_mut().zip(&current) {
                    *h = value / cycle.dt;
                }
                1.0 / cycle.dt
            };
            // 初期値は線形外挿（負の値は0）
            for ((g, now), before) in guess.iter_mut().zip(&current).zip(&previous) {
                *g = if k > 0 {
                    (2.0 * now - before).max(0.0)
                } else {
                    *now
                };
            }
            let coefficients = StepCoefficients {
                acceleration,
                inverse_dt,
                history: &history,
            };
            let outcome = cycle
                .stepper
                .step(
                    &coefficients,
                    &guess,
                    RF_STEP_TOL,
                    RF_STEP_MAX_ITERATIONS,
                    ANDERSON_DEPTH,
                    work,
                )
                .map_err(SolverError::InvalidInput)?;
            if !outcome.converged {
                return Err(SolverError::InvalidInput(format!(
                    "implicit RF step {} of {} did not converge in {RF_STEP_MAX_ITERATIONS} \
                     iterations; increase steps_per_cycle",
                    k + 1,
                    cycle.steps
                )));
            }
            iterations += outcome.iterations;
            previous = std::mem::replace(&mut current, outcome.state);
            recorder.record(&current, cycle.sampling, k + 1, &self.mesh, &self.processes);
            meter.add(
                &current,
                acceleration,
                self.parallel,
                &mut recorder.face_flux,
            );
        }
        Ok(CycleRun {
            end_state: current,
            recorder,
            iterations,
        })
    }
}

/// 独立な計算を Rayon で並列に解く（結果は入力順）。`max_workers` を与えればその数のスレッドで解く。
fn run_parallel<T, R, F>(
    items: Vec<T>,
    max_workers: Option<usize>,
    solve: F,
) -> Result<Vec<Result<R, SolverError>>, SolverError>
where
    T: Send,
    R: Send,
    F: Fn(T) -> Result<R, SolverError> + Sync,
{
    let run = || items.into_par_iter().map(&solve).collect::<Vec<_>>();
    match max_workers {
        Some(0) => Err(SolverError::InvalidInput(
            "max_workers must be positive".into(),
        )),
        Some(workers) => rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|err| SolverError::InvalidInput(format!("thread pool: {err}")))
            .map(|pool| pool.install(run)),
        None => Ok(run()),
    }
}

/// BDF2 の右辺 h = (4 n_k − n_{k−1})/(2Δt) を作る。どこかが負になるなら `false`（後退 Euler にする）。
fn bdf2_history(current: &[f64], previous: &[f64], dt: f64, history: &mut [f64]) -> bool {
    let mut largest = 0.0_f64;
    let mut smallest = 0.0_f64;
    for ((h, now), before) in history.iter_mut().zip(current).zip(previous) {
        *h = (4.0 * now - before) / (2.0 * dt);
        largest = largest.max(*h);
        smallest = smallest.min(*h);
    }
    if smallest < -1.0e-14 * largest {
        return false;
    }
    for h in history.iter_mut() {
        *h = h.max(0.0);
    }
    true
}

/// 1周期の陰的な時間発展の設定。
struct RfCycle<'a> {
    stepper: &'a TimeStepper<'a>,
    acceleration_peak: f64,
    angular_frequency: f64,
    dt: f64,
    steps: usize,
    sampling: &'a RfSampling,
}

struct CycleRun {
    end_state: Vec<f64>,
    recorder: CycleRecorder,
    iterations: usize,
}

/// 実効電場の定常解（RFの初期状態）と、非線形拡散加速。
struct EffectiveField {
    steady: Vec<f64>,
    /// 二項近似を作れなければ `None`。
    acceleration: Option<DiffusionAcceleration>,
    error: Option<String>,
}

/// 段の終わりの状態について、エネルギー面の上向きの流束を実際の移流スキームで求めて足す。
///
/// 面 b（セル b と b + 1 の間）の流束は、移流の出力をエネルギーセルごとに角度で足した値 Y の累積和
/// Γ_b = −Σ_{i ≤ b} Y_i（エネルギー0の面では0）。
struct FluxMeter {
    /// 加速度の向き +1, −1 の順
    operators: [AdvectionOperator; 2],
    output: Vec<f64>,
    edge_flux: Vec<f64>,
    n_theta: usize,
}

impl FluxMeter {
    fn new(mesh: &VelocityMesh, scheme: AdvectionScheme) -> Result<Self, String> {
        let operators = [
            AdvectionOperator::with_scheme(mesh, scheme, 1)?,
            AdvectionOperator::with_scheme(mesh, scheme, -1)?,
        ];
        let edges = operators[0].edge_count().max(operators[1].edge_count());
        Ok(Self {
            operators,
            output: vec![0.0; mesh.n_cells],
            edge_flux: vec![0.0; edges],
            n_theta: mesh.n_theta,
        })
    }

    /// 符号付きの加速度 `acceleration` での面の流束を `sums` に足す。
    fn add(&mut self, state: &[f64], acceleration: f64, parallel: bool, sums: &mut [f64]) {
        let direction = usize::from(acceleration < 0.0);
        self.operators[direction].apply(state, &mut self.output, &mut self.edge_flux, parallel);
        let magnitude = acceleration.abs();
        let mut flux = 0.0;
        for (sum, row) in sums.iter_mut().zip(self.output.chunks(self.n_theta)) {
            flux -= row.iter().sum::<f64>();
            *sum += magnitude * flux;
        }
    }
}

/// 1周期の保存点の値と、時間平均の分布を集める。
struct CycleRecorder {
    mean_energy: Vec<f64>,
    drift_velocity: Vec<f64>,
    ionization: Vec<f64>,
    eedf: Vec<Vec<f64>>,
    state_at_max_field: Vec<f64>,
    /// 各段の終わりの状態の和（周期は一様な段に分かれるので、段数で割れば時間平均）。
    sum: Vec<f64>,
    /// 各段の終わりのエネルギー面の流束の和（陰解法だけ。`FluxMeter`）。
    face_flux: Vec<f64>,
    steps: usize,
}

impl CycleRecorder {
    fn new(count: usize, start: &[f64], faces: usize) -> Self {
        Self {
            mean_energy: vec![0.0; count],
            drift_velocity: vec![0.0; count],
            ionization: vec![0.0; count],
            eedf: vec![Vec::new(); count],
            state_at_max_field: start.to_vec(),
            sum: vec![0.0; start.len()],
            face_flux: vec![0.0; faces],
            steps: 0,
        }
    }

    /// 段 `step`（1から数える）の終わりの状態を足し、保存点なら値を記録する。
    fn record(
        &mut self,
        state: &[f64],
        sampling: &RfSampling,
        step: usize,
        mesh: &VelocityMesh,
        processes: &[ProcessSpec],
    ) {
        for (total, value) in self.sum.iter_mut().zip(state) {
            *total += value;
        }
        self.steps += 1;
        if let Some(index) = sampling.slot[step] {
            let (energy, drift) = cheap_scalars(state, mesh);
            self.mean_energy[index] = energy;
            self.drift_velocity[index] = drift;
            self.ionization[index] = reduced_ionization_frequency(state, mesh, processes);
            self.eedf[index] = energy_distribution(state, mesh);
            if index == sampling.index_max_field {
                self.state_at_max_field.copy_from_slice(state);
            }
        }
    }
}

/// 1周期の保存点。時刻 t_i = m_i Δt（m_i = round(i N / n)、0 は周期の終わりと同じ状態）の値を、
/// 段 m_i（m_i = 0 なら段 N）の終わりに保存する。
struct RfSampling {
    slot: Vec<Option<usize>>,
    time: Vec<f64>,
    field: Vec<f64>,
    index_max_field: usize,
}

impl RfSampling {
    fn new(steps: usize, n_store: usize, dt: f64, field_peak: f64, frequency_hz: f64) -> Self {
        let count = n_store.min(steps);
        let mut slot = vec![None; steps + 1];
        let mut time = Vec::with_capacity(count);
        for i in 0..count {
            let m = ((i * steps) as f64 / count as f64).round() as usize % steps;
            slot[if m == 0 { steps } else { m }] = Some(i);
            time.push(m as f64 * dt);
        }
        let field: Vec<f64> = time
            .iter()
            .map(|t| field_peak * (2.0 * PI * frequency_hz * t).cos())
            .collect();
        let index_max_field = field
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map_or(0, |(index, _)| index);
        Self {
            slot,
            time,
            field,
            index_max_field,
        }
    }
}

struct PeriodicRun {
    xi_used: f64,
    converged: bool,
    n_cycles: usize,
    steps_per_cycle: usize,
    dt: f64,
    inner_iterations: usize,
    cycle_residuals: Vec<f64>,
    two_term_error: Option<String>,
}

/// 最後の1周期の記録から、実効値、位相遅れ、時間平均の値を計算して結果にまとめる。
fn periodic_result(
    state: Vec<f64>,
    recorder: CycleRecorder,
    sampling: &RfSampling,
    run: PeriodicRun,
    mesh: &VelocityMesh,
    processes: &[ProcessSpec],
) -> RfResult {
    let phase_field_1 = dft_phase(&sampling.field, 1);
    let phase_drift_1 = dft_phase(&recorder.drift_velocity, 1);
    let phase_energy_2 = dft_phase(&recorder.mean_energy, 2);
    let average: Vec<f64> = recorder
        .sum
        .iter()
        .map(|value| value / recorder.steps.max(1) as f64)
        .collect();
    RfResult {
        state,
        swarm_at_max_field: compute_swarm(&recorder.state_at_max_field, mesh, processes),
        xi_used: run.xi_used,
        converged: run.converged,
        n_cycles: run.n_cycles,
        steps_per_cycle: run.steps_per_cycle,
        dt: run.dt,
        mean_energy_rms: rms(&recorder.mean_energy),
        drift_velocity_rms: rms(&recorder.drift_velocity),
        ionization_rms_over_n: rms(&recorder.ionization),
        phase_delay_energy: wrap_angle(phase_energy_2 - 2.0 * phase_field_1),
        phase_delay_drift: wrap_angle(phase_drift_1 - phase_field_1),
        swarm_average: compute_swarm(&average, mesh, processes),
        time: sampling.time.clone(),
        field: sampling.field.clone(),
        mean_energy: recorder.mean_energy,
        drift_velocity: recorder.drift_velocity,
        reduced_ionization_frequency: recorder.ionization,
        eedf_t: recorder.eedf,
        inner_iterations: run.inner_iterations,
        cycle_residuals: run.cycle_residuals,
        two_term_error: run.two_term_error,
    }
}

/// エネルギーセルごとの EEDF（eV⁻¹）。
fn energy_distribution(state: &[f64], mesh: &VelocityMesh) -> Vec<f64> {
    let total: f64 = state.iter().sum();
    state
        .chunks(mesh.n_theta)
        .zip(&mesh.d_eps)
        .map(|(row, width)| row.iter().sum::<f64>() / (total * width))
        .collect()
}

enum RfMarch {
    Complete(Box<RfResult>),
    Negative { step: usize },
}

/// スキーム名から移流スキームを決める。2つ目の値は ξ を下げながら探索するか（`blending`）。
///
/// - `upwind`: ξ = 0
/// - `limiter`: van Leer の制限関数（2次精度、負にならない）
/// - `xi` を指定: その ξ に固定
/// - `blending`: ξ = 1 から始め、負の値が出るたびに 0.02 ずつ下げる
fn choose_scheme(scheme: &str, xi: Option<f64>) -> Result<(AdvectionScheme, bool), SolverError> {
    if scheme == "upwind" {
        Ok((AdvectionScheme::Linear(0.0), false))
    } else if scheme == "limiter" {
        Ok((AdvectionScheme::VanLeer, false))
    } else if let Some(xi) = xi {
        Ok((AdvectionScheme::Linear(xi), false))
    } else if scheme == "blending" {
        Ok((AdvectionScheme::Linear(1.0), true))
    } else {
        Err(SolverError::InvalidInput(format!(
            "unknown scheme: {scheme}"
        )))
    }
}

/// 結果に記録する ξ。制限関数スキームでは面ごとに変わるので NaN とする。
fn xi_of(advection: AdvectionScheme) -> f64 {
    match advection {
        AdvectionScheme::Linear(xi) => xi,
        AdvectionScheme::VanLeer => f64::NAN,
    }
}

fn lower_xi(xi: f64) -> f64 {
    let lowered = (xi - 0.02).max(0.0);
    if lowered.abs() < 1.0e-12 {
        0.0
    } else {
        lowered
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rf_sampling_covers_the_whole_cycle() {
        let sampling = RfSampling::new(256, 64, 1.0, 1.0, 1.0 / 256.0);
        assert_eq!(sampling.time.len(), 64);
        // 時刻0の値は周期の終わり（段256）の状態、ほかは4段ごと
        assert_eq!(sampling.slot[256], Some(0));
        assert_eq!(sampling.slot[4], Some(1));
        assert_eq!(sampling.slot[252], Some(63));
        assert_eq!(sampling.slot.iter().flatten().count(), 64);
        assert!((sampling.time[63] - 252.0).abs() < 1e-12);
        // |E| が最大の点（t = 0 と半周期の2点のどちらか）
        assert!((sampling.field[sampling.index_max_field].abs() - 1.0).abs() < 1e-12);
        // 保存点数が段数より多ければ、全段を保存する
        let every = RfSampling::new(8, 64, 1.0, 1.0, 1.0 / 8.0);
        assert_eq!(every.time.len(), 8);
        assert_eq!(every.slot.iter().flatten().count(), 8);
    }

    #[test]
    fn bdf2_history_falls_back_when_it_would_be_negative() {
        let mut history = vec![0.0; 2];
        assert!(bdf2_history(&[1.0, 2.0], &[1.0, 1.0], 0.5, &mut history));
        assert_eq!(history, vec![3.0, 7.0]);
        assert!(!bdf2_history(&[1.0, 0.1], &[1.0, 1.0], 0.5, &mut history));
    }
}
