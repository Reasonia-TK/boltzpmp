//! DC定常解とRFの時間発展の陰解法。
//!
//! 定常状態 `a·A(n) + C(n) = λ n`（λ は電離と付着による電子数の増加率。和で規格化する陽解法と同じ）を
//! 次の不動点反復で解く。
//!
//! ```text
//! (ν + λ⁺ − a·A_up) n' = G(n) + a·(A(n) − A_up(n)) + λ⁻ n,    n' を和が1になるよう規格化
//! ```
//!
//! - 左辺は風上差分の移流と衝突の損失で、流れに沿った1回の走査（`UpwindSweep`）で厳密に解ける。
//! - `G` は衝突による再注入（異方散乱と熱運動の交換を含む）。前の反復の値を使う（ソース反復）。
//! - `A` が制限関数スキームのときの高次の補正は、欠損補正として右辺に入れる。
//! - λ ≥ 0 は左辺、λ < 0 は右辺に入れ、どちらでも右辺の各項が正になるようにする。
//!
//! ソース反復は衝突1回分ずつしか進まない。エネルギー緩和のように多くの衝突を経てゆっくり進むモードは、
//! エネルギーだけの二項近似の演算子 L₂（`two_term`）で直接解いて補正する（合成加速）。
//!
//! ```text
//! n' = g(n)（上の走査）,   L₂ δ = P S (n' − n),   n ← n' + E δ
//! ```
//!
//! S は右辺に回した項（再注入、等方散乱、λ⁻、高次の移流の補正）、P は角度についての和、E は等方な分布への
//! 展開。δ の和は0にする（規格化の向きは補正しない）。残りのモードは Anderson 加速（履歴 `depth`）で収束を
//! 速める。補正は n' = n で0になるので、不動点は陽解法の定常解と同じ離散方程式を満たす。
//!
//! RF の1段（`TimeStepper`）は、対角に c/Δt、右辺に前の段の値の組み合わせ h を加えた同じ形の方程式で、
//! 同じ反復で解く（後退 Euler: c = 1, h = n_k/Δt。BDF2: c = 3/2, h = (4 n_k − n_{k−1})/(2Δt)）。
//! 和をとると c/Δt·Σn' = Σh となるので、和が1の解はそのまま時間発展の解になる。1段では対角の c/Δt が
//! 大きく、ソース反復が数回で収束するので、合成加速は使わない。

use crate::{
    AdvectionOperator, AdvectionScheme, CollisionOperator, VelocityMesh,
    operators::UpwindSweep,
    two_term::{TwoTermParameters, TwoTermSolver, cell_volumes, field_diffusion},
};

pub(crate) struct ImplicitOutcome {
    pub state: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

/// 定常反復（`solve`）の結果。
pub(crate) struct SteadySolution {
    pub state: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    /// 二項近似の合成加速を使えなかったときの理由（使えたら `None`）。
    pub acceleration_error: Option<String>,
}

struct Buffers {
    collision: Vec<f64>,
    energy_sum: Vec<f64>,
    reinject: Vec<f64>,
    source: Vec<f64>,
    diagonal: Vec<f64>,
    advection_high: Vec<f64>,
    advection_upwind: Vec<f64>,
    edge_flux: Vec<f64>,
    /// 合成加速で使う、反復の差 n' − n と、エネルギーセルごとの右辺と補正。
    residual: Vec<f64>,
    energy_rhs: Vec<f64>,
    energy_delta: Vec<f64>,
    energy_change: Vec<f64>,
}

impl Buffers {
    fn new(mesh: &VelocityMesh, edges: usize, synthetic: bool) -> Self {
        let (cells, energies) = if synthetic {
            (mesh.n_cells, mesh.n_eps)
        } else {
            (0, 0)
        };
        Self {
            collision: vec![0.0; mesh.n_cells],
            energy_sum: vec![0.0; mesh.n_eps],
            reinject: vec![0.0; mesh.n_eps],
            source: vec![0.0; mesh.n_cells],
            diagonal: vec![0.0; mesh.n_eps],
            advection_high: vec![0.0; mesh.n_cells],
            advection_upwind: vec![0.0; mesh.n_cells],
            edge_flux: vec![0.0; edges],
            residual: vec![0.0; cells],
            energy_rhs: vec![0.0; energies],
            energy_delta: vec![0.0; energies],
            energy_change: vec![0.0; energies],
        }
    }
}

pub(crate) struct ImplicitProblem<'a> {
    mesh: &'a VelocityMesh,
    collision: &'a CollisionOperator,
    acceleration: f64,
    scheme: AdvectionScheme,
    parallel: bool,
    sweep: UpwindSweep,
    correction: Option<(AdvectionOperator, AdvectionOperator)>,
    /// エネルギーを変えない等方散乱の周波数（エネルギーセルごと）。RFの初期状態を作るときだけ使う。
    isotropic: Option<Vec<f64>>,
}

impl<'a> ImplicitProblem<'a> {
    pub fn new(
        mesh: &'a VelocityMesh,
        collision: &'a CollisionOperator,
        acceleration: f64,
        scheme: AdvectionScheme,
        parallel: bool,
    ) -> Result<Self, String> {
        let sweep = UpwindSweep::new(mesh, 1)?;
        let correction = if scheme == AdvectionScheme::Linear(0.0) {
            None
        } else {
            Some((
                AdvectionOperator::with_scheme(mesh, scheme, 1)?,
                AdvectionOperator::new(mesh, 0.0, 1)?,
            ))
        };
        Ok(Self {
            mesh,
            collision,
            acceleration,
            scheme,
            parallel,
            sweep,
            correction,
            isotropic: None,
        })
    }

    /// エネルギーを変えない等方散乱（周波数はエネルギーセルごと）を加える。
    ///
    /// 周波数 ω²/ν_m の散乱を加えたDC解は、高周波（ω ≫ エネルギー緩和の周波数）のRFの時間平均の
    /// 分布になる（二項近似の実効電場 E_eff² = E_rms² ν_m²/(ν_m² + ω²) と同じ）。
    pub fn with_isotropic_scattering(mut self, frequency: Vec<f64>) -> Result<Self, String> {
        if frequency.len() != self.mesh.n_eps
            || frequency.iter().any(|f| !f.is_finite() || *f < 0.0)
        {
            return Err(
                "isotropic scattering needs one finite, non-negative frequency per energy cell"
                    .into(),
            );
        }
        self.isotropic = Some(frequency);
        Ok(self)
    }

    fn buffers(&self) -> Buffers {
        let edges = self
            .correction
            .as_ref()
            .map_or(0, |(high, _)| high.edge_count());
        Buffers::new(self.mesh, edges, true)
    }

    /// 1回の反復 `next = g(x)`（`x` は和が1）。戻り値は `x` での増加率 λ。
    fn map(&self, x: &[f64], next: &mut [f64], work: &mut Buffers) -> Result<f64, String> {
        let n_theta = self.mesh.n_theta;
        let nu = &self.collision.nu_total;
        self.collision.apply(
            x,
            &mut work.collision,
            &mut work.energy_sum,
            &mut work.reinject,
            self.parallel,
        );
        let growth: f64 = work.collision.iter().sum::<f64>() / x.iter().sum::<f64>();
        for (k, source) in work.source.iter_mut().enumerate() {
            *source = work.collision[k] + nu[k / n_theta] * x[k];
        }
        if let Some((high, upwind)) = &self.correction {
            high.apply(
                x,
                &mut work.advection_high,
                &mut work.edge_flux,
                self.parallel,
            );
            upwind.apply(
                x,
                &mut work.advection_upwind,
                &mut work.edge_flux,
                self.parallel,
            );
            for (k, source) in work.source.iter_mut().enumerate() {
                *source += self.acceleration * (work.advection_high[k] - work.advection_upwind[k]);
            }
        }
        let (shift, extra) = if growth >= 0.0 {
            (growth, 0.0)
        } else {
            (0.0, -growth)
        };
        for (diagonal, frequency) in work.diagonal.iter_mut().zip(nu) {
            *diagonal = frequency + shift;
        }
        if extra > 0.0 {
            for (source, value) in work.source.iter_mut().zip(x) {
                *source += extra * value;
            }
        }
        if let Some(frequency) = &self.isotropic {
            let weight_sum: f64 = self.mesh.w_theta.iter().sum();
            for (i, f) in frequency.iter().enumerate() {
                let row = i * n_theta..(i + 1) * n_theta;
                let total: f64 = x[row.clone()].iter().sum();
                for (source, weight) in work.source[row].iter_mut().zip(&self.mesh.w_theta) {
                    *source += f * total * weight / weight_sum;
                }
                work.diagonal[i] += f;
            }
        }
        self.sweep
            .solve(self.acceleration, &work.diagonal, &work.source, next)?;
        // 欠損補正で生じうる小さな負の値を除き、和を1にする
        clip_and_normalize(next)?;
        Ok(growth)
    }

    /// 二項近似の演算子（モジュールの説明と `Synthetic` を参照）。
    fn two_term(&self) -> Result<TwoTermSolver, String> {
        TwoTermSolver::new(
            self.mesh,
            self.collision,
            &TwoTermParameters {
                acceleration: self.acceleration,
                scheme: self.scheme,
                isotropic: self.isotropic.as_deref(),
            },
        )
    }

    /// 合成加速の補正 `next ← next + β E δ`、`L₂ δ = P [S(next) − S(x)]`（和は1のまま、β は
    /// `SYNTHETIC_DAMPING`）。
    ///
    /// S は `map` で前の反復の値を使う項。再注入などは線形なので S(next − x) で計算する。高次の移流の補正
    /// a(A − A_up) は、制限関数スキームでは非線形なので、`map` が `work` に残した x での値との差をとる。
    fn synthetic_correction(
        &self,
        solver: &TwoTermSolver,
        growth: f64,
        x: &[f64],
        next: &mut [f64],
        work: &mut Buffers,
    ) -> Result<(), String> {
        let n_theta = self.mesh.n_theta;
        work.energy_change.fill(0.0);
        if let Some((high, upwind)) = &self.correction {
            let sums = |work: &Buffers, i: usize| -> f64 {
                let row = i * n_theta..(i + 1) * n_theta;
                work.advection_high[row.clone()]
                    .iter()
                    .zip(&work.advection_upwind[row])
                    .map(|(high, low)| high - low)
                    .sum()
            };
            for i in 0..self.mesh.n_eps {
                work.energy_change[i] = -sums(work, i);
            }
            high.apply(
                next,
                &mut work.advection_high,
                &mut work.edge_flux,
                self.parallel,
            );
            upwind.apply(
                next,
                &mut work.advection_upwind,
                &mut work.edge_flux,
                self.parallel,
            );
            for i in 0..self.mesh.n_eps {
                work.energy_change[i] += sums(work, i);
            }
        }
        for ((difference, after), before) in work.residual.iter_mut().zip(next.iter()).zip(x) {
            *difference = after - before;
        }
        // 出力は −ν r ＋ 再注入。energy_sum には r の角度についての和が入る
        self.collision.apply(
            &work.residual,
            &mut work.collision,
            &mut work.energy_sum,
            &mut work.reinject,
            self.parallel,
        );
        let extra = (-growth).max(0.0);
        for (i, rhs) in work.energy_rhs.iter_mut().enumerate() {
            let total = work.energy_sum[i];
            let reinjected = work.collision[i * n_theta..(i + 1) * n_theta]
                .iter()
                .sum::<f64>()
                + self.collision.nu_total[i] * total;
            let isotropic = self.isotropic.as_ref().map_or(0.0, |f| f[i] * total);
            *rhs =
                reinjected + isotropic + extra * total + self.acceleration * work.energy_change[i];
        }
        solver.solve_zero_sum(&work.energy_rhs, &mut work.energy_delta);
        for (row, delta) in next.chunks_mut(n_theta).zip(&work.energy_delta) {
            for (value, weight) in row.iter_mut().zip(&self.mesh.w_theta) {
                *value += SYNTHETIC_DAMPING * delta * weight;
            }
        }
        clip_and_normalize(next)
    }
}

/// RFの周期写像の遅いモード（エネルギー緩和）の補正。低次の演算子を高次の解の流束に合わせる
/// （中性子輸送の非線形拡散加速・CMFD と同じ考え方）。
///
/// 1周期の高次（輸送）の解から、周期平均の等方な分布 ū と、各エネルギー面の周期平均の流束 Γ̄ を求め、
/// 面の流束を
///
/// ```text
/// Γ_b = K̂_b (f_b − f_{b+1}) + w_b f_{風上}
/// ```
///
/// とした低次の演算子 L̂（衝突はそのまま、λ は ū での増加率）を作る。
///
/// - K̂ は、Γ̄ と ū の勾配の向きが同じ面では K̂ = Γ̄/(f̄_b − f̄_{b+1})（拡散係数そのものを合わせる。数値拡散も
///   拡散の形をしている）。そうでない面は実効電場の二項近似の係数 K のままにして、残りを移動の項 w で合わせる。
///   低圧の粗い格子では、実際の離散化の加熱が二項近似の何十倍にもなるので、合わせないと補正が大きく外れる。
/// - 遅いモードでは1周期の等方成分の変化が Δū ≈ −T L̂ (ū − ū*) なので、δ = L̂⁻¹ Δū/T（Σδ = 0）を等方成分に
///   足す。周期解では Δū = 0 なので δ = 0（不動点は変わらない）。‖δ‖₁ は遅いモードの誤差の見積もりになる。
/// - 非等方成分は変えない（倍率を掛けると非等方成分も同じ倍率で変わり、新しい過渡が起きる）。
pub(crate) struct DiffusionAcceleration {
    diffusion: Vec<f64>,
    volume: Vec<f64>,
    n_theta: usize,
    period: f64,
}

/// 合わせた拡散係数を、二項近似の係数のこの倍の範囲に収める。
const NDA_FIT_RANGE: f64 = 1.0e6;

impl DiffusionAcceleration {
    /// `problem` は等方散乱を加えた実効電場の問題（K を作るのに使う）、`period` は周期。
    pub fn new(problem: &ImplicitProblem<'_>, period: f64) -> Self {
        let mesh = problem.mesh;
        let relaxation: Vec<f64> = (0..mesh.n_eps)
            .map(|i| {
                problem.collision.nu_momentum[i]
                    + problem
                        .isotropic
                        .as_ref()
                        .map_or(0.0, |frequency| frequency[i])
            })
            .collect();
        Self {
            diffusion: field_diffusion(mesh, problem.acceleration, &relaxation),
            volume: cell_volumes(mesh),
            n_theta: mesh.n_theta,
            period,
        }
    }

    /// 周期平均の状態 `average` と面の流束 `face_flux`（周期平均、上向きが正）から低次の演算子 L̂ を作り、
    /// 1周期の等方成分の変化 Δū（`state` − `start`、`state` は周期の終わりの状態）から
    /// `L̂ δ = Δū / T`（Σδ = 0）を解いて、`state` の等方成分に δ を足す（和は1に規格化）。
    /// 戻り値は ‖δ‖₁（遅いモードの誤差の見積もり）。
    pub fn apply(
        &self,
        mesh: &VelocityMesh,
        collision: &CollisionOperator,
        average: &[f64],
        face_flux: &[f64],
        start: &[f64],
        state: &mut [f64],
    ) -> Result<f64, String> {
        let n = mesh.n_eps;
        let total: f64 = average.iter().sum();
        let mean: Vec<f64> = average
            .chunks(self.n_theta)
            .map(|row| row.iter().sum::<f64>() / total)
            .collect();
        let density: Vec<f64> = mean.iter().zip(&self.volume).map(|(u, v)| u / v).collect();
        let faces = n.saturating_sub(1);
        let mut diffusion = self.diffusion.clone();
        let mut drift = vec![0.0; faces];
        for b in 0..faces {
            let gradient = density[b] - density[b + 1];
            if gradient * face_flux[b] > 0.0 {
                let base = self.diffusion[b];
                diffusion[b] =
                    (face_flux[b] / gradient).clamp(base / NDA_FIT_RANGE, base * NDA_FIT_RANGE);
            }
            let correction = face_flux[b] - diffusion[b] * gradient;
            let upwind = if correction > 0.0 {
                density[b]
            } else {
                density[b + 1]
            };
            if upwind > 1.0e-280 {
                drift[b] = correction / upwind;
            }
        }
        let growth = {
            let mut output = vec![0.0; average.len()];
            let mut energy_sum = vec![0.0; n];
            let mut reinject = vec![0.0; n];
            collision.apply(average, &mut output, &mut energy_sum, &mut reinject, false);
            output.iter().sum::<f64>() / total
        };
        let solver = TwoTermSolver::with_fluxes(mesh, collision, &diffusion, &drift, growth)?;
        let rhs: Vec<f64> = state
            .chunks(self.n_theta)
            .zip(start.chunks(self.n_theta))
            .map(|(after, before)| {
                (after.iter().sum::<f64>() - before.iter().sum::<f64>()) / self.period
            })
            .collect();
        let mut delta = vec![0.0; n];
        solver.solve_zero_sum(&rhs, &mut delta);
        for (row, d) in state.chunks_mut(self.n_theta).zip(&delta) {
            for (value, weight) in row.iter_mut().zip(&mesh.w_theta) {
                *value += d * weight;
            }
        }
        clip_and_normalize(state)?;
        Ok(delta.iter().map(|d| d.abs()).sum())
    }
}

/// 定常反復の合成加速。二項近似の演算子は最初に一度だけ作る。
///
/// 演算子には電子数の増加率 λ を入れない。λ は反復の途中で大きく変わり（初期状態の Maxwell 分布と解とで
/// 電離の割合が違う）、その値を入れた演算子はかえって遅いモードの見積もりを外すことがあった
/// （同梱 Ar の 2 Td で収束しなかった）。λ の効果は、外側の反復の規格化と右辺の λ⁻ が受け持つ。
struct Synthetic {
    solver: Option<TwoTermSolver>,
    error: Option<String>,
}

/// 合成加速の補正に掛ける係数 β。
///
/// L₂ が遅いモードの減衰を小さく見積もる（制限関数が極値で風上差分になる、二項近似の誤差など）と、
/// 補正が行き過ぎて振動し、β = 1 では収束しないことがある。遅いモードの誤差の倍率は 1 − β μ/μ₂
/// （μ, μ₂ は真の演算子と L₂ の固有値）なので、μ/μ₂ < 2/β なら安定になる（β = 0.7 で約 2.9 倍まで）。
/// 同梱 Ar と HF の DC では、β = 0.55〜0.85 でほぼ同じ反復回数だった。
const SYNTHETIC_DAMPING: f64 = 0.7;

impl Synthetic {
    fn new() -> Self {
        Self {
            solver: None,
            error: None,
        }
    }

    fn correct(
        &mut self,
        problem: &ImplicitProblem<'_>,
        growth: f64,
        x: &[f64],
        next: &mut [f64],
        work: &mut Buffers,
    ) -> Result<(), String> {
        match self.ensure(problem) {
            Some(solver) => problem.synthetic_correction(solver, growth, x, next, work),
            None => Ok(()),
        }
    }

    /// 演算子（初めて呼ばれたときに作る）。作れなければ `None`（理由は `error`）。
    fn ensure(&mut self, problem: &ImplicitProblem<'_>) -> Option<&TwoTermSolver> {
        if self.solver.is_none() && self.error.is_none() {
            match problem.two_term() {
                Ok(solver) => self.solver = Some(solver),
                Err(error) => self.error = Some(error),
            }
        }
        self.solver.as_ref()
    }
}

pub(crate) fn clip_and_normalize(values: &mut [f64]) -> Result<(), String> {
    let mut total = 0.0;
    for value in values.iter_mut() {
        if !value.is_finite() {
            return Err("implicit iteration produced a non-finite value".into());
        }
        if *value < 0.0 {
            *value = 0.0;
        }
        total += *value;
    }
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err("implicit iteration lost all electrons".into());
    }
    for value in values.iter_mut() {
        *value /= total;
    }
    Ok(())
}

/// Anderson加速（type II）。`x_{k+1} = g_k − ΔG γ`、γ は `‖f_k − ΔF γ‖₂` を最小にする係数。
struct Anderson {
    depth: usize,
    gs: Vec<Vec<f64>>,
    fs: Vec<Vec<f64>>,
}

impl Anderson {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            gs: Vec::new(),
            fs: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.gs.clear();
        self.fs.clear();
    }

    fn next(&mut self, x: &[f64], g: &[f64], out: &mut [f64]) -> Result<(), String> {
        self.combine(x, g, out);
        clip_and_normalize(out)
    }

    /// 加速した値（規格化も負の値の除去もしない）。
    fn combine(&mut self, x: &[f64], g: &[f64], out: &mut [f64]) {
        let f: Vec<f64> = g.iter().zip(x).map(|(a, b)| a - b).collect();
        self.gs.push(g.to_vec());
        self.fs.push(f);
        if self.gs.len() > self.depth + 1 {
            self.gs.remove(0);
            self.fs.remove(0);
        }
        let m = self.gs.len() - 1;
        if m == 0 {
            out.copy_from_slice(g);
            return;
        }
        let current = &self.fs[m];
        let delta_f: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                self.fs[i + 1]
                    .iter()
                    .zip(&self.fs[i])
                    .map(|(a, b)| a - b)
                    .collect()
            })
            .collect();
        // 正規方程式 (ΔFᵀΔF + δI) γ = ΔFᵀ f_k
        let mut gram = vec![0.0; m * m];
        let mut rhs = vec![0.0; m];
        for i in 0..m {
            rhs[i] = dot(&delta_f[i], current);
            for j in 0..=i {
                let value = dot(&delta_f[i], &delta_f[j]);
                gram[i * m + j] = value;
                gram[j * m + i] = value;
            }
        }
        let trace: f64 = (0..m).map(|i| gram[i * m + i]).sum();
        let regularization = 1.0e-12 * trace.max(f64::MIN_POSITIVE) / m as f64;
        for i in 0..m {
            gram[i * m + i] += regularization;
        }
        let Some(gamma) = solve_dense(&mut gram, &mut rhs, m) else {
            self.reset();
            out.copy_from_slice(g);
            return;
        };
        out.copy_from_slice(g);
        for (i, coefficient) in gamma.iter().enumerate() {
            for ((value, newer), older) in out.iter_mut().zip(&self.gs[i + 1]).zip(&self.gs[i]) {
                *value -= coefficient * (newer - older);
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// 部分ピボット付きのガウス消去。特異なら `None`。
fn solve_dense(matrix: &mut [f64], rhs: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for column in 0..n {
        let pivot = (column..n).max_by(|a, b| {
            matrix[a * n + column]
                .abs()
                .total_cmp(&matrix[b * n + column].abs())
        })?;
        if matrix[pivot * n + column].abs() < 1.0e-300 {
            return None;
        }
        if pivot != column {
            for j in 0..n {
                matrix.swap(pivot * n + j, column * n + j);
            }
            rhs.swap(pivot, column);
        }
        for row in column + 1..n {
            let factor = matrix[row * n + column] / matrix[column * n + column];
            for j in column..n {
                matrix[row * n + j] -= factor * matrix[column * n + j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = rhs[row];
        for j in row + 1..n {
            value -= matrix[row * n + j] * solution[j];
        }
        solution[row] = value / matrix[row * n + row];
    }
    solution.iter().all(|v| v.is_finite()).then_some(solution)
}

/// 残差 `‖g(x) − x‖₁` が `tol` 未満になるまで反復する（g は合成加速を含む1回の反復）。
///
/// `two_term_start` なら、二項近似の定常解（等方な分布）から始める（二項近似を使えなければ `initial` から）。
/// 二項近似の解は多くの場合すでに解に近いので、遠い初期状態から大きな補正をかけて負の値を切り捨てるより、
/// ずっと早く収束する。
pub(crate) fn solve(
    problem: &ImplicitProblem<'_>,
    initial: &[f64],
    two_term_start: bool,
    tol: f64,
    max_iterations: usize,
    depth: usize,
) -> Result<SteadySolution, String> {
    let mut work = problem.buffers();
    let mut synthetic = Synthetic::new();
    let mut start = initial.to_vec();
    if two_term_start && let Some(solver) = synthetic.ensure(problem) {
        let mesh = problem.mesh;
        for (row, value) in start.chunks_mut(mesh.n_theta).zip(solver.steady()) {
            for (cell, weight) in row.iter_mut().zip(&mesh.w_theta) {
                *cell = value * weight;
            }
        }
    }
    let outcome = iterate(
        |x, g| {
            let growth = problem.map(x, g, &mut work)?;
            synthetic.correct(problem, growth, x, g, &mut work)
        },
        &start,
        tol,
        max_iterations,
        depth,
    )?;
    Ok(SteadySolution {
        state: outcome.state,
        converged: outcome.converged,
        iterations: outcome.iterations,
        acceleration_error: synthetic.error,
    })
}

/// 不動点 `x = g(x)`（`g` は和が1の状態を返す）を Anderson 加速で求める。残差は `‖g(x) − x‖₁`。
pub(crate) fn iterate<F>(
    mut map: F,
    initial: &[f64],
    tol: f64,
    max_iterations: usize,
    depth: usize,
) -> Result<ImplicitOutcome, String>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<(), String>,
{
    let mut x = initial.to_vec();
    clip_and_normalize(&mut x)?;
    let mut g = vec![0.0; x.len()];
    let mut next = vec![0.0; x.len()];
    let mut anderson = Anderson::new(depth);
    let mut best = f64::INFINITY;
    for iteration in 1..=max_iterations {
        map(&x, &mut g)?;
        let residual = l1_distance(&g, &x);
        if residual < tol {
            return Ok(ImplicitOutcome {
                state: g,
                converged: true,
                iterations: iteration,
            });
        }
        // 残差が最良値から大きく悪化したら加速の履歴を捨てる
        if residual > 10.0 * best {
            anderson.reset();
        }
        best = best.min(residual);
        if depth == 0 {
            std::mem::swap(&mut x, &mut g);
            continue;
        }
        anderson.next(&x, &g, &mut next)?;
        std::mem::swap(&mut x, &mut next);
    }
    map(&x, &mut g)?;
    Ok(ImplicitOutcome {
        state: g,
        converged: false,
        iterations: max_iterations,
    })
}

pub(crate) fn l1_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// 外側の Anderson 加速で、外挿の幅を決めるのに使わないセルの値（最大値との比）。
const OUTER_NEGLIGIBLE: f64 = 1.0e-10;

/// 周期写像の不動点探索などで、外側の反復に使う Anderson 加速（`next` は和が1に規格化した値を返す）。
pub(crate) struct OuterAnderson {
    inner: Anderson,
    best: f64,
}

impl OuterAnderson {
    pub fn new(depth: usize) -> Self {
        Self {
            inner: Anderson::new(depth),
            best: f64::INFINITY,
        }
    }

    /// 履歴を捨てる（写像を変えたとき）。
    pub fn reset(&mut self) {
        self.inner.reset();
        self.best = f64::INFINITY;
    }

    /// `x` を写した結果が `g`（非負）のとき、次に試す状態を `out` に入れる。
    ///
    /// 加速した値が負になるセルがあれば、`g` からの外挿の幅を負にならない所まで縮める。0 で切り捨てると
    /// 加速の履歴と実際の状態が食い違い、遅いモードの収束が止まる。ただし `g` がほぼ0のセル（分布の裾の
    /// 外）は幅を決めるのに使わず、0 で切り捨てる。そうしないと、そうしたセル1つで外挿の幅が0になる。
    pub fn next(&mut self, x: &[f64], g: &[f64], out: &mut [f64]) -> Result<(), String> {
        if g.iter().any(|value| value.is_nan() || *value < 0.0) {
            return Err("outer acceleration needs a non-negative mapped state".into());
        }
        let residual = l1_distance(g, x);
        if residual > 10.0 * self.best {
            self.inner.reset();
        }
        self.best = self.best.min(residual);
        self.inner.combine(x, g, out);
        let negligible = OUTER_NEGLIGIBLE * g.iter().copied().fold(0.0, f64::max);
        let mut step = 1.0_f64;
        for (value, base) in out.iter().zip(g) {
            if *value < 0.0 && *base > negligible {
                // base > 0 > value なので分母は正
                step = step.min(base / (base - value));
            }
        }
        if step < 1.0 {
            for (value, base) in out.iter_mut().zip(g) {
                *value = base + step * (*value - base);
            }
        }
        clip_and_normalize(out)
    }
}

/// RF の陰的時間発展の1段（モジュールの説明を参照）。
pub(crate) struct TimeStepper<'a> {
    mesh: &'a VelocityMesh,
    collision: &'a CollisionOperator,
    parallel: bool,
    /// 加速度の向き +1, −1 の順
    sweeps: [UpwindSweep; 2],
    corrections: Option<[(AdvectionOperator, AdvectionOperator); 2]>,
}

/// 1段の方程式の係数。`acceleration` は符号付き（負なら θ = π 側へ加速）。
pub(crate) struct StepCoefficients<'b> {
    pub acceleration: f64,
    /// c/Δt
    pub inverse_dt: f64,
    /// 前の段からの右辺 h（非負）
    pub history: &'b [f64],
}

impl<'a> TimeStepper<'a> {
    pub fn new(
        mesh: &'a VelocityMesh,
        collision: &'a CollisionOperator,
        scheme: AdvectionScheme,
        parallel: bool,
    ) -> Result<Self, String> {
        let sweeps = [UpwindSweep::new(mesh, 1)?, UpwindSweep::new(mesh, -1)?];
        let corrections = if scheme == AdvectionScheme::Linear(0.0) {
            None
        } else {
            Some([
                (
                    AdvectionOperator::with_scheme(mesh, scheme, 1)?,
                    AdvectionOperator::new(mesh, 0.0, 1)?,
                ),
                (
                    AdvectionOperator::with_scheme(mesh, scheme, -1)?,
                    AdvectionOperator::new(mesh, 0.0, -1)?,
                ),
            ])
        };
        Ok(Self {
            mesh,
            collision,
            parallel,
            sweeps,
            corrections,
        })
    }

    pub fn buffers(&self) -> StepBuffers {
        let edges = self.corrections.as_ref().map_or(0, |pairs| {
            pairs
                .iter()
                .map(|(high, _)| high.edge_count())
                .max()
                .unwrap_or(0)
        });
        StepBuffers {
            inner: Buffers::new(self.mesh, edges, false),
        }
    }

    /// 1回の反復 `next = g(x)`。
    fn map(
        &self,
        coefficients: &StepCoefficients<'_>,
        x: &[f64],
        next: &mut [f64],
        work: &mut Buffers,
    ) -> Result<(), String> {
        let n_theta = self.mesh.n_theta;
        let nu = &self.collision.nu_total;
        let direction = usize::from(coefficients.acceleration < 0.0);
        let magnitude = coefficients.acceleration.abs();
        self.collision.apply(
            x,
            &mut work.collision,
            &mut work.energy_sum,
            &mut work.reinject,
            self.parallel,
        );
        let growth: f64 = work.collision.iter().sum::<f64>() / x.iter().sum::<f64>();
        for (k, source) in work.source.iter_mut().enumerate() {
            *source = work.collision[k] + nu[k / n_theta] * x[k] + coefficients.history[k];
        }
        if let Some(pairs) = &self.corrections {
            let (high, upwind) = &pairs[direction];
            high.apply(
                x,
                &mut work.advection_high,
                &mut work.edge_flux,
                self.parallel,
            );
            upwind.apply(
                x,
                &mut work.advection_upwind,
                &mut work.edge_flux,
                self.parallel,
            );
            for (k, source) in work.source.iter_mut().enumerate() {
                *source += magnitude * (work.advection_high[k] - work.advection_upwind[k]);
            }
        }
        let (shift, extra) = if growth >= 0.0 {
            (growth, 0.0)
        } else {
            (0.0, -growth)
        };
        for (diagonal, frequency) in work.diagonal.iter_mut().zip(nu) {
            *diagonal = frequency + shift + coefficients.inverse_dt;
        }
        if extra > 0.0 {
            for (source, value) in work.source.iter_mut().zip(x) {
                *source += extra * value;
            }
        }
        self.sweeps[direction].solve(magnitude, &work.diagonal, &work.source, next)?;
        clip_and_normalize(next)
    }

    /// 1段を解く。`guess` は反復の初期値。
    pub fn step(
        &self,
        coefficients: &StepCoefficients<'_>,
        guess: &[f64],
        tol: f64,
        max_iterations: usize,
        depth: usize,
        work: &mut StepBuffers,
    ) -> Result<ImplicitOutcome, String> {
        iterate(
            |x, g| self.map(coefficients, x, g, &mut work.inner),
            guess,
            tol,
            max_iterations,
            depth,
        )
    }
}

/// `TimeStepper` の作業領域。
pub(crate) struct StepBuffers {
    inner: Buffers,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_solver_solves_small_system() {
        let mut matrix = vec![4.0, 1.0, 2.0, 3.0];
        let mut rhs = vec![1.0, 2.0];
        let x = solve_dense(&mut matrix, &mut rhs, 2).unwrap();
        assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1e-14);
        assert!((2.0 * x[0] + 3.0 * x[1] - 2.0).abs() < 1e-14);
    }
}
