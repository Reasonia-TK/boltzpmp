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
//! ソース反復は衝突1回分ずつしか進まないので、Anderson 加速（履歴 `depth`）で収束を速める。
//! 不動点は陽解法の定常解と同じ離散方程式を満たす。
//!
//! RF の1段（`TimeStepper`）は、対角に c/Δt、右辺に前の段の値の組み合わせ h を加えた同じ形の方程式で、
//! 同じ反復で解く（後退 Euler: c = 1, h = n_k/Δt。BDF2: c = 3/2, h = (4 n_k − n_{k−1})/(2Δt)）。
//! 和をとると c/Δt·Σn' = Σh となるので、和が1の解はそのまま時間発展の解になる。

use crate::{
    AdvectionOperator, AdvectionScheme, CollisionOperator, VelocityMesh, operators::UpwindSweep,
};

pub(crate) struct ImplicitOutcome {
    pub state: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
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
}

pub(crate) struct ImplicitProblem<'a> {
    mesh: &'a VelocityMesh,
    collision: &'a CollisionOperator,
    acceleration: f64,
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
        Buffers {
            collision: vec![0.0; self.mesh.n_cells],
            energy_sum: vec![0.0; self.mesh.n_eps],
            reinject: vec![0.0; self.mesh.n_eps],
            source: vec![0.0; self.mesh.n_cells],
            diagonal: vec![0.0; self.mesh.n_eps],
            advection_high: vec![0.0; self.mesh.n_cells],
            advection_upwind: vec![0.0; self.mesh.n_cells],
            edge_flux: vec![0.0; edges],
        }
    }

    /// 1回の反復 `next = g(x)`（`x` は和が1）。
    fn map(&self, x: &[f64], next: &mut [f64], work: &mut Buffers) -> Result<(), String> {
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
        clip_and_normalize(next)
    }

    /// 線形の方程式 `(L − λ) z = s` のソース反復の1回（増加率 λ は固定、規格化はしない）。
    ///
    /// `(ν + λ⁺ − a·A_up) z' = G(z) + a·(A(z) − A_up(z)) + λ⁻ z − s`。`L` は `map` と同じ演算子。
    fn linear_map(
        &self,
        z: &[f64],
        rhs: &[f64],
        growth: f64,
        next: &mut [f64],
        work: &mut Buffers,
    ) -> Result<(), String> {
        let n_theta = self.mesh.n_theta;
        let nu = &self.collision.nu_total;
        self.collision.apply(
            z,
            &mut work.collision,
            &mut work.energy_sum,
            &mut work.reinject,
            self.parallel,
        );
        for (k, source) in work.source.iter_mut().enumerate() {
            *source = work.collision[k] + nu[k / n_theta] * z[k] - rhs[k];
        }
        if let Some((high, upwind)) = &self.correction {
            high.apply(
                z,
                &mut work.advection_high,
                &mut work.edge_flux,
                self.parallel,
            );
            upwind.apply(
                z,
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
            for (source, value) in work.source.iter_mut().zip(z) {
                *source += extra * value;
            }
        }
        if let Some(frequency) = &self.isotropic {
            let weight_sum: f64 = self.mesh.w_theta.iter().sum();
            for (i, f) in frequency.iter().enumerate() {
                let row = i * n_theta..(i + 1) * n_theta;
                let total: f64 = z[row.clone()].iter().sum();
                for (source, weight) in work.source[row].iter_mut().zip(&self.mesh.w_theta) {
                    *source += f * total * weight / weight_sum;
                }
                work.diagonal[i] += f;
            }
        }
        self.sweep
            .solve(self.acceleration, &work.diagonal, &work.source, next)
    }

    /// 状態 `x` での増加率 λ = Σ C(x) / Σ x。
    fn growth_of(&self, x: &[f64], work: &mut Buffers) -> f64 {
        self.collision.apply(
            x,
            &mut work.collision,
            &mut work.energy_sum,
            &mut work.reinject,
            self.parallel,
        );
        work.collision.iter().sum::<f64>() / x.iter().sum::<f64>()
    }
}

/// RFの周期写像の前処理（実効電場の合成加速）。
///
/// 周期写像 Φ の遅いモード（エネルギー緩和）では Φ ≈ exp(T L̄) で、L̄ は周期平均の演算子。
/// 高周波では L̄ ≈ L_eff（実効値の電場 ＋ エネルギーを変えない等方散乱 ω²/ν の定常演算子）なので、
///
/// ```text
/// x' = Φ(x) − z,   T (L_eff − λ) z = Φ(x) − x,   Σz = 0
/// ```
///
/// とすると、遅いモードの誤差は約 μT 倍（μ は L_eff の固有値）、速いモードの誤差は約 1/(μT) 倍になる。
/// z は DC と同じ掃き出しとソース反復（Anderson 加速）で解き、L_eff の零空間（定常解 x_eff の向き）は
/// 反復のたびに取り除く。‖z‖₁ は x の誤差の見積もりにもなる。
pub(crate) struct Preconditioner<'a> {
    problem: ImplicitProblem<'a>,
    steady: Vec<f64>,
    growth: f64,
    period: f64,
    work: Buffers,
}

impl<'a> Preconditioner<'a> {
    /// `problem` は等方散乱を加えた実効電場の問題、`steady` はその定常解（和が1）。
    pub fn new(problem: ImplicitProblem<'a>, steady: Vec<f64>, period: f64) -> Self {
        let mut work = problem.buffers();
        let growth = problem.growth_of(&steady, &mut work);
        Self {
            problem,
            steady,
            growth,
            period,
            work,
        }
    }

    pub fn steady(&self) -> &[f64] {
        &self.steady
    }

    /// 補正 z、反復回数、線形反復が収束したか。z は `T (L_eff − λ) z = P r`（和が0）の解の等方成分 P z。
    ///
    /// P は角度平均（等方成分への射影）。遅いモードはエネルギー分布の形（等方成分）なので、残差も補正も
    /// 等方成分に限る。非等方成分（1周期のうちに減衰する速いモード）を右辺に入れると、電場の結合を通して
    /// 定常な加熱のように働き、ありもしない遅い補正を作ってしまう。
    /// 前処理なので、線形反復が上限までに収束しなくてもその時点の値を返す（収束の判定は外側で行う）。
    pub fn solve(
        &mut self,
        residual: &[f64],
        tol: f64,
        max_iterations: usize,
        depth: usize,
    ) -> Result<(Vec<f64>, usize, bool), String> {
        let mut rhs: Vec<f64> = residual.iter().map(|r| r / self.period).collect();
        isotropic_projection(self.problem.mesh, &mut rhs);
        let problem = &self.problem;
        let steady = &self.steady;
        let growth = self.growth;
        let work = &mut self.work;
        let outcome = iterate_linear(
            |z, next| {
                problem.linear_map(z, &rhs, growth, next, work)?;
                let total: f64 = next.iter().sum();
                for (value, base) in next.iter_mut().zip(steady) {
                    *value -= total * base;
                }
                Ok(())
            },
            &vec![0.0; residual.len()],
            tol,
            max_iterations,
            depth,
        )?;
        let mut correction = outcome.state;
        isotropic_projection(self.problem.mesh, &mut correction);
        Ok((correction, outcome.iterations, outcome.converged))
    }
}

/// 各エネルギーセルの値を、立体角の重みに比例する等方な分布に置き換える（和は保つ）。
fn isotropic_projection(mesh: &VelocityMesh, values: &mut [f64]) {
    let n_theta = mesh.n_theta;
    let weight_sum: f64 = mesh.w_theta.iter().sum();
    for row in values.chunks_mut(n_theta) {
        let total: f64 = row.iter().sum();
        for (value, weight) in row.iter_mut().zip(&mesh.w_theta) {
            *value = total * weight / weight_sum;
        }
    }
}

/// 線形の不動点 `z = g(z)` を Anderson 加速で求める（規格化しない）。`‖g(z) − z‖₁ < tol ‖g(z)‖₁` で収束。
fn iterate_linear<F>(
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
    let mut g = vec![0.0; x.len()];
    let mut next = vec![0.0; x.len()];
    let mut anderson = Anderson::new(depth);
    let mut best = f64::INFINITY;
    for iteration in 1..=max_iterations {
        map(&x, &mut g)?;
        if g.iter().any(|value| !value.is_finite()) {
            return Err("linear iteration produced a non-finite value".into());
        }
        let residual = l1_distance(&g, &x);
        let scale: f64 = g.iter().map(|v| v.abs()).sum();
        if residual <= tol * scale || scale == 0.0 {
            return Ok(ImplicitOutcome {
                state: g,
                converged: true,
                iterations: iteration,
            });
        }
        if residual > 10.0 * best {
            anderson.reset();
        }
        best = best.min(residual);
        anderson.combine(&x, &g, &mut next);
        std::mem::swap(&mut x, &mut next);
    }
    Ok(ImplicitOutcome {
        state: x,
        converged: false,
        iterations: max_iterations,
    })
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

/// 残差 `‖g(x) − x‖₁` が `tol` 未満になるまで反復する。
pub(crate) fn solve(
    problem: &ImplicitProblem<'_>,
    initial: &[f64],
    tol: f64,
    max_iterations: usize,
    depth: usize,
) -> Result<ImplicitOutcome, String> {
    let mut work = problem.buffers();
    iterate(
        |x, g| problem.map(x, g, &mut work),
        initial,
        tol,
        max_iterations,
        depth,
    )
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

    /// `x` を写した結果が `g`（非負）のとき、次に試す状態を `out` に入れる。
    ///
    /// 加速した値が負になるセルがあれば、`g` からの外挿の幅を負にならない所まで縮める。0 で切り捨てると
    /// 加速の履歴と実際の状態が食い違い、遅いモードの収束が止まる。
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
        let mut step = 1.0_f64;
        for (value, base) in out.iter().zip(g) {
            if *value < 0.0 {
                // base ≥ 0 > value なので分母は正
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
            inner: Buffers {
                collision: vec![0.0; self.mesh.n_cells],
                energy_sum: vec![0.0; self.mesh.n_eps],
                reinject: vec![0.0; self.mesh.n_eps],
                source: vec![0.0; self.mesh.n_cells],
                diagonal: vec![0.0; self.mesh.n_eps],
                advection_high: vec![0.0; self.mesh.n_cells],
                advection_upwind: vec![0.0; self.mesh.n_cells],
                edge_flux: vec![0.0; edges],
            },
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
