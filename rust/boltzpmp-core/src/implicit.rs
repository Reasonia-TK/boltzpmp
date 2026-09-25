//! DC定常解の陰解法。
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
        })
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
        self.sweep
            .solve(self.acceleration, &work.diagonal, &work.source, next)?;
        // 欠損補正で生じうる小さな負の値を除き、和を1にする
        clip_and_normalize(next)
    }
}

fn clip_and_normalize(values: &mut [f64]) -> Result<(), String> {
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
            return Ok(());
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
            return Ok(());
        };
        out.copy_from_slice(g);
        for (i, coefficient) in gamma.iter().enumerate() {
            for ((value, newer), older) in out.iter_mut().zip(&self.gs[i + 1]).zip(&self.gs[i]) {
                *value -= coefficient * (newer - older);
            }
        }
        clip_and_normalize(out)
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
    let mut x = initial.to_vec();
    clip_and_normalize(&mut x)?;
    let mut g = vec![0.0; x.len()];
    let mut next = vec![0.0; x.len()];
    let mut anderson = Anderson::new(depth);
    let mut best = f64::INFINITY;
    for iteration in 1..=max_iterations {
        problem.map(&x, &mut g, &mut work)?;
        let residual: f64 = g.iter().zip(&x).map(|(a, b)| (a - b).abs()).sum();
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
    problem.map(&x, &mut g, &mut work)?;
    Ok(ImplicitOutcome {
        state: g,
        converged: false,
        iterations: max_iterations,
    })
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
