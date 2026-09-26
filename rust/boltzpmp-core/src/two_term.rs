//! エネルギーだけの二項近似の演算子と、その直接解法（陰解法の遅いモードの加速に使う）。
//!
//! 等方な分布 u（エネルギーセル i の電子の割合）に対する定常の演算子を、流出 − 流入の向きで
//!
//! ```text
//! (L u)_i = Γ_{i+1/2} − Γ_{i−1/2} + ν_i u_i − Σ_j ν_{j→i} u_j − a (N u)_i
//! Γ_b = (4π/3) a² v_b² ⟨1/ν̃⟩_b (f_i − f_{i+1}) / (v_{i+1} − v_i),   f_i = u_i / V_i
//! ```
//!
//! とする。
//!
//! - Γ は電場によるエネルギー方向の拡散（二項近似）。ν̃ は運動量移行の周波数に、エネルギーを変えない
//!   等方散乱を足したもの。面の値は両側のセルの 1/ν̃ の平均。V_i は速度空間の体積。
//! - ν_{j→i} は衝突による再注入（`CollisionOperator::energy_couplings`）、ν_i は全衝突周波数。
//!   電離は1回の衝突で電子1個が出るものとして数え（増える電子は入れない）、増加率 λ も入れない。
//!   こうすると L は非対角が負で列の和が非負の M 行列になり、ピボット選択なしで安定に消去できる。
//!   電子数の増減は合成加速の外側の反復（規格化と λ）が受け持つ。
//! - RF の遅いモードの補正（`TwoTermSolver::with_fluxes`）では、面の流束を高次の解に合わせ、衝突は電離で
//!   増える電子と λ を含めてそのまま使う。
//! - N は、移流の離散化が等方な分布に与える数値拡散（風上差分と固定の ξ）。制限関数スキームは滑らかな
//!   ところで中心差分に近いので 0 とする。
//!
//! 行列は帯行列になる。最低エネルギーのセルの行を和の条件 Σu に置き換え、高エネルギー側から
//! ピボット選択なしの LU 分解で消去する（`TwoTermSolver::from_entries`）。

use std::f64::consts::PI;

use crate::{AdvectionOperator, AdvectionScheme, CollisionOperator, VelocityMesh};

/// 帯の要素数の上限。これを超える格子では加速を使わない。
const MAX_BAND_ELEMENTS: usize = 40_000_000;
/// 対角に比べてこれより小さい結合は、帯の幅を決めるときに捨てる。
const DROP_TOLERANCE: f64 = 1.0e-12;

/// 演算子の係数。
pub(crate) struct TwoTermParameters<'a> {
    /// 電場による加速度の大きさ（m/s²）。
    pub acceleration: f64,
    pub scheme: AdvectionScheme,
    /// エネルギーを変えない等方散乱の周波数（エネルギーセルごと）。
    pub isotropic: Option<&'a [f64]>,
}

/// LU 分解した二項近似の演算子と、その定常解。
pub(crate) struct TwoTermSolver {
    n: usize,
    lower: usize,
    upper: usize,
    width: usize,
    /// 行 i の列 i − lower ..= i + upper。L の係数（対角より左）と U（対角から右）を重ねて持つ。
    band: Vec<f64>,
    /// 和の条件の行を消去したときの係数と、最後のピボット。
    last_multipliers: Vec<f64>,
    last_pivot: f64,
    /// 置き換える前の最後の行（疎）。
    original_last: Vec<(usize, f64)>,
    /// L u = 0、Σu = 1 の解（負の値は0）。
    steady: Vec<f64>,
    /// 和0の解で使う、定常解の向きの応答とその係数。
    border_response: Vec<f64>,
    border_denominator: f64,
}

impl TwoTermSolver {
    pub fn new(
        mesh: &VelocityMesh,
        collision: &CollisionOperator,
        parameters: &TwoTermParameters<'_>,
    ) -> Result<Self, String> {
        let entries = assemble(mesh, collision, parameters)?;
        Self::from_entries(mesh.n_eps, &entries)
    }

    /// 面の流束を `Γ_b = K_b (f_b − f_{b+1}) + w_b f_{風上}` で与えた演算子（衝突は電離で増える電子を含めて
    /// そのまま、λ は符号付き）。非線形拡散加速で、高次の解から求めた流束と矛盾しない低次の方程式に使う。
    pub fn with_fluxes(
        mesh: &VelocityMesh,
        collision: &CollisionOperator,
        diffusion: &[f64],
        drift: &[f64],
        growth: f64,
    ) -> Result<Self, String> {
        let faces = mesh.n_eps.saturating_sub(1);
        if diffusion.len() != faces || drift.len() != faces {
            return Err(format!(
                "face fluxes need {faces} values, got {} and {}",
                diffusion.len(),
                drift.len()
            ));
        }
        let mut entries = collision_entries(collision, growth, true);
        push_diffusion(&mut entries, mesh, diffusion);
        push_drift(&mut entries, mesh, drift);
        if entries.iter().any(|(_, _, value)| !value.is_finite()) {
            return Err("two-term operator has a non-finite coefficient".into());
        }
        Self::from_entries(mesh.n_eps, &entries)
    }

    /// 要素 `(行, 列, 値)` から作る（重複は足し合わせる）。
    ///
    /// 内部ではエネルギーの高いセルから並べる（内部の番号 = n − 1 − セル）。消去は高エネルギー側から進み、
    /// 和の条件に置き換える最後の行は最低エネルギーのセルになる。低エネルギー側から消去すると、電場が弱く
    /// 上向きの移動がほとんどない条件で、分布のある低エネルギーの範囲がそれだけでほぼ特異になり、
    /// ピボットが丸め誤差に埋もれる。高エネルギー側のセルは下向きへの流出が大きいので、この順なら
    /// ほぼ特異になるのは最後（和の条件の行）だけになる。
    fn from_entries(n: usize, entries: &[(usize, usize, f64)]) -> Result<Self, String> {
        if n == 0 {
            return Err("the energy grid is empty".into());
        }
        let reversed: Vec<(usize, usize, f64)> = entries
            .iter()
            .map(|&(row, column, value)| (n - 1 - row, n - 1 - column, value))
            .collect();
        let mut solver = Self::factorize(n, &reversed)?;
        let mut steady = vec![0.0; n];
        steady[n - 1] = 1.0;
        solver.solve_in_place(&mut steady);
        let mut total = 0.0;
        for value in &mut steady {
            if !value.is_finite() {
                return Err("two-term steady state is not finite".into());
            }
            *value = value.max(0.0);
            total += *value;
        }
        if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err("two-term steady state has no electrons".into());
        }
        for value in &mut steady {
            *value /= total;
        }
        let mut response = steady.clone();
        response[n - 1] = 0.0;
        solver.solve_in_place(&mut response);
        solver.border_denominator = steady[n - 1] - solver.last_row_dot(&response);
        solver.border_response = response;
        steady.reverse();
        solver.steady = steady;
        Ok(solver)
    }

    /// L u = 0、Σu = 1 の解（各エネルギーセルの電子の割合）。
    pub fn steady(&self) -> &[f64] {
        &self.steady
    }

    /// `L δ = b − c u₀`、`Σδ = 0` を解く（u₀ は定常解、c は b のうち和の条件と両立しない部分）。
    pub fn solve_zero_sum(&self, rhs: &[f64], out: &mut [f64]) {
        let n = self.n;
        for (value, source) in out.iter_mut().zip(rhs.iter().rev()) {
            *value = *source;
        }
        let constrained = out[n - 1];
        out[n - 1] = 0.0;
        self.solve_in_place(out);
        let denominator = self.border_denominator;
        let c = if denominator.is_finite() && denominator != 0.0 {
            (constrained - self.last_row_dot(out)) / denominator
        } else {
            0.0
        };
        if c != 0.0 {
            for (value, response) in out.iter_mut().zip(&self.border_response) {
                *value -= c * response;
            }
        }
        out.reverse();
    }

    fn last_row_dot(&self, values: &[f64]) -> f64 {
        self.original_last
            .iter()
            .map(|(column, coefficient)| coefficient * values[*column])
            .sum()
    }

    fn factorize(n: usize, entries: &[(usize, usize, f64)]) -> Result<Self, String> {
        if n == 0 {
            return Err("the energy grid is empty".into());
        }
        let mut diagonal = vec![0.0_f64; n];
        for &(row, column, value) in entries {
            if row == column {
                diagonal[row] += value;
            }
        }
        let significant = |row: usize, column: usize, value: f64| {
            value.abs() > DROP_TOLERANCE * diagonal[row].abs().max(diagonal[column].abs())
        };
        let mut last_row = vec![0.0; n];
        let (mut lower, mut upper) = (0, 0);
        for &(row, column, value) in entries {
            if row + 1 == n {
                last_row[column] += value;
            } else if significant(row, column, value) {
                if row > column {
                    lower = lower.max(row - column);
                } else {
                    upper = upper.max(column - row);
                }
            }
        }
        let width = lower + upper + 1;
        if n.saturating_mul(width) > MAX_BAND_ELEMENTS {
            return Err(format!(
                "two-term band too large ({n} cells, bandwidth {width})"
            ));
        }
        let mut band = vec![0.0; n * width];
        for &(row, column, value) in entries {
            if row + 1 < n && (row == column || significant(row, column, value)) {
                band[row * width + column + lower - row] += value;
            }
        }
        // 最後の行は和の条件（すべて1）にして消去する
        let mut last = vec![1.0; n];
        let mut last_multipliers = vec![0.0; n];
        for k in 0..n - 1 {
            let pivot = band[k * width + lower];
            if !pivot.is_finite() || pivot <= 0.0 {
                return Err(format!(
                    "two-term elimination found a non-positive pivot at cell {k}"
                ));
            }
            let row_end = (k + upper).min(n - 1);
            let pivot_row = k * width + lower - k;
            for i in k + 1..=(k + lower).min(n.saturating_sub(2)) {
                let index = i * width + k + lower - i;
                let factor = band[index] / pivot;
                band[index] = factor;
                if factor != 0.0 {
                    let row = i * width + lower - i;
                    for j in k + 1..=row_end {
                        band[row + j] -= factor * band[pivot_row + j];
                    }
                }
            }
            let factor = last[k] / pivot;
            last_multipliers[k] = factor;
            if factor != 0.0 {
                for j in k + 1..=row_end {
                    last[j] -= factor * band[pivot_row + j];
                }
            }
        }
        let last_pivot = last[n - 1];
        if !last_pivot.is_finite() || last_pivot == 0.0 {
            return Err("two-term elimination is singular".into());
        }
        let original_last = last_row
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0)
            .map(|(column, value)| (column, *value))
            .collect();
        Ok(Self {
            n,
            lower,
            upper,
            width,
            band,
            last_multipliers,
            last_pivot,
            original_last,
            steady: Vec::new(),
            border_response: Vec::new(),
            border_denominator: f64::NAN,
        })
    }

    /// 最後の行を和の条件に置き換えた行列 M について、M x = b を解く（b を x で上書き）。
    fn solve_in_place(&self, x: &mut [f64]) {
        let (n, width, lower) = (self.n, self.width, self.lower);
        for i in 1..n.saturating_sub(1) {
            let row = i * width + lower - i;
            let first = i.saturating_sub(lower);
            let value = x[i]
                - self.band[row + first..row + i]
                    .iter()
                    .zip(&x[first..i])
                    .map(|(coefficient, known)| coefficient * known)
                    .sum::<f64>();
            x[i] = value;
        }
        if n > 1 {
            let mut value = x[n - 1];
            for (multiplier, previous) in self.last_multipliers[..n - 1].iter().zip(&x[..n - 1]) {
                value -= multiplier * previous;
            }
            x[n - 1] = value;
        }
        x[n - 1] /= self.last_pivot;
        for i in (0..n - 1).rev() {
            let row = i * width + lower - i;
            let end = (i + self.upper).min(n - 1);
            let value = x[i]
                - self.band[row + i + 1..=row + end]
                    .iter()
                    .zip(&x[i + 1..=end])
                    .map(|(coefficient, known)| coefficient * known)
                    .sum::<f64>();
            x[i] = value / self.band[row + i];
        }
    }
}

/// 演算子の要素 `(行, 列, 値)`（重複は足し合わせる）。
fn assemble(
    mesh: &VelocityMesh,
    collision: &CollisionOperator,
    parameters: &TwoTermParameters<'_>,
) -> Result<Vec<(usize, usize, f64)>, String> {
    let n = mesh.n_eps;
    let a = parameters.acceleration;
    if !a.is_finite() || a < 0.0 {
        return Err("two-term acceleration must be finite and non-negative".into());
    }
    // 電離で増える電子と増加率は入れない（L を M 行列に保つため。モジュールの説明を参照）
    let mut entries = collision_entries(collision, 0.0, false);
    if a > 0.0 && n > 1 {
        // ν̃ = 運動量移行 ＋ 等方散乱
        let relaxation: Vec<f64> = (0..n)
            .map(|i| {
                collision.nu_momentum[i]
                    + parameters.isotropic.map_or(0.0, |frequency| frequency[i])
            })
            .collect();
        push_diffusion(&mut entries, mesh, &field_diffusion(mesh, a, &relaxation));
        if let AdvectionScheme::Linear(xi) = parameters.scheme
            && xi < 1.0
        {
            for (row, column, value) in numerical_diffusion(mesh, xi)? {
                entries.push((row, column, -a * value));
            }
        }
    }
    if entries.iter().any(|(_, _, value)| !value.is_finite()) {
        return Err("two-term operator has a non-finite coefficient".into());
    }
    Ok(entries)
}

/// 衝突の要素。`exact` なら電離で増える電子も入れる（そうでなければ1回の衝突で1個）。
fn collision_entries(
    collision: &CollisionOperator,
    growth: f64,
    exact: bool,
) -> Vec<(usize, usize, f64)> {
    let mut entries: Vec<(usize, usize, f64)> = collision
        .nu_total
        .iter()
        .enumerate()
        .map(|(i, nu)| (i, i, nu + growth))
        .collect();
    for (target, source, rate, electrons) in collision.energy_couplings() {
        let per_collision = if exact {
            rate
        } else {
            rate / f64::from(electrons)
        };
        entries.push((target, source, -per_collision));
    }
    entries
}

/// 速度空間でのセルの体積 V_i = (4π/3)(v_{i+1/2}³ − v_{i−1/2}³)。
pub(crate) fn cell_volumes(mesh: &VelocityMesh) -> Vec<f64> {
    (0..mesh.n_eps)
        .map(|i| 4.0 / 3.0 * PI * (mesh.v_b[i + 1].powi(3) - mesh.v_b[i].powi(3)))
        .collect()
}

/// 二項近似の電場による拡散の係数 K_b（面 b = セル b と b + 1 の間の上向きの流束 Γ_b = K_b (f_b − f_{b+1})、
/// f = u/V）。`relaxation` は非等方成分の緩和の周波数 ν̃（セルごと）。
pub(crate) fn field_diffusion(
    mesh: &VelocityMesh,
    acceleration: f64,
    relaxation: &[f64],
) -> Vec<f64> {
    let n = mesh.n_eps;
    let floor = 1.0e-12 * relaxation.iter().copied().fold(0.0, f64::max);
    let inverse: Vec<f64> = relaxation
        .iter()
        .map(|value| 1.0 / value.max(floor).max(f64::MIN_POSITIVE))
        .collect();
    (0..n.saturating_sub(1))
        .map(|i| {
            let speed = mesh.v_b[i + 1];
            4.0 / 3.0
                * PI
                * acceleration
                * acceleration
                * speed
                * speed
                * 0.5
                * (inverse[i] + inverse[i + 1])
                / (mesh.v_c[i + 1] - mesh.v_c[i])
        })
        .collect()
}

/// 拡散の流束 Γ_b = K_b (f_b − f_{b+1}) の要素を足す。
fn push_diffusion(entries: &mut Vec<(usize, usize, f64)>, mesh: &VelocityMesh, diffusion: &[f64]) {
    let volume = cell_volumes(mesh);
    for (i, coefficient) in diffusion.iter().enumerate() {
        let (from_lower, from_upper) = (coefficient / volume[i], coefficient / volume[i + 1]);
        entries.push((i, i, from_lower));
        entries.push((i, i + 1, -from_upper));
        entries.push((i + 1, i, -from_lower));
        entries.push((i + 1, i + 1, from_upper));
    }
}

/// 移動の流束 Γ_b = w_b f_{風上}（w_b > 0 なら下のセル、負なら上のセルの値）の要素を足す。
fn push_drift(entries: &mut Vec<(usize, usize, f64)>, mesh: &VelocityMesh, drift: &[f64]) {
    let volume = cell_volumes(mesh);
    for (i, velocity) in drift.iter().copied().enumerate() {
        if velocity > 0.0 {
            let rate = velocity / volume[i];
            entries.push((i, i, rate));
            entries.push((i + 1, i, -rate));
        } else if velocity < 0.0 {
            let rate = -velocity / volume[i + 1];
            entries.push((i + 1, i + 1, rate));
            entries.push((i, i + 1, -rate));
        }
    }
}

/// 移流 A（ξ 固定の線形スキーム、+θ 向き）が等方な分布に与えるエネルギー方向の移動
/// `(行, 列, 値)`。`du_row/dt = a × 値 × u_col`。3つおきのセルに等方な分布を置いて調べる。
fn numerical_diffusion(mesh: &VelocityMesh, xi: f64) -> Result<Vec<(usize, usize, f64)>, String> {
    let operator = AdvectionOperator::new(mesh, xi, 1)?;
    let n_theta = mesh.n_theta;
    let mut state = vec![0.0; mesh.n_cells];
    let mut output = vec![0.0; mesh.n_cells];
    let mut edge_flux = vec![0.0; operator.edge_count()];
    let mut entries = Vec::with_capacity(3 * mesh.n_eps);
    for color in 0..3 {
        state.fill(0.0);
        for i in (color..mesh.n_eps).step_by(3) {
            state[i * n_theta..(i + 1) * n_theta].copy_from_slice(&mesh.w_theta);
        }
        operator.apply(&state, &mut output, &mut edge_flux, false);
        for i in (color..mesh.n_eps).step_by(3) {
            for row in i.saturating_sub(1)..=(i + 1).min(mesh.n_eps - 1) {
                let value: f64 = output[row * n_theta..(row + 1) * n_theta].iter().sum();
                if value != 0.0 {
                    entries.push((row, i, value));
                }
            }
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 列の和が0の三重対角 ＋ 遠くへの結合（電子数を保つ M 行列）。
    fn conservative_entries(n: usize) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::new();
        for i in 0..n - 1 {
            let up = 1.0 + 0.1 * i as f64;
            let down = 2.0 + 0.05 * i as f64;
            entries.push((i, i, up));
            entries.push((i + 1, i, -up));
            entries.push((i + 1, i + 1, down));
            entries.push((i, i + 1, -down));
        }
        // セル j から j − 3 への跳び
        for j in 3..n {
            entries.push((j, j, 0.7));
            entries.push((j - 3, j, -0.7));
        }
        entries
    }

    fn apply(entries: &[(usize, usize, f64)], x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; x.len()];
        for &(row, column, value) in entries {
            y[row] += value * x[column];
        }
        y
    }

    #[test]
    fn bordered_solve_satisfies_all_rows_and_zero_sum() {
        let n = 12;
        let entries = conservative_entries(n);
        let solver = TwoTermSolver::from_entries(n, &entries).unwrap();
        let steady = solver.steady().to_vec();
        // 定常解: すべての行で L u = 0、和は1
        let residual = apply(&entries, &steady);
        assert!(
            residual.iter().all(|value| value.abs() < 1e-12),
            "{residual:?}"
        );
        assert!((steady.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // 和が0でない右辺: L δ = b − c u₀、Σδ = 0
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() + 0.2).collect();
        let mut delta = vec![0.0; n];
        solver.solve_zero_sum(&rhs, &mut delta);
        let scale: f64 = delta.iter().map(|value| value.abs()).sum();
        assert!(delta.iter().sum::<f64>().abs() < 1e-12 * scale);
        let c = rhs.iter().sum::<f64>();
        let lhs = apply(&entries, &delta);
        for i in 0..n {
            assert!((lhs[i] - (rhs[i] - c * steady[i])).abs() < 1e-10, "row {i}");
        }
    }
}
