//! 遮蔽Rutherford型の角度分布と、θ格子上の再分配核。
//!
//! 角度分布は Okhrimovskyy, Bogaerts & Gijbels, Phys. Rev. E 65, 037402 (2002) の
//! `I(χ) = (1 − ξ²) / (4π (1 − ξ cos χ)²)` を使う。パラメータ ξ は運動量移行断面積と
//! 積分断面積の比 `σ_m/σ = 1 − <cos χ>` から決める。
//!
//! 入射方向の極角余弦を μ、散乱後を μ' とすると、方位角で積分した累積分布は
//! `P(μ' < c | μ) = 1/2 + (c − ξμ) / (2 S(μ, c))`、`S = sqrt(ξ²(μ² + c²) − 2ξμc + 1 − ξ²)`
//! となる。さらに μ で積分すると `∫ P dμ = μ/2 − S/(2ξ)` なので、ビン間の遷移確率は
//! Sの2重差分で厳密に求まる。

/// ξ の上限。これより前方（後方）に偏った分布は ξ = ±XI_MAX で近似する。
pub const XI_MAX: f64 = 1.0 - 1.0e-9;
/// |ξ| がこれ未満なら等方として扱う。
pub const XI_ISOTROPIC: f64 = 1.0e-6;
/// 再分配核を表にする変数 u = ln((1+ξ)/(1−ξ)) の刻み。
const NODE_SPACING: f64 = 0.1;

/// 遮蔽Rutherford分布の `σ_m/σ = (1−ξ)/(2ξ²)[(1+ξ)ln((1+ξ)/(1−ξ)) − 2ξ]`。
pub fn ratio_from_xi(xi: f64) -> f64 {
    if xi.abs() < 0.1 {
        // 1 − 2 Σ ξ^(2k+1) / ((2k+1)(2k+3))
        let x2 = xi * xi;
        let mut power = xi;
        let mut sum = 0.0;
        for k in 0..20 {
            sum += power / (((2 * k + 1) * (2 * k + 3)) as f64);
            power *= x2;
        }
        1.0 - 2.0 * sum
    } else {
        (1.0 - xi) / (2.0 * xi * xi) * ((1.0 + xi) * ((1.0 + xi) / (1.0 - xi)).ln() - 2.0 * xi)
    }
}

/// `σ_m/σ`（0から2）から ξ を求める。比は ξ について単調減少。
pub fn xi_from_ratio(ratio: f64) -> f64 {
    if !ratio.is_finite() {
        return 0.0;
    }
    if ratio <= ratio_from_xi(XI_MAX) {
        return XI_MAX;
    }
    if ratio >= ratio_from_xi(-XI_MAX) {
        return -XI_MAX;
    }
    let (mut lo, mut hi) = (-XI_MAX, XI_MAX);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if ratio_from_xi(mid) > ratio {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1.0e-15 {
            break;
        }
    }
    0.5 * (lo + hi)
}

fn xi_to_u(xi: f64) -> f64 {
    ((1.0 + xi) / (1.0 - xi)).ln()
}

/// ξ = tanh(u/2)
fn u_to_xi(u: f64) -> f64 {
    (0.5 * u).tanh()
}

/// θ ビン j から j' への遷移確率（行ごとに和が1）。`theta_b` は 0 から π の境界。
pub fn kernel(xi: f64, theta_b: &[f64]) -> Vec<f64> {
    let n = theta_b.len() - 1;
    let mu: Vec<f64> = theta_b.iter().map(|theta| theta.cos()).collect();
    let mut matrix = vec![0.0; n * n];
    if xi.abs() < XI_ISOTROPIC {
        for j in 0..n {
            for jp in 0..n {
                matrix[j * n + jp] = 0.5 * (mu[jp] - mu[jp + 1]);
            }
        }
        return matrix;
    }
    let xi2 = xi * xi;
    let s = |a: f64, c: f64| {
        (xi2 * (a * a + c * c) - 2.0 * xi * a * c + 1.0 - xi2)
            .max(0.0)
            .sqrt()
    };
    let edge: Vec<Vec<f64>> = mu
        .iter()
        .map(|a| mu.iter().map(|c| s(*a, *c)).collect())
        .collect();
    for j in 0..n {
        let d_mu = mu[j] - mu[j + 1];
        let mut row_sum = 0.0;
        for jp in 0..n {
            let double_difference =
                edge[j][jp] - edge[j + 1][jp] - edge[j][jp + 1] + edge[j + 1][jp + 1];
            let value = (-double_difference / (2.0 * xi * d_mu)).max(0.0);
            matrix[j * n + jp] = value;
            row_sum += value;
        }
        // 丸め誤差を除いて厳密に保存させる
        for jp in 0..n {
            matrix[j * n + jp] /= row_sum;
        }
    }
    matrix
}

/// u = ln((1+ξ)/(1−ξ)) の等間隔点で再分配核を表にしたもの。点の間は線形補間する
/// （確率行列の凸結合なので、非負性と保存性が保たれる）。
#[derive(Clone, Debug)]
pub struct AngularBank {
    pub n_theta: usize,
    u_start: f64,
    kernels: Vec<Vec<f64>>,
}

impl AngularBank {
    /// `xi_min`から`xi_max`までを覆う表を作る。
    pub fn new(theta_b: &[f64], xi_min: f64, xi_max: f64) -> Self {
        let n_theta = theta_b.len() - 1;
        let u_min = xi_to_u(xi_min.clamp(-XI_MAX, XI_MAX));
        let u_max = xi_to_u(xi_max.clamp(-XI_MAX, XI_MAX));
        let first = (u_min / NODE_SPACING).floor() as i64;
        let last = (u_max / NODE_SPACING).ceil() as i64;
        let kernels = (first..=last.max(first + 1))
            .map(|k| kernel(u_to_xi(k as f64 * NODE_SPACING), theta_b))
            .collect();
        Self {
            n_theta,
            u_start: first as f64 * NODE_SPACING,
            kernels,
        }
    }

    /// ξ に対応する表の区間と、上側の点の重み。
    pub fn locate(&self, xi: f64) -> (usize, f64) {
        let u = xi_to_u(xi.clamp(-XI_MAX, XI_MAX));
        let position =
            ((u - self.u_start) / NODE_SPACING).clamp(0.0, (self.kernels.len() - 1) as f64);
        let lower = (position.floor() as usize).min(self.kernels.len() - 2);
        (lower, position - lower as f64)
    }

    pub fn node(&self, index: usize) -> &[f64] {
        &self.kernels[index]
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    fn theta_edges(n: usize) -> Vec<f64> {
        (0..=n).map(|j| j as f64 * PI / n as f64).collect()
    }

    #[test]
    fn ratio_matches_limits_and_inverts() {
        assert!((ratio_from_xi(0.0) - 1.0).abs() < 1e-15);
        assert!((ratio_from_xi(0.0999) - ratio_from_xi(0.1001)).abs() < 1e-3);
        assert!(ratio_from_xi(0.999) < 0.01);
        assert!(ratio_from_xi(-0.999) > 1.99);
        for ratio in [1e-4, 0.01, 0.3, 0.9, 1.0, 1.4, 1.99] {
            let xi = xi_from_ratio(ratio);
            assert!(
                (ratio_from_xi(xi) - ratio).abs() < 1e-10 * ratio.max(1e-3),
                "{ratio}"
            );
        }
    }

    #[test]
    fn kernel_is_stochastic_and_reduces_to_limits() {
        let theta_b = theta_edges(16);
        let isotropic = kernel(0.0, &theta_b);
        let almost_isotropic = kernel(2.0e-6, &theta_b);
        for (a, b) in isotropic.iter().zip(&almost_isotropic) {
            assert!((a - b).abs() < 1e-5);
        }
        let forward = kernel(XI_MAX, &theta_b);
        for j in 0..16 {
            assert!((forward[j * 16 + j] - 1.0).abs() < 1e-3);
        }
        for xi in [-0.9, -0.3, 0.2, 0.7, 0.99] {
            let matrix = kernel(xi, &theta_b);
            assert!(matrix.iter().all(|value| *value >= 0.0));
            for j in 0..16 {
                let sum: f64 = matrix[j * 16..(j + 1) * 16].iter().sum();
                assert!((sum - 1.0).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn kernel_reproduces_momentum_transfer() {
        // 細かい θ 格子では、等方分布からの <cos χ> が 1 − σ_m/σ に近づく
        let n = 400;
        let theta_b = theta_edges(n);
        let mu: Vec<f64> = theta_b.iter().map(|t| t.cos()).collect();
        let centre: Vec<f64> = (0..n).map(|j| 0.5 * (mu[j] + mu[j + 1])).collect();
        for xi in [0.3, 0.8] {
            let matrix = kernel(xi, &theta_b);
            // 入射方向 μ_j と散乱後の平均の積の平均 = <cos χ> <μ²> = <cos χ>/3
            let mut acc = 0.0;
            for j in 0..n {
                let w = 0.5 * (mu[j] - mu[j + 1]);
                let mean_out: f64 = (0..n).map(|jp| matrix[j * n + jp] * centre[jp]).sum();
                acc += w * centre[j] * mean_out;
            }
            let expected = (1.0 - ratio_from_xi(xi)) / 3.0;
            assert!(
                (acc - expected).abs() < 2e-3,
                "xi={xi}: {acc} vs {expected}"
            );
        }
    }

    #[test]
    fn bank_interpolates_between_nodes() {
        let theta_b = theta_edges(8);
        let bank = AngularBank::new(&theta_b, -0.5, 0.99);
        let (lower, weight) = bank.locate(0.4);
        assert!((0.0..=1.0).contains(&weight));
        assert!(lower + 1 < 1000);
        let (_, w_low) = bank.locate(-0.9);
        assert_eq!(w_low, 0.0);
    }
}
