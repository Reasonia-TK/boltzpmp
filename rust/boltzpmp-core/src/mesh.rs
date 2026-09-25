use std::f64::consts::PI;

use crate::constants::speed_from_ev;

#[derive(Clone, Debug)]
pub struct VelocityMesh {
    pub eps_max_ev: f64,
    /// 一様格子の刻み。非一様格子では`None`。
    pub d_eps_ev: Option<f64>,
    pub n_eps: usize,
    pub n_theta: usize,
    pub n_cells: usize,
    pub d_theta: f64,
    pub eps_b: Vec<f64>,
    pub eps_c: Vec<f64>,
    /// 各エネルギーセルの幅。
    pub d_eps: Vec<f64>,
    pub v_b: Vec<f64>,
    pub v_c: Vec<f64>,
    pub theta_b: Vec<f64>,
    pub theta_c: Vec<f64>,
    pub volume: Vec<f64>,
    pub s_plus_eps: Vec<f64>,
    pub s_minus_eps: Vec<f64>,
    pub s_plus_theta: Vec<f64>,
    pub s_minus_theta: Vec<f64>,
    pub w_theta: Vec<f64>,
}

impl VelocityMesh {
    /// 一定エネルギー幅の格子。
    pub fn new(eps_max_ev: f64, d_eps_ev: f64, n_theta: usize) -> Result<Self, String> {
        if !eps_max_ev.is_finite() || !d_eps_ev.is_finite() || eps_max_ev <= 0.0 || d_eps_ev <= 0.0
        {
            return Err("eps_max_eV and d_eps_eV must be finite and positive".into());
        }
        let n_eps = (eps_max_ev / d_eps_ev).round() as usize;
        if n_eps == 0 {
            return Err("eps_max_eV / d_eps_eV must be >= 1".into());
        }
        let eps_b: Vec<_> = (0..=n_eps).map(|i| i as f64 * d_eps_ev).collect();
        let eps_c: Vec<_> = (0..n_eps).map(|i| (i as f64 + 0.5) * d_eps_ev).collect();
        let d_eps = vec![d_eps_ev; n_eps];
        Self::build(eps_max_ev, Some(d_eps_ev), eps_b, eps_c, d_eps, n_theta)
    }

    /// 任意のエネルギー境界（0から始まる狭義単調増加列）による格子。
    pub fn from_edges(edges_ev: &[f64], n_theta: usize) -> Result<Self, String> {
        if edges_ev.len() < 2 {
            return Err("energy grid needs at least two edges".into());
        }
        if edges_ev[0] != 0.0 {
            return Err(format!(
                "energy grid must start at 0 eV, got {}",
                edges_ev[0]
            ));
        }
        if edges_ev.iter().any(|value| !value.is_finite()) {
            return Err("energy grid edges must be finite".into());
        }
        if let Some(index) = edges_ev.windows(2).position(|pair| pair[1] <= pair[0]) {
            return Err(format!(
                "energy grid edges must be strictly increasing ({} after {})",
                edges_ev[index + 1],
                edges_ev[index]
            ));
        }
        let n_eps = edges_ev.len() - 1;
        let eps_b = edges_ev.to_vec();
        let eps_c: Vec<_> = (0..n_eps)
            .map(|i| 0.5 * (eps_b[i] + eps_b[i + 1]))
            .collect();
        let d_eps: Vec<_> = (0..n_eps).map(|i| eps_b[i + 1] - eps_b[i]).collect();
        let eps_max = eps_b[n_eps];
        Self::build(eps_max, None, eps_b, eps_c, d_eps, n_theta)
    }

    fn build(
        eps_max_ev: f64,
        d_eps_ev: Option<f64>,
        eps_b: Vec<f64>,
        eps_c: Vec<f64>,
        d_eps: Vec<f64>,
        n_theta: usize,
    ) -> Result<Self, String> {
        if n_theta == 0 {
            return Err("n_theta must be positive".into());
        }
        let n_eps = eps_c.len();
        let n_cells = n_eps
            .checked_mul(n_theta)
            .ok_or_else(|| "mesh size overflow".to_string())?;
        let d_theta = PI / n_theta as f64;
        let v_b: Vec<_> = eps_b.iter().copied().map(speed_from_ev).collect();
        let v_c: Vec<_> = eps_c.iter().copied().map(speed_from_ev).collect();
        let theta_b: Vec<_> = (0..=n_theta).map(|j| j as f64 * d_theta).collect();
        let theta_c: Vec<_> = (0..n_theta).map(|j| (j as f64 + 0.5) * d_theta).collect();

        let mut volume = vec![0.0; n_cells];
        let mut s_plus_eps = vec![0.0; n_cells];
        let mut s_minus_eps = vec![0.0; n_cells];
        let mut s_plus_theta = vec![0.0; n_cells];
        let mut s_minus_theta = vec![0.0; n_cells];
        let mut w_theta = vec![0.0; n_theta];

        for j in 0..n_theta {
            let cos_lo = theta_b[j].cos();
            let cos_hi = theta_b[j + 1].cos();
            let dcos = cos_lo - cos_hi;
            w_theta[j] = dcos / 2.0;

            let sin2_lo = theta_b[j].sin().powi(2);
            let sin2_hi = theta_b[j + 1].sin().powi(2);
            let mut max_sin2 = sin2_lo.max(sin2_hi);
            if theta_b[j] <= PI / 2.0 && theta_b[j + 1] >= PI / 2.0 {
                max_sin2 = 1.0;
            }
            let sin2_diff = max_sin2 - sin2_lo.min(sin2_hi);

            for i in 0..n_eps {
                let k = i * n_theta + j;
                let dv3 = v_b[i + 1].powi(3) - v_b[i].powi(3);
                let dv2 = v_b[i + 1].powi(2) - v_b[i].powi(2);
                volume[k] = (2.0 / 3.0) * PI * dv3 * dcos;
                s_plus_eps[k] = PI * v_b[i + 1].powi(2) * sin2_diff;
                s_minus_eps[k] = PI * v_b[i].powi(2) * sin2_diff;
                s_plus_theta[k] = PI * dv2 * sin2_hi;
                s_minus_theta[k] = PI * dv2 * sin2_lo;
            }
        }

        Ok(Self {
            eps_max_ev,
            d_eps_ev,
            n_eps,
            n_theta,
            n_cells,
            d_theta,
            eps_b,
            eps_c,
            d_eps,
            v_b,
            v_c,
            theta_b,
            theta_c,
            volume,
            s_plus_eps,
            s_minus_eps,
            s_plus_theta,
            s_minus_theta,
            w_theta,
        })
    }

    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i * self.n_theta + j
    }

    #[inline]
    pub fn mirror_idx(&self, k: usize) -> usize {
        let i = k / self.n_theta;
        let j = k % self.n_theta;
        self.idx(i, self.n_theta - 1 - j)
    }
}

/// 低エネルギー側を細かくした境界列。
///
/// `eps_uniform_ev`までは`d_eps_min_ev`の一様刻みとし、その上は刻みを`sqrt(eps)`に比例して広げる
/// （速度の刻みが一定になり、移流のCFL条件が緩む）。最後の境界は`eps_max_ev`にそろえる。
pub fn graded_edges(
    eps_max_ev: f64,
    d_eps_min_ev: f64,
    eps_uniform_ev: f64,
) -> Result<Vec<f64>, String> {
    for (label, value) in [
        ("eps_max_eV", eps_max_ev),
        ("d_eps_min_eV", d_eps_min_ev),
        ("eps_uniform_eV", eps_uniform_ev),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{label} must be finite and positive, got {value}"));
        }
    }
    if d_eps_min_ev >= eps_max_ev {
        return Err("d_eps_min_eV must be smaller than eps_max_eV".into());
    }
    let mut edges = vec![0.0];
    let mut energy: f64 = 0.0;
    while energy < eps_max_ev {
        let width = if energy < eps_uniform_ev {
            d_eps_min_ev
        } else {
            d_eps_min_ev * (energy / eps_uniform_ev).sqrt()
        };
        energy += width;
        edges.push(energy);
    }
    // 最後のセルが半端に細くならないよう、はみ出しが半セルを超えたら1つ手前の境界を上端にする
    let n = edges.len();
    if n > 2 && edges[n - 1] - eps_max_ev > 0.5 * (edges[n - 1] - edges[n - 2]) {
        edges.pop();
    }
    let last = edges.len() - 1;
    edges[last] = eps_max_ev;
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theta_weights_sum_to_one() {
        let mesh = VelocityMesh::new(8.0, 0.2, 16).unwrap();
        assert!((mesh.w_theta.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn mirror_is_an_involution() {
        let mesh = VelocityMesh::new(8.0, 0.2, 16).unwrap();
        for k in 0..mesh.n_cells {
            assert_eq!(mesh.mirror_idx(mesh.mirror_idx(k)), k);
        }
    }

    #[test]
    fn edges_reproduce_uniform_mesh() {
        let uniform = VelocityMesh::new(8.0, 0.25, 12).unwrap();
        let graded = VelocityMesh::from_edges(&uniform.eps_b, 12).unwrap();
        for (a, b) in uniform.volume.iter().zip(&graded.volume) {
            assert!((a - b).abs() <= 1e-15 * a.abs());
        }
        for (a, b) in uniform.eps_c.iter().zip(&graded.eps_c) {
            assert!((a - b).abs() <= 1e-15 * a.abs());
        }
    }

    #[test]
    fn graded_edges_are_increasing_and_end_at_max() {
        let edges = graded_edges(60.0, 0.0025, 0.5).unwrap();
        assert_eq!(edges[0], 0.0);
        assert_eq!(*edges.last().unwrap(), 60.0);
        assert!(edges.windows(2).all(|pair| pair[1] > pair[0]));
        assert!((edges[1] - 0.0025).abs() < 1e-15);
        // 一様刻み2.5 meVなら24000セルになる
        assert!(edges.len() < 5000);
        assert!(VelocityMesh::from_edges(&[0.1, 1.0], 4).is_err());
        assert!(VelocityMesh::from_edges(&[0.0, 1.0, 1.0], 4).is_err());
    }
}
