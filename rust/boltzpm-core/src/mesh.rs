use std::f64::consts::PI;

use crate::constants::speed_from_ev;

#[derive(Clone, Debug)]
pub struct VelocityMesh {
    pub eps_max_ev: f64,
    pub d_eps_ev: f64,
    pub n_eps: usize,
    pub n_theta: usize,
    pub n_cells: usize,
    pub d_theta: f64,
    pub eps_b: Vec<f64>,
    pub eps_c: Vec<f64>,
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
    pub fn new(eps_max_ev: f64, d_eps_ev: f64, n_theta: usize) -> Result<Self, String> {
        if !eps_max_ev.is_finite() || !d_eps_ev.is_finite() || eps_max_ev <= 0.0 || d_eps_ev <= 0.0
        {
            return Err("eps_max_eV and d_eps_eV must be finite and positive".into());
        }
        if n_theta == 0 {
            return Err("n_theta must be positive".into());
        }
        let n_eps = (eps_max_ev / d_eps_ev).round() as usize;
        if n_eps == 0 {
            return Err("eps_max_eV / d_eps_eV must be >= 1".into());
        }
        let n_cells = n_eps
            .checked_mul(n_theta)
            .ok_or_else(|| "mesh size overflow".to_string())?;
        let d_theta = PI / n_theta as f64;
        let eps_b: Vec<_> = (0..=n_eps).map(|i| i as f64 * d_eps_ev).collect();
        let eps_c: Vec<_> = (0..n_eps).map(|i| (i as f64 + 0.5) * d_eps_ev).collect();
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
}
