use std::{cmp::Ordering, f64::consts::PI};

use rayon::prelude::*;

use crate::mesh::VelocityMesh;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessKind {
    Elastic,
    Effective,
    Excitation,
    Ionization,
    Attachment,
}

impl ProcessKind {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_uppercase().as_str() {
            "ELASTIC" => Ok(Self::Elastic),
            "EFFECTIVE" => Ok(Self::Effective),
            "EXCITATION" => Ok(Self::Excitation),
            "IONIZATION" => Ok(Self::Ionization),
            "ATTACHMENT" => Ok(Self::Attachment),
            _ => Err(format!("unknown cross-section kind: {value}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProcessSpec {
    pub gas_name: String,
    pub name: String,
    pub kind: ProcessKind,
    pub fraction: f64,
    pub threshold_ev: f64,
    pub mass_ratio: f64,
    pub sigma: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct FluxEdge {
    upstream: usize,
    downstream: usize,
    coeff_upstream: f64,
    coeff_downstream: f64,
}

#[derive(Clone, Copy, Debug)]
struct SignedEdge {
    edge: usize,
    sign: f64,
}

#[derive(Clone, Debug)]
pub struct AdvectionOperator {
    edges: Vec<FluxEdge>,
    cell_edges: Vec<Vec<SignedEdge>>,
    n_cells: usize,
}

impl AdvectionOperator {
    pub fn new(mesh: &VelocityMesh, xi: f64, sign: i8) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&xi) {
            return Err(format!("xi must be in [0, 1], got {xi}"));
        }
        if sign != 1 && sign != -1 {
            return Err("sign must be +1 or -1".into());
        }
        let mut edges = Vec::with_capacity(
            (mesh.n_eps.saturating_sub(1) * mesh.n_theta)
                + (mesh.n_eps * mesh.n_theta.saturating_sub(1)),
        );

        let map = |k: usize| {
            if sign == 1 { k } else { mesh.mirror_idx(k) }
        };
        let mut add_edge = |u: usize, d: usize, area: f64| {
            let u_mapped = map(u);
            let d_mapped = map(d);
            let volume_u = mesh.volume[u];
            let volume_d = mesh.volume[d];
            let denom = xi * volume_u + volume_d;
            edges.push(FluxEdge {
                upstream: u_mapped,
                downstream: d_mapped,
                coeff_upstream: area * volume_d / (volume_u * denom),
                coeff_downstream: area * xi * volume_u / (volume_d * denom),
            });
        };

        for i in 0..mesh.n_eps.saturating_sub(1) {
            for j in 0..mesh.n_theta {
                let lower = mesh.idx(i, j);
                let upper = mesh.idx(i + 1, j);
                let (upstream, downstream) = if mesh.theta_c[j] < PI / 2.0 {
                    (lower, upper)
                } else {
                    (upper, lower)
                };
                add_edge(upstream, downstream, mesh.s_plus_eps[lower]);
            }
        }
        for i in 0..mesh.n_eps {
            for j in 0..mesh.n_theta.saturating_sub(1) {
                let downstream = mesh.idx(i, j);
                let upstream = mesh.idx(i, j + 1);
                add_edge(upstream, downstream, mesh.s_plus_theta[downstream]);
            }
        }
        let mut cell_edges = vec![Vec::with_capacity(4); mesh.n_cells];
        for (index, edge) in edges.iter().enumerate() {
            cell_edges[edge.upstream].push(SignedEdge {
                edge: index,
                sign: -1.0,
            });
            cell_edges[edge.downstream].push(SignedEdge {
                edge: index,
                sign: 1.0,
            });
        }
        Ok(Self {
            edges,
            cell_edges,
            n_cells: mesh.n_cells,
        })
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn apply(&self, state: &[f64], output: &mut [f64], edge_flux: &mut [f64], parallel: bool) {
        assert_eq!(state.len(), self.n_cells);
        assert_eq!(output.len(), self.n_cells);
        assert!(edge_flux.len() >= self.edges.len());
        let flux = &mut edge_flux[..self.edges.len()];
        if parallel {
            flux.par_iter_mut()
                .zip(self.edges.par_iter())
                .for_each(|(value, edge)| {
                    *value = edge.coeff_upstream * state[edge.upstream]
                        + edge.coeff_downstream * state[edge.downstream];
                });
            output
                .par_iter_mut()
                .zip(self.cell_edges.par_iter())
                .for_each(|(value, adjacent)| {
                    *value = adjacent
                        .iter()
                        .map(|item| item.sign * flux[item.edge])
                        .sum();
                });
        } else {
            output.fill(0.0);
            for edge in &self.edges {
                let value = edge.coeff_upstream * state[edge.upstream]
                    + edge.coeff_downstream * state[edge.downstream];
                output[edge.upstream] -= value;
                output[edge.downstream] += value;
            }
        }
    }

    pub fn column_sums(&self) -> Vec<f64> {
        let mut sums = vec![0.0; self.n_cells];
        for edge in &self.edges {
            sums[edge.upstream] -= edge.coeff_upstream;
            sums[edge.upstream] += edge.coeff_upstream;
            sums[edge.downstream] -= edge.coeff_downstream;
            sums[edge.downstream] += edge.coeff_downstream;
        }
        sums
    }
}

#[derive(Clone, Copy, Debug)]
struct Deposit {
    target: usize,
    source: usize,
    coefficient: f64,
}

#[derive(Clone, Debug)]
pub struct CollisionOperator {
    pub nu_total: Vec<f64>,
    deposits: Vec<Deposit>,
    pub processes: Vec<ProcessSpec>,
    n_eps: usize,
    n_theta: usize,
    w_theta: Vec<f64>,
}

impl CollisionOperator {
    pub fn new(
        mesh: &VelocityMesh,
        number_density: f64,
        processes: Vec<ProcessSpec>,
    ) -> Result<Self, String> {
        let mut nu_total = vec![0.0; mesh.n_eps];
        let mut deposits = Vec::new();
        for process in &processes {
            if process.sigma.len() != mesh.n_eps {
                return Err(format!(
                    "sigma for {} has length {}, expected {}",
                    process.name,
                    process.sigma.len(),
                    mesh.n_eps
                ));
            }
            for (i, total_frequency) in nu_total.iter_mut().enumerate() {
                let nu = process.fraction * number_density * process.sigma[i] * mesh.v_c[i];
                if nu.partial_cmp(&0.0) != Some(Ordering::Greater) {
                    continue;
                }
                *total_frequency += nu;
                let (energy, multiplier) = match process.kind {
                    ProcessKind::Elastic | ProcessKind::Effective => {
                        (mesh.eps_c[i] * (1.0 - 2.0 * process.mass_ratio), 1.0)
                    }
                    ProcessKind::Excitation => (mesh.eps_c[i] - process.threshold_ev, 1.0),
                    ProcessKind::Ionization => ((mesh.eps_c[i] - process.threshold_ev) / 2.0, 2.0),
                    ProcessKind::Attachment => continue,
                };
                let (lo, hi, w_lo, w_hi) = deposit_targets(&mesh.eps_c, energy);
                deposits.push(Deposit {
                    target: lo,
                    source: i,
                    coefficient: multiplier * nu * w_lo,
                });
                deposits.push(Deposit {
                    target: hi,
                    source: i,
                    coefficient: multiplier * nu * w_hi,
                });
            }
        }
        Ok(Self {
            nu_total,
            deposits,
            processes,
            n_eps: mesh.n_eps,
            n_theta: mesh.n_theta,
            w_theta: mesh.w_theta.clone(),
        })
    }

    pub fn apply(
        &self,
        state: &[f64],
        output: &mut [f64],
        energy_sum: &mut [f64],
        reinject: &mut [f64],
        parallel: bool,
    ) {
        assert_eq!(state.len(), self.n_eps * self.n_theta);
        assert_eq!(output.len(), state.len());
        assert_eq!(energy_sum.len(), self.n_eps);
        assert_eq!(reinject.len(), self.n_eps);
        energy_sum.fill(0.0);
        reinject.fill(0.0);
        if parallel {
            energy_sum
                .par_iter_mut()
                .zip(state.par_chunks(self.n_theta))
                .for_each(|(sum, row)| *sum = row.iter().sum());
        } else {
            for i in 0..self.n_eps {
                let row = &state[i * self.n_theta..(i + 1) * self.n_theta];
                energy_sum[i] = row.iter().sum();
            }
        }
        for deposit in &self.deposits {
            reinject[deposit.target] += deposit.coefficient * energy_sum[deposit.source];
        }
        if parallel {
            output
                .par_chunks_mut(self.n_theta)
                .zip(state.par_chunks(self.n_theta))
                .enumerate()
                .for_each(|(i, (output_row, state_row))| {
                    for j in 0..self.n_theta {
                        output_row[j] =
                            -self.nu_total[i] * state_row[j] + reinject[i] * self.w_theta[j];
                    }
                });
        } else {
            for (i, reinjection) in reinject.iter().copied().enumerate() {
                for j in 0..self.n_theta {
                    let k = i * self.n_theta + j;
                    output[k] = -self.nu_total[i] * state[k] + reinjection * self.w_theta[j];
                }
            }
        }
    }
}

fn deposit_targets(centers: &[f64], energy: f64) -> (usize, usize, f64, f64) {
    let n = centers.len();
    if n == 1 || energy <= centers[0] {
        return (0, 0, 1.0, 0.0);
    }
    if energy >= centers[n - 1] {
        // Python版は上端でlo重み0、hi重み1となるが、両添字は最終セルになる。
        return (n - 1, n - 1, 0.0, 1.0);
    }
    let upper = centers.partition_point(|value| *value <= energy);
    let lower = upper - 1;
    let w_lower = ((centers[upper] - energy) / (centers[upper] - centers[lower])).clamp(0.0, 1.0);
    (lower, upper, w_lower, 1.0 - w_lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advection_conserves_sum() {
        let mesh = VelocityMesh::new(8.0, 0.2, 16).unwrap();
        let state: Vec<_> = (0..mesh.n_cells).map(|i| (i + 1) as f64).collect();
        for xi in [0.0, 0.5, 1.0] {
            for sign in [-1, 1] {
                let op = AdvectionOperator::new(&mesh, xi, sign).unwrap();
                let mut output = vec![0.0; mesh.n_cells];
                let mut edge_flux = vec![0.0; op.edge_count()];
                op.apply(&state, &mut output, &mut edge_flux, false);
                let scale = output.iter().map(|x| x.abs()).sum::<f64>();
                assert!(output.iter().sum::<f64>().abs() < 1.0e-14 * scale);
            }
        }
    }

    #[test]
    fn parallel_advection_matches_sequential() {
        let mesh = VelocityMesh::new(8.0, 0.02, 48).unwrap();
        let state: Vec<_> = (0..mesh.n_cells)
            .map(|i| ((i * 17 + 3) % 101) as f64 / 101.0)
            .collect();
        let op = AdvectionOperator::new(&mesh, 0.5, 1).unwrap();
        let mut sequential = vec![0.0; mesh.n_cells];
        let mut parallel = vec![0.0; mesh.n_cells];
        let mut edge_flux = vec![0.0; op.edge_count()];
        op.apply(&state, &mut sequential, &mut edge_flux, false);
        op.apply(&state, &mut parallel, &mut edge_flux, true);
        for (a, b) in sequential.iter().zip(parallel) {
            assert!((a - b).abs() <= 1.0e-12 * a.abs().max(1.0));
        }
    }
}
