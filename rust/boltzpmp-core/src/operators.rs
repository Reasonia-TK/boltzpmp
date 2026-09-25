use std::{cmp::Ordering, f64::consts::PI};

use rayon::prelude::*;

use crate::{
    anisotropy::{AngularBank, XI_ISOTROPIC, xi_from_ratio},
    mesh::VelocityMesh,
};

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
    /// 標的の数密度の割合。
    pub fraction: f64,
    /// 負の値は超弾性衝突（エネルギー獲得）。
    pub threshold_ev: f64,
    pub mass_ratio: f64,
    /// セル中心の断面積。衝突頻度、エネルギー損失、速度係数に使う。
    pub sigma: Vec<f64>,
    /// セル中心の運動量移行断面積。`sigma`と異なる（異方散乱の）ときだけ与える。
    pub sigma_mt: Option<Vec<f64>>,
    /// 弾性衝突で気体の熱運動によるエネルギー交換を入れるときの kT (eV)。0 なら冷たい気体。
    pub gas_temperature_ev: f64,
    /// 内側のセル境界での運動量移行断面積（気体温度を入れた弾性衝突で使う）。
    pub sigma_mt_edges: Option<Vec<f64>>,
}

impl ProcessSpec {
    /// 等方散乱・冷たい気体の過程。
    pub fn new(
        gas_name: String,
        name: String,
        kind: ProcessKind,
        fraction: f64,
        threshold_ev: f64,
        mass_ratio: f64,
        sigma: Vec<f64>,
    ) -> Self {
        Self {
            gas_name,
            name,
            kind,
            fraction,
            threshold_ev,
            mass_ratio,
            sigma,
            sigma_mt: None,
            gas_temperature_ev: 0.0,
            sigma_mt_edges: None,
        }
    }

    fn uses_gas_temperature(&self) -> bool {
        matches!(self.kind, ProcessKind::Elastic | ProcessKind::Effective)
            && self.gas_temperature_ev > 0.0
    }
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

/// 異方散乱の再注入。散乱後の角度分布は再分配核 `(1−w) K[node] + w K[node+1]` で与える。
#[derive(Clone, Copy, Debug)]
struct AnisotropicDeposit {
    source: usize,
    node: usize,
    weight: f64,
    targets: [(usize, f64); 2],
}

#[derive(Clone, Debug)]
pub struct CollisionOperator {
    pub nu_total: Vec<f64>,
    deposits: Vec<Deposit>,
    anisotropic: Vec<AnisotropicDeposit>,
    bank: Option<AngularBank>,
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
        let mut xi_rows = Vec::with_capacity(processes.len());
        let (mut xi_min, mut xi_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for process in &processes {
            validate_process(process, mesh)?;
            let row = match (&process.sigma_mt, process.kind) {
                (
                    Some(mt),
                    ProcessKind::Elastic | ProcessKind::Effective | ProcessKind::Excitation,
                ) => {
                    let row: Vec<f64> = (0..mesh.n_eps)
                        .map(|i| {
                            if process.sigma[i] > 0.0 {
                                xi_from_ratio(mt[i] / process.sigma[i])
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let significant: Vec<f64> = row
                        .iter()
                        .copied()
                        .filter(|xi| xi.abs() >= XI_ISOTROPIC)
                        .collect();
                    if significant.is_empty() {
                        None
                    } else {
                        for xi in significant {
                            xi_min = xi_min.min(xi);
                            xi_max = xi_max.max(xi);
                        }
                        Some(row)
                    }
                }
                // 電離（二次電子を含む）と付着は等方として扱う
                _ => None,
            };
            xi_rows.push(row);
        }
        let bank = (xi_min <= xi_max).then(|| AngularBank::new(&mesh.theta_b, xi_min, xi_max));

        let mut nu_total = vec![0.0; mesh.n_eps];
        let mut deposits = Vec::new();
        let mut anisotropic = Vec::new();
        for (process, xi_row) in processes.iter().zip(&xi_rows) {
            let thermal = process.uses_gas_temperature();
            for (i, total_frequency) in nu_total.iter_mut().enumerate() {
                let nu = process.fraction * number_density * process.sigma[i] * mesh.v_c[i];
                if nu.partial_cmp(&0.0) != Some(Ordering::Greater) {
                    continue;
                }
                *total_frequency += nu;
                let (energy, multiplier) = match process.kind {
                    ProcessKind::Elastic | ProcessKind::Effective => {
                        if thermal && in_thermal_range(mesh.eps_c[i], process.gas_temperature_ev) {
                            // エネルギー交換は下のFokker–Planck項で与え、ここでは方向だけを変える
                            (mesh.eps_c[i], 1.0)
                        } else {
                            // 平均のエネルギー損失 (2m/M)(σ_m/σ)ε
                            let loss = process.mass_ratio
                                * match &process.sigma_mt {
                                    Some(mt) => mt[i] / process.sigma[i],
                                    None => 1.0,
                                };
                            (mesh.eps_c[i] * (1.0 - 2.0 * loss), 1.0)
                        }
                    }
                    ProcessKind::Excitation => (mesh.eps_c[i] - process.threshold_ev, 1.0),
                    ProcessKind::Ionization => ((mesh.eps_c[i] - process.threshold_ev) / 2.0, 2.0),
                    ProcessKind::Attachment => continue,
                };
                let (lo, hi, w_lo, w_hi) = deposit_targets(&mesh.eps_c, energy);
                let targets = [(lo, multiplier * nu * w_lo), (hi, multiplier * nu * w_hi)];
                match (xi_row, &bank) {
                    (Some(row), Some(bank)) if row[i].abs() >= XI_ISOTROPIC => {
                        let (node, weight) = bank.locate(row[i]);
                        anisotropic.push(AnisotropicDeposit {
                            source: i,
                            node,
                            weight,
                            targets,
                        });
                    }
                    _ => {
                        for (target, coefficient) in targets {
                            deposits.push(Deposit {
                                target,
                                source: i,
                                coefficient,
                            });
                        }
                    }
                }
            }
            if thermal {
                add_thermal_exchange(process, mesh, number_density, &mut nu_total, &mut deposits);
            }
        }
        anisotropic.sort_by_key(|deposit| deposit.source);
        Ok(Self {
            nu_total,
            deposits,
            anisotropic,
            bank,
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
        if !self.anisotropic.is_empty() {
            self.apply_anisotropic(state, output);
        }
    }

    fn apply_anisotropic(&self, state: &[f64], output: &mut [f64]) {
        let bank = self
            .bank
            .as_ref()
            .expect("anisotropic deposits always come with an angular bank");
        let n = self.n_theta;
        // 同じセルから出る過程の間で、行ベクトルと再分配核の積を使い回す
        let mut cache: Vec<(usize, Vec<f64>)> = Vec::new();
        let mut source = usize::MAX;
        let mut mixed = vec![0.0; n];
        for deposit in &self.anisotropic {
            if deposit.source != source {
                cache.clear();
                source = deposit.source;
            }
            let row = &state[source * n..(source + 1) * n];
            let lower = cached_product(&mut cache, deposit.node, row, bank);
            let upper = cached_product(&mut cache, deposit.node + 1, row, bank);
            let w = deposit.weight;
            for ((value, a), b) in mixed.iter_mut().zip(&cache[lower].1).zip(&cache[upper].1) {
                *value = (1.0 - w) * a + w * b;
            }
            for (target, coefficient) in deposit.targets {
                if coefficient != 0.0 {
                    let out = &mut output[target * n..(target + 1) * n];
                    for (value, angular) in out.iter_mut().zip(&mixed) {
                        *value += coefficient * angular;
                    }
                }
            }
        }
    }
}

fn cached_product(
    cache: &mut Vec<(usize, Vec<f64>)>,
    node: usize,
    row: &[f64],
    bank: &AngularBank,
) -> usize {
    if let Some(position) = cache.iter().position(|(key, _)| *key == node) {
        return position;
    }
    let n = row.len();
    let kernel = bank.node(node);
    let mut product = vec![0.0; n];
    for (j, value) in row.iter().copied().enumerate() {
        if value != 0.0 {
            for (out, weight) in product.iter_mut().zip(&kernel[j * n..(j + 1) * n]) {
                *out += value * weight;
            }
        }
    }
    cache.push((node, product));
    cache.len() - 1
}

fn validate_process(process: &ProcessSpec, mesh: &VelocityMesh) -> Result<(), String> {
    if process.sigma.len() != mesh.n_eps {
        return Err(format!(
            "sigma for {} has length {}, expected {}",
            process.name,
            process.sigma.len(),
            mesh.n_eps
        ));
    }
    if let Some(mt) = &process.sigma_mt
        && mt.len() != mesh.n_eps
    {
        return Err(format!(
            "momentum-transfer sigma for {} has length {}, expected {}",
            process.name,
            mt.len(),
            mesh.n_eps
        ));
    }
    if process.uses_gas_temperature() {
        let expected = mesh.n_eps.saturating_sub(1);
        match &process.sigma_mt_edges {
            Some(edges) if edges.len() == expected => {}
            Some(edges) => {
                return Err(format!(
                    "edge momentum-transfer sigma for {} has length {}, expected {expected}",
                    process.name,
                    edges.len()
                ));
            }
            None => {
                return Err(format!(
                    "{}: elastic energy exchange with a thermal gas needs sigma_mt_edges",
                    process.name
                ));
            }
        }
    }
    Ok(())
}

/// 熱運動する気体との弾性衝突によるエネルギー交換（Fokker–Planck近似）。
///
/// エネルギー密度 n(ε) に対する流束
/// `G = −(2m/M) ν_m [(ε − kT/2) n + ε kT ∂n/∂ε]`
/// を、指数フィッティング（Scharfetter–Gummel型）で離散化する。指数を
/// `z = ln(n_eq(ε_{i+1}) / n_eq(ε_i))`、`n_eq ∝ sqrt(ε) exp(−ε/kT)` に選ぶので、
/// 電場がなければ格子上のMaxwell分布が厳密に定常解になる。
///
/// 拡散の速度はセル幅の2乗に反比例して時間刻みを縮めるので、熱運動の効果が無視できる
/// `THERMAL_EXCHANGE_LIMIT` × kT より上では、冷たい気体のエネルギー損失（跳び）に切り替える。
fn add_thermal_exchange(
    process: &ProcessSpec,
    mesh: &VelocityMesh,
    number_density: f64,
    nu_total: &mut [f64],
    deposits: &mut Vec<Deposit>,
) {
    let kt = process.gas_temperature_ev;
    let edges_mt = process
        .sigma_mt_edges
        .as_ref()
        .expect("validated in CollisionOperator::new");
    for (b, sigma_edge) in edges_mt.iter().copied().enumerate() {
        let (lower, upper) = (b, b + 1);
        if !in_thermal_range(mesh.eps_c[upper], kt) {
            break;
        }
        let nu_m = process.fraction * number_density * sigma_edge * mesh.v_b[upper];
        if nu_m.partial_cmp(&0.0) != Some(Ordering::Greater) {
            continue;
        }
        let diffusion = 2.0 * process.mass_ratio * nu_m * mesh.eps_b[upper] * kt;
        let spacing = mesh.eps_c[upper] - mesh.eps_c[lower];
        let z = -spacing / kt + 0.5 * (mesh.eps_c[upper] / mesh.eps_c[lower]).ln();
        let rate_up = diffusion / spacing * bernoulli(-z) / mesh.d_eps[lower];
        let rate_down = diffusion / spacing * bernoulli(z) / mesh.d_eps[upper];
        nu_total[lower] += rate_up;
        deposits.push(Deposit {
            target: upper,
            source: lower,
            coefficient: rate_up,
        });
        nu_total[upper] += rate_down;
        deposits.push(Deposit {
            target: lower,
            source: upper,
            coefficient: rate_down,
        });
    }
}

/// 熱運動によるエネルギー交換を Fokker–Planck 項で扱う上限（kT の倍数）。
/// Maxwell分布は 40 kT で exp(−40) ≈ 4e-18 まで下がる。
const THERMAL_EXCHANGE_LIMIT: f64 = 40.0;

fn in_thermal_range(energy_ev: f64, kt_ev: f64) -> bool {
    energy_ev <= THERMAL_EXCHANGE_LIMIT * kt_ev
}

/// B(z) = z / (exp(z) − 1)
fn bernoulli(z: f64) -> f64 {
    if z.abs() < 1.0e-8 {
        1.0 - 0.5 * z
    } else {
        z / z.exp_m1()
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
    use crate::constants::kelvin_to_ev;

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

    fn apply(operator: &CollisionOperator, mesh: &VelocityMesh, state: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; mesh.n_cells];
        let mut energy_sum = vec![0.0; mesh.n_eps];
        let mut reinject = vec![0.0; mesh.n_eps];
        operator.apply(state, &mut output, &mut energy_sum, &mut reinject, false);
        output
    }

    fn elastic(mesh: &VelocityMesh, kt: f64) -> ProcessSpec {
        let sigma = vec![1.0e-19; mesh.n_eps];
        let mut process = ProcessSpec::new(
            "G".into(),
            "elastic".into(),
            ProcessKind::Elastic,
            1.0,
            0.0,
            1.0e-4,
            sigma,
        );
        process.gas_temperature_ev = kt;
        process.sigma_mt_edges = Some(vec![1.0e-19; mesh.n_eps - 1]);
        process
    }

    #[test]
    fn thermal_gas_keeps_maxwellian_stationary() {
        let mesh = VelocityMesh::new(1.0, 0.002, 4).unwrap();
        let kt = kelvin_to_ev(300.0);
        let operator = CollisionOperator::new(&mesh, 1.0e22, vec![elastic(&mesh, kt)]).unwrap();
        let state: Vec<f64> = (0..mesh.n_cells)
            .map(|k| {
                let (i, j) = (k / mesh.n_theta, k % mesh.n_theta);
                let e = mesh.eps_c[i];
                e.sqrt() * (-e / kt).exp() * mesh.d_eps[i] * mesh.w_theta[j]
            })
            .collect();
        let output = apply(&operator, &mesh, &state);
        for (k, (value, n)) in output.iter().zip(&state).enumerate() {
            let rate = operator.nu_total[k / mesh.n_theta];
            assert!(
                value.abs() <= 1.0e-9 * rate * n,
                "cell {k}: {value} vs {}",
                rate * n
            );
        }
    }

    #[test]
    fn isotropic_momentum_transfer_matches_plain_process() {
        // σ_m = σ（ξ = 0）なら異方散乱の経路を通らず、0.1.3と同じ結果になる
        let mesh = VelocityMesh::new(4.0, 0.1, 8).unwrap();
        let plain = elastic(&mesh, 0.0);
        let mut with_mt = plain.clone();
        with_mt.sigma_mt = Some(plain.sigma.clone());
        let a = CollisionOperator::new(&mesh, 1.0e22, vec![plain]).unwrap();
        let b = CollisionOperator::new(&mesh, 1.0e22, vec![with_mt]).unwrap();
        assert!(b.anisotropic.is_empty());
        let state: Vec<f64> = (0..mesh.n_cells).map(|k| 1.0 + (k % 7) as f64).collect();
        assert_eq!(apply(&a, &mesh, &state), apply(&b, &mesh, &state));
    }

    #[test]
    fn forward_scattering_conserves_electrons_and_keeps_direction() {
        let mesh = VelocityMesh::new(4.0, 0.1, 8).unwrap();
        let sigma = vec![1.0e-19; mesh.n_eps];
        let mut process = ProcessSpec::new(
            "G".into(),
            "rot".into(),
            ProcessKind::Excitation,
            1.0,
            0.05,
            0.0,
            sigma.clone(),
        );
        process.sigma_mt = Some(sigma.iter().map(|s| 1.0e-3 * s).collect());
        let operator = CollisionOperator::new(&mesh, 1.0e22, vec![process]).unwrap();
        assert!(!operator.anisotropic.is_empty());
        let mut state = vec![0.0; mesh.n_cells];
        for i in 0..mesh.n_eps {
            state[mesh.idx(i, 0)] = 1.0;
        }
        let output = apply(&operator, &mesh, &state);
        let total_rate: f64 = operator.nu_total.iter().sum();
        assert!(output.iter().sum::<f64>().abs() < 1.0e-12 * total_rate);
        // 等方なら 96% が他の方向へ移るが、前方散乱ではほとんど元の方向（j=0）に戻る
        let gain_other: f64 = (0..mesh.n_eps)
            .flat_map(|i| (1..mesh.n_theta).map(move |j| (i, j)))
            .map(|(i, j)| output[mesh.idx(i, j)])
            .sum();
        assert!(gain_other >= -1.0e-12 * total_rate);
        assert!(
            gain_other < 0.2 * total_rate,
            "{gain_other} vs {total_rate}"
        );
    }
}
