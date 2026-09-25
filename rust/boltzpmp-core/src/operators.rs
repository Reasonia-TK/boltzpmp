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

/// 制限関数つきの面。面の密度は `f_u + φ(r) (f_c − f_u)`、`f_c` は ξ = 1 と同じ体積重みの中心値、
/// `r = (f_u − f_uu) / (f_d − f_u)`、φ は van Leer の制限関数。
#[derive(Clone, Copy, Debug)]
struct LimitedEdge {
    second_upstream: Option<usize>,
    area: f64,
    inv_volume_upstream: f64,
    inv_volume_downstream: f64,
    inv_volume_second: f64,
    weight_upstream: f64,
    weight_downstream: f64,
}

/// 移流の離散化。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdvectionScheme {
    /// 風上（ξ = 0）から中心差分（ξ = 1）までの線形な重み。
    Linear(f64),
    /// van Leer の制限関数による2次精度のTVDスキーム（負の値を作らない）。
    VanLeer,
}

#[derive(Clone, Debug)]
pub struct AdvectionOperator {
    edges: Vec<FluxEdge>,
    limited: Option<Vec<LimitedEdge>>,
    cell_edges: Vec<Vec<SignedEdge>>,
    n_cells: usize,
}

impl AdvectionOperator {
    pub fn new(mesh: &VelocityMesh, xi: f64, sign: i8) -> Result<Self, String> {
        Self::with_scheme(mesh, AdvectionScheme::Linear(xi), sign)
    }

    pub fn with_scheme(
        mesh: &VelocityMesh,
        scheme: AdvectionScheme,
        sign: i8,
    ) -> Result<Self, String> {
        let xi = match scheme {
            AdvectionScheme::Linear(xi) => xi,
            AdvectionScheme::VanLeer => 0.0,
        };
        if !(0.0..=1.0).contains(&xi) {
            return Err(format!("xi must be in [0, 1], got {xi}"));
        }
        if sign != 1 && sign != -1 {
            return Err("sign must be +1 or -1".into());
        }
        let capacity = (mesh.n_eps.saturating_sub(1) * mesh.n_theta)
            + (mesh.n_eps * mesh.n_theta.saturating_sub(1));
        let mut edges = Vec::with_capacity(capacity);
        let mut limited = Vec::with_capacity(capacity);

        let map = |k: usize| {
            if sign == 1 { k } else { mesh.mirror_idx(k) }
        };
        let mut add_edge = |u: usize, d: usize, uu: Option<usize>, area: f64| {
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
            limited.push(LimitedEdge {
                second_upstream: uu.map(map),
                area,
                inv_volume_upstream: 1.0 / volume_u,
                inv_volume_downstream: 1.0 / volume_d,
                inv_volume_second: uu.map_or(0.0, |k| 1.0 / mesh.volume[k]),
                weight_upstream: volume_d / (volume_u + volume_d),
                weight_downstream: volume_u / (volume_u + volume_d),
            });
        };

        for i in 0..mesh.n_eps.saturating_sub(1) {
            for j in 0..mesh.n_theta {
                let lower = mesh.idx(i, j);
                let upper = mesh.idx(i + 1, j);
                let (upstream, downstream, second) = if mesh.theta_c[j] < PI / 2.0 {
                    (lower, upper, (i >= 1).then(|| mesh.idx(i - 1, j)))
                } else {
                    (
                        upper,
                        lower,
                        (i + 2 < mesh.n_eps).then(|| mesh.idx(i + 2, j)),
                    )
                };
                add_edge(upstream, downstream, second, mesh.s_plus_eps[lower]);
            }
        }
        for i in 0..mesh.n_eps {
            for j in 0..mesh.n_theta.saturating_sub(1) {
                let downstream = mesh.idx(i, j);
                let upstream = mesh.idx(i, j + 1);
                let second = (j + 2 < mesh.n_theta).then(|| mesh.idx(i, j + 2));
                add_edge(upstream, downstream, second, mesh.s_plus_theta[downstream]);
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
            limited: (scheme == AdvectionScheme::VanLeer).then_some(limited),
            cell_edges,
            n_cells: mesh.n_cells,
        })
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    #[inline]
    fn flux(&self, index: usize, state: &[f64]) -> f64 {
        let edge = &self.edges[index];
        match &self.limited {
            None => {
                edge.coeff_upstream * state[edge.upstream]
                    + edge.coeff_downstream * state[edge.downstream]
            }
            Some(limited) => {
                let face = &limited[index];
                let f_u = state[edge.upstream] * face.inv_volume_upstream;
                let f_d = state[edge.downstream] * face.inv_volume_downstream;
                let central = face.weight_upstream * f_u + face.weight_downstream * f_d;
                let phi = face.second_upstream.map_or(0.0, |uu| {
                    van_leer(f_u - state[uu] * face.inv_volume_second, f_d - f_u)
                });
                face.area * (f_u + phi * (central - f_u))
            }
        }
    }

    pub fn apply(&self, state: &[f64], output: &mut [f64], edge_flux: &mut [f64], parallel: bool) {
        assert_eq!(state.len(), self.n_cells);
        assert_eq!(output.len(), self.n_cells);
        assert!(edge_flux.len() >= self.edges.len());
        let flux = &mut edge_flux[..self.edges.len()];
        if parallel {
            flux.par_iter_mut()
                .enumerate()
                .for_each(|(index, value)| *value = self.flux(index, state));
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
            for (index, edge) in self.edges.iter().enumerate() {
                let value = self.flux(index, state);
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

/// 風上差分の移流と対角項からなる連立一次方程式を、流れに沿った1回の走査で解く。
///
/// θ方向の流れは常に θ = 0 側（負の加速度では π 側）へ向かい、エネルギー方向は前方半球で上向き、
/// 後方半球で下向きなので、流れのグラフに閉路はない。セルを流れの順（位相的順序）に並べると
/// 係数行列は三角行列になり、上流のセルから順に代入するだけで厳密に解ける。
#[derive(Clone, Debug)]
pub struct UpwindSweep {
    order: Vec<usize>,
    inflow_start: Vec<usize>,
    inflow_cell: Vec<usize>,
    inflow_coeff: Vec<f64>,
    outflow: Vec<f64>,
    n_theta: usize,
}

impl UpwindSweep {
    pub fn new(mesh: &VelocityMesh, sign: i8) -> Result<Self, String> {
        let operator = AdvectionOperator::new(mesh, 0.0, sign)?;
        let n = operator.n_cells;
        let mut outflow = vec![0.0; n];
        let mut indegree = vec![0usize; n];
        let mut downstream_of: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut inflow: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for edge in &operator.edges {
            outflow[edge.upstream] += edge.coeff_upstream;
            inflow[edge.downstream].push((edge.upstream, edge.coeff_upstream));
            downstream_of[edge.upstream].push(edge.downstream);
            indegree[edge.downstream] += 1;
        }
        // Kahnの方法で位相的順序を作る
        let mut order = Vec::with_capacity(n);
        let mut ready: Vec<usize> = (0..n).filter(|k| indegree[*k] == 0).collect();
        while let Some(k) = ready.pop() {
            order.push(k);
            for &d in &downstream_of[k] {
                indegree[d] -= 1;
                if indegree[d] == 0 {
                    ready.push(d);
                }
            }
        }
        if order.len() != n {
            return Err("the upwind advection graph has a cycle".into());
        }
        let mut inflow_start = Vec::with_capacity(n + 1);
        let mut inflow_cell = Vec::new();
        let mut inflow_coeff = Vec::new();
        inflow_start.push(0);
        for cell in &inflow {
            for (upstream, coeff) in cell {
                inflow_cell.push(*upstream);
                inflow_coeff.push(*coeff);
            }
            inflow_start.push(inflow_cell.len());
        }
        Ok(Self {
            order,
            inflow_start,
            inflow_cell,
            inflow_coeff,
            outflow,
            n_theta: mesh.n_theta,
        })
    }

    /// `(d_i + a·out_k) x_k − a Σ c x_upstream = s_k` を解く。`d_i` はエネルギーセルごとの対角項。
    pub fn solve(
        &self,
        acceleration: f64,
        diagonal_energy: &[f64],
        source: &[f64],
        x: &mut [f64],
    ) -> Result<(), String> {
        for &k in &self.order {
            let mut value = source[k];
            for index in self.inflow_start[k]..self.inflow_start[k + 1] {
                value += acceleration * self.inflow_coeff[index] * x[self.inflow_cell[index]];
            }
            let denominator = diagonal_energy[k / self.n_theta] + acceleration * self.outflow[k];
            if denominator.partial_cmp(&0.0) != Some(Ordering::Greater) {
                return Err(format!(
                    "cell {k} has neither collisions nor outflow; the implicit sweep cannot be solved"
                ));
            }
            x[k] = value / denominator;
        }
        Ok(())
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

/// van Leer の制限関数 φ(r) = (r + |r|)/(1 + |r|)、r = upwind/downwind。
/// 勾配の符号が変わる（極値）ところでは 0（風上差分）になる。
#[inline]
fn van_leer(upwind: f64, downwind: f64) -> f64 {
    let product = upwind * downwind;
    if product <= 0.0 {
        return 0.0;
    }
    2.0 * product / (downwind * downwind + product)
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

    #[test]
    fn limiter_conserves_and_keeps_steps_positive() {
        let mesh = VelocityMesh::new(8.0, 0.2, 16).unwrap();
        let op = AdvectionOperator::with_scheme(&mesh, AdvectionScheme::VanLeer, 1).unwrap();
        // 加速度1のときのCFL条件（出ていく面の面積の和）に安全係数0.2を掛けた時間刻み
        let dt = 0.2
            * (0..mesh.n_cells)
                .map(|k| {
                    let (i, j) = (k / mesh.n_theta, k % mesh.n_theta);
                    let out_eps = if mesh.theta_c[j] < PI / 2.0 {
                        if i + 1 == mesh.n_eps {
                            0.0
                        } else {
                            mesh.s_plus_eps[k]
                        }
                    } else {
                        mesh.s_minus_eps[k]
                    };
                    mesh.volume[k] / (out_eps + mesh.s_minus_theta[k])
                })
                .fold(f64::INFINITY, f64::min);
        // 階段状の分布（急な段差は2次精度の中心差分では負の値を生む）
        let mut state: Vec<f64> = (0..mesh.n_cells)
            .map(|k| {
                if k / mesh.n_theta < 8 {
                    mesh.volume[k]
                } else {
                    0.0
                }
            })
            .collect();
        let total: f64 = state.iter().sum();
        let mut output = vec![0.0; mesh.n_cells];
        let mut edge_flux = vec![0.0; op.edge_count()];
        for _ in 0..300 {
            op.apply(&state, &mut output, &mut edge_flux, false);
            for (value, change) in state.iter_mut().zip(&output) {
                *value += dt * change;
            }
            let max = state.iter().copied().fold(0.0, f64::max);
            assert!(state.iter().all(|value| *value >= -1.0e-14 * max));
        }
        assert!((state.iter().sum::<f64>() - total).abs() < 1.0e-12 * total);
        // 並列版も同じ流束を与える
        let mut parallel = vec![0.0; mesh.n_cells];
        op.apply(&state, &mut output, &mut edge_flux, false);
        op.apply(&state, &mut parallel, &mut edge_flux, true);
        for (a, b) in output.iter().zip(&parallel) {
            assert!((a - b).abs() <= 1.0e-12 * a.abs().max(1.0e-30));
        }
    }

    #[test]
    fn upwind_sweep_solves_the_linear_system() {
        let mesh = VelocityMesh::new(4.0, 0.1, 10).unwrap();
        for sign in [1, -1] {
            let sweep = UpwindSweep::new(&mesh, sign).unwrap();
            let upwind = AdvectionOperator::new(&mesh, 0.0, sign).unwrap();
            let acceleration = 3.0e5;
            let diagonal: Vec<f64> = (0..mesh.n_eps).map(|i| 1.0 + 0.1 * i as f64).collect();
            let source: Vec<f64> = (0..mesh.n_cells).map(|k| 1.0 + (k % 7) as f64).collect();
            let mut x = vec![0.0; mesh.n_cells];
            sweep
                .solve(acceleration, &diagonal, &source, &mut x)
                .unwrap();
            // (d − a·A_up) x = s を確かめる
            let mut advection = vec![0.0; mesh.n_cells];
            let mut edge_flux = vec![0.0; upwind.edge_count()];
            upwind.apply(&x, &mut advection, &mut edge_flux, false);
            for k in 0..mesh.n_cells {
                let lhs = diagonal[k / mesh.n_theta] * x[k] - acceleration * advection[k];
                assert!(
                    (lhs - source[k]).abs() <= 1.0e-10 * source[k],
                    "sign {sign} cell {k}"
                );
            }
            assert!(x.iter().all(|value| *value > 0.0));
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
