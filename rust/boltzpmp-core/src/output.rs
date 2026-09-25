use crate::{
    VelocityMesh,
    operators::{ProcessKind, ProcessSpec},
};

#[derive(Clone, Debug)]
pub struct SwarmScalars {
    pub eedf: Vec<f64>,
    pub eepf: Vec<f64>,
    pub mean_energy: f64,
    pub drift_velocity: f64,
    /// 電離周波数/N = Σ（標的の割合 × 速度係数）。
    pub reduced_ionization_frequency: f64,
    /// 付着周波数/N = Σ（標的の割合 × 速度係数）。
    pub reduced_attachment_frequency: f64,
    /// 過程ごとの速度係数（その過程の標的1個あたり）。キーは `気体名:過程名`。
    pub rate_coefficients: Vec<(String, f64)>,
    /// 各過程の標的の、全数密度に対する割合。速度係数と掛けると全体への寄与になる。
    pub fractions: Vec<(String, f64)>,
    /// EEPFの最終セルの値と最大値の比。
    pub eepf_tail_ratio: f64,
}

fn process_key(process: &ProcessSpec) -> String {
    format!("{}:{}", process.gas_name, process.name)
}

pub fn compute_swarm(
    state: &[f64],
    mesh: &VelocityMesh,
    processes: &[ProcessSpec],
) -> SwarmScalars {
    let total: f64 = state.iter().sum();
    let inv_total = 1.0 / total;
    let mut energy_density = vec![0.0; mesh.n_eps];
    let mut drift_velocity = 0.0;
    for i in 0..mesh.n_eps {
        for j in 0..mesh.n_theta {
            let normalized = state[mesh.idx(i, j)] * inv_total;
            energy_density[i] += normalized;
            drift_velocity += mesh.v_c[i] * mesh.theta_c[j].cos() * normalized;
        }
    }
    let mean_energy = mesh
        .eps_c
        .iter()
        .zip(&energy_density)
        .map(|(energy, density)| energy * density)
        .sum();
    let eedf: Vec<_> = energy_density
        .iter()
        .zip(&mesh.d_eps)
        .map(|(value, width)| value / width)
        .collect();
    let eepf: Vec<_> = eedf
        .iter()
        .zip(&mesh.eps_c)
        .map(|(value, energy)| value / energy.sqrt())
        .collect();
    let peak = eepf.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let eepf_tail_ratio = eepf.last().copied().unwrap_or(0.0) / peak.max(1.0e-300);

    let mut reduced_ionization_frequency = 0.0;
    let mut reduced_attachment_frequency = 0.0;
    let mut rate_coefficients = Vec::with_capacity(processes.len());
    let mut fractions = Vec::with_capacity(processes.len());
    for process in processes {
        let rate = (0..mesh.n_eps)
            .map(|i| process.sigma[i] * mesh.v_c[i] * energy_density[i])
            .sum::<f64>();
        match process.kind {
            ProcessKind::Ionization => reduced_ionization_frequency += process.fraction * rate,
            ProcessKind::Attachment => reduced_attachment_frequency += process.fraction * rate,
            _ => {}
        }
        rate_coefficients.push((process_key(process), rate));
        fractions.push((process_key(process), process.fraction));
    }
    SwarmScalars {
        eedf,
        eepf,
        mean_energy,
        drift_velocity,
        reduced_ionization_frequency,
        reduced_attachment_frequency,
        rate_coefficients,
        fractions,
        eepf_tail_ratio,
    }
}

pub fn cheap_scalars(state: &[f64], mesh: &VelocityMesh) -> (f64, f64) {
    let mut mean_energy = 0.0;
    let mut drift_velocity = 0.0;
    for i in 0..mesh.n_eps {
        let mut energy_density = 0.0;
        for j in 0..mesh.n_theta {
            let value = state[mesh.idx(i, j)];
            energy_density += value;
            drift_velocity += mesh.v_c[i] * mesh.theta_c[j].cos() * value;
        }
        mean_energy += mesh.eps_c[i] * energy_density;
    }
    (mean_energy, drift_velocity)
}

pub fn reduced_ionization_frequency(
    state: &[f64],
    mesh: &VelocityMesh,
    processes: &[ProcessSpec],
) -> f64 {
    let mut energy_density = vec![0.0; mesh.n_eps];
    for i in 0..mesh.n_eps {
        energy_density[i] = state[i * mesh.n_theta..(i + 1) * mesh.n_theta].iter().sum();
    }
    processes
        .iter()
        .filter(|process| process.kind == ProcessKind::Ionization)
        .map(|process| {
            process.fraction
                * (0..mesh.n_eps)
                    .map(|i| process.sigma[i] * mesh.v_c[i] * energy_density[i])
                    .sum::<f64>()
        })
        .sum()
}
