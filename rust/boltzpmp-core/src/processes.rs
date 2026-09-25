//! 混合気体とメッシュから、セル中心の衝突過程を組み立てる。
//!
//! 超弾性衝突（逆過程）の規則:
//! - ROTATION: 同じ気体のROTATIONブロックに現れる全準位の占有をBoltzmann因子で求め
//!   （全体で規格化）、各遷移に下準位の占有を掛け、逆過程には上準位の占有を使う。
//! - EXCITATION で`<->`を使うか、生成物が混合気体の成分にあるとき: 逆過程の標的は生成物の気体で、
//!   その割合を使う（`<->`なのに生成物が成分にないときはエラー）。
//! - それ以外の EXCITATION: 下準位と上準位だけの2準位系として占有を求める（BOLSIG+と同じ）。
//! - しきい値が負の EXCITATION は、ファイルで明示された逆過程とみなし、さらに逆過程は作らない。
//!   生成物の気体がこの明示的な逆過程を持つときは、自動の逆過程を作らない（二重計上の防止）。
//!
//! 逆過程の断面積は詳細釣り合い `σ_sup(ε) = (g_low/g_up) (ε+u)/ε σ(ε+u)` をセル中心で直接評価する。

use std::collections::HashMap;

use crate::{
    constants::kelvin_to_ev,
    lxcat::{CrossSection, Kind, LevelState},
    mesh::VelocityMesh,
    mixture::{Gas, Mixture},
    operators::{ProcessKind, ProcessSpec},
};

/// これより小さい占有（気体の割合に対する比）の上準位からの逆過程は作らない。
/// 例えば 300 K で 10 eV の電子励起準位の占有は exp(−390) で、計算量だけが増える。
const NEGLIGIBLE_POPULATION: f64 = 1.0e-20;

#[derive(Clone, Copy, Debug)]
pub struct ModelOptions {
    /// 逆過程（超弾性衝突）を入れる。
    pub superelastic: bool,
    /// 弾性衝突で気体の熱運動によるエネルギー交換を入れる。
    pub gas_heating: bool,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            superelastic: true,
            gas_heating: true,
        }
    }
}

pub fn build_processes(
    mixture: &Mixture,
    mesh: &VelocityMesh,
    options: ModelOptions,
) -> Result<Vec<ProcessSpec>, String> {
    let kt_gas = kelvin_to_ev(mixture.temperature_k);
    let gases: HashMap<&str, &Gas> = mixture
        .gases
        .iter()
        .map(|gas| (gas.name.as_str(), gas))
        .collect();
    let mut specs = Vec::new();
    for gas in &mixture.gases {
        let rotational = rotational_populations(gas, mixture)?;
        for section in &gas.cross_sections {
            section.validate()?;
            let centres =
                |f: &dyn Fn(f64) -> f64| mesh.eps_c.iter().map(|e| f(*e)).collect::<Vec<_>>();
            let sigma = centres(&|e| section.sigma_at(e));
            let sigma_mt = section
                .momentum_transfer
                .as_ref()
                .map(|_| centres(&|e| section.momentum_transfer_at(e).expect("present")));
            let base = |kind: ProcessKind, fraction: f64, threshold: f64, mass_ratio: f64| {
                let mut spec = ProcessSpec::new(
                    gas.name.clone(),
                    section.name.clone(),
                    kind,
                    fraction,
                    threshold,
                    mass_ratio,
                    sigma.clone(),
                );
                spec.sigma_mt = sigma_mt.clone();
                spec
            };
            match section.kind {
                Kind::Elastic | Kind::Effective => {
                    let kind = if section.kind == Kind::Elastic {
                        ProcessKind::Elastic
                    } else {
                        ProcessKind::Effective
                    };
                    let mut spec = base(kind, gas.fraction, 0.0, gas.mass_ratio(section)?);
                    if options.gas_heating && kt_gas > 0.0 {
                        spec.gas_temperature_ev = kt_gas;
                        spec.sigma_mt_edges = Some(
                            mesh.eps_b[1..mesh.n_eps]
                                .iter()
                                .map(|e| {
                                    section
                                        .momentum_transfer_at(*e)
                                        .unwrap_or_else(|| section.sigma_at(*e))
                                })
                                .collect(),
                        );
                    }
                    specs.push(spec);
                }
                Kind::Ionization => {
                    let mut spec = base(
                        ProcessKind::Ionization,
                        gas.fraction,
                        section.threshold_ev,
                        0.0,
                    );
                    spec.sigma_mt = None;
                    specs.push(spec);
                }
                Kind::Attachment => {
                    let mut spec = base(ProcessKind::Attachment, gas.fraction, 0.0, 0.0);
                    spec.sigma_mt = None;
                    specs.push(spec);
                }
                Kind::Excitation => {
                    let u = section.threshold_ev;
                    let weight = section.weight_ratio.unwrap_or(1.0);
                    let explicit_inverse = u < 0.0;
                    let product = section.product().and_then(|name| gases.get(name).copied());
                    if section.is_reversible() && product.is_none() {
                        return Err(format!(
                            "{}: product {:?} of a '<->' process is not a gas of the mixture",
                            section.name,
                            section.product().unwrap_or("")
                        ));
                    }
                    let (fraction, inverse) =
                        if !options.superelastic || explicit_inverse || u <= 0.0 {
                            (gas.fraction, None)
                        } else if let Some(upper) = product {
                            let inverse = (!has_explicit_inverse(upper, gas, u))
                                .then_some((upper, upper.fraction));
                            (gas.fraction, inverse)
                        } else {
                            let f = weight * mixture.populations.factor(u);
                            let y_low = 1.0 / (1.0 + f);
                            (
                                gas.fraction * y_low,
                                (f * y_low > NEGLIGIBLE_POPULATION)
                                    .then_some((gas, gas.fraction * f * y_low)),
                            )
                        };
                    specs.push(base(ProcessKind::Excitation, fraction, u, 0.0));
                    if let Some((target_gas, inverse_fraction)) = inverse {
                        specs.push(superelastic(
                            section,
                            &target_gas.name,
                            inverse_fraction,
                            weight,
                            mesh,
                        ));
                    }
                }
                Kind::Rotation => {
                    let (lower, upper) = (
                        section.lower_state.expect("validated"),
                        section.upper_state.expect("validated"),
                    );
                    let y_low = rotational.population(lower);
                    let y_up = rotational.population(upper);
                    specs.push(base(
                        ProcessKind::Excitation,
                        gas.fraction * y_low,
                        section.threshold_ev,
                        0.0,
                    ));
                    if options.superelastic && y_up > NEGLIGIBLE_POPULATION {
                        specs.push(superelastic(
                            section,
                            &gas.name,
                            gas.fraction * y_up,
                            upper.weight / lower.weight,
                            mesh,
                        ));
                    }
                }
            }
        }
    }
    Ok(specs)
}

/// 詳細釣り合いによる逆過程。`weight`は g_up/g_low。
fn superelastic(
    section: &CrossSection,
    gas_name: &str,
    fraction: f64,
    weight: f64,
    mesh: &VelocityMesh,
) -> ProcessSpec {
    let u = section.threshold_ev;
    let factor = |e: f64| (e + u) / e / weight;
    let sigma = mesh
        .eps_c
        .iter()
        .map(|e| factor(*e) * section.sigma_at(e + u))
        .collect();
    let mut spec = ProcessSpec::new(
        gas_name.to_string(),
        format!("{} (superelastic)", section.name),
        ProcessKind::Excitation,
        fraction,
        -u,
        0.0,
        sigma,
    );
    // 逆過程の角度分布は、順過程のエネルギー ε+u での分布に等しい
    spec.sigma_mt = section.momentum_transfer.as_ref().map(|_| {
        mesh.eps_c
            .iter()
            .map(|e| factor(*e) * section.momentum_transfer_at(e + u).expect("present"))
            .collect()
    });
    spec
}

/// 生成物の気体が、この励起の逆過程（しきい値 −u で元の気体へ戻る EXCITATION）をすでに持つか。
fn has_explicit_inverse(upper: &Gas, lower: &Gas, u: f64) -> bool {
    upper.cross_sections.iter().any(|section| {
        section.kind == Kind::Excitation
            && section.threshold_ev < 0.0
            && (section.threshold_ev + u).abs() <= 1.0e-9 * u.abs().max(1.0e-12)
            && section.product() == Some(lower.name.as_str())
    })
}

struct RotationalEnsemble {
    states: Vec<(LevelState, f64)>,
}

impl RotationalEnsemble {
    fn population(&self, state: LevelState) -> f64 {
        self.states
            .iter()
            .find(|(candidate, _)| same_state(*candidate, state))
            .map(|(_, y)| *y)
            .expect("state collected from the same gas")
    }
}

fn same_state(a: LevelState, b: LevelState) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= 1.0e-10 * x.abs().max(y.abs()).max(1.0e-12);
    close(a.energy_ev, b.energy_ev) && close(a.weight, b.weight)
}

/// ROTATIONブロックの全準位について、Boltzmann因子で占有を求めて規格化する（BOLSIG+の規則）。
fn rotational_populations(gas: &Gas, mixture: &Mixture) -> Result<RotationalEnsemble, String> {
    let mut states: Vec<(LevelState, f64)> = Vec::new();
    for section in gas
        .cross_sections
        .iter()
        .filter(|s| s.kind == Kind::Rotation)
    {
        for state in [section.lower_state, section.upper_state]
            .into_iter()
            .flatten()
        {
            if !states.iter().any(|(known, _)| same_state(*known, state)) {
                states.push((state, 0.0));
            }
        }
    }
    if states.is_empty() {
        return Ok(RotationalEnsemble { states });
    }
    // エネルギーは基底状態から測った値。桁あふれを避けるため対数で規格化する
    for (state, y) in &mut states {
        *y = state.weight.ln() + mixture.populations.log_factor(state.energy_ev);
    }
    let largest = states
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    if largest == f64::NEG_INFINITY {
        return Err(format!(
            "{}: all rotational states have zero population (excitation temperature 0?)",
            gas.name
        ));
    }
    for (_, y) in &mut states {
        *y = (*y - largest).exp();
    }
    let total: f64 = states.iter().map(|(_, y)| y).sum();
    for (_, y) in &mut states {
        *y /= total;
    }
    Ok(RotationalEnsemble { states })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lxcat::parse_str;

    fn mixture(text: &str, t_k: f64) -> Mixture {
        let gas = Gas {
            name: "HF".into(),
            fraction: 1.0,
            cross_sections: parse_str(text).unwrap(),
            mass_amu: Some(20.0),
        };
        Mixture::new(vec![gas], 3.2e22, t_k, None, 0.0).unwrap()
    }

    const ROTATION: &str = "ROTATION\nHF\n 0.0 1.0\n 0.005 3.0\n-----\n 0.005 0\n 1.0 1e-19\n-----\n\
ROTATION\nHF\n 0.005 3.0\n 0.015 5.0\n-----\n 0.010 0\n 1.0 1e-19\n-----\n";

    #[test]
    fn rotational_ensemble_is_normalised_with_inverses() {
        let mesh = VelocityMesh::new(2.0, 0.01, 4).unwrap();
        let mix = mixture(ROTATION, 300.0);
        let specs = build_processes(&mix, &mesh, ModelOptions::default()).unwrap();
        assert_eq!(specs.len(), 4);
        let kt = kelvin_to_ev(300.0);
        let w = [1.0, 3.0 * (-0.005 / kt).exp(), 5.0 * (-0.015 / kt).exp()];
        let total: f64 = w.iter().sum();
        assert!((specs[0].fraction - w[0] / total).abs() < 1e-12);
        assert!((specs[1].fraction - w[1] / total).abs() < 1e-12);
        assert_eq!(specs[1].threshold_ev, -0.005);
        // 詳細釣り合い: g_low σ_exc(ε+u)(ε+u) = g_up σ_sup(ε) ε
        let e = mesh.eps_c[10];
        let expected = (e + 0.005) / e / 3.0 * mix.gases[0].cross_sections[0].sigma_at(e + 0.005);
        assert!((specs[1].sigma[10] - expected).abs() <= 1e-15 * expected);
        let without = build_processes(
            &mix,
            &mesh,
            ModelOptions {
                superelastic: false,
                gas_heating: true,
            },
        )
        .unwrap();
        assert_eq!(without.len(), 2);
    }

    #[test]
    fn product_species_is_the_inverse_target() {
        let text_low = "EXCITATION\nA -> B\n 0.1 2.0\n-----\n 0.1 0\n 1 1e-20\n-----\n";
        let low = Gas {
            name: "A".into(),
            fraction: 0.9,
            cross_sections: parse_str(text_low).unwrap(),
            mass_amu: Some(4.0),
        };
        let high = Gas {
            name: "B".into(),
            fraction: 0.1,
            cross_sections: vec![],
            mass_amu: Some(4.0),
        };
        let mix = Mixture::new(vec![low, high], 1e22, 300.0, None, 0.0).unwrap();
        let mesh = VelocityMesh::new(2.0, 0.01, 4).unwrap();
        let specs = build_processes(&mix, &mesh, ModelOptions::default()).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].fraction, 0.9);
        assert_eq!(specs[1].gas_name, "B");
        assert_eq!(specs[1].fraction, 0.1);
    }

    #[test]
    fn two_level_populations_without_product_species() {
        let text = "EXCITATION\nHF -> HF(v1)\n 0.5 1.0\n-----\n 0.5 0\n 2 1e-20\n-----\n";
        let mix = mixture(text, 3000.0);
        let mesh = VelocityMesh::new(2.0, 0.01, 4).unwrap();
        let specs = build_processes(&mix, &mesh, ModelOptions::default()).unwrap();
        let f = (-0.5 / kelvin_to_ev(3000.0)).exp();
        assert!((specs[0].fraction - 1.0 / (1.0 + f)).abs() < 1e-14);
        assert!((specs[1].fraction - f / (1.0 + f)).abs() < 1e-14);
    }

    #[test]
    fn reversible_arrow_requires_product_gas() {
        let text = "EXCITATION\nHF <-> HF*\n 0.5\n-----\n 0.5 0\n 2 1e-20\n-----\n";
        let mix = mixture(text, 300.0);
        let mesh = VelocityMesh::new(2.0, 0.01, 4).unwrap();
        assert!(build_processes(&mix, &mesh, ModelOptions::default()).is_err());
    }

    #[test]
    fn elastic_gets_thermal_exchange_data() {
        let text = "ELASTIC\nHF\n 2.7e-5\n-----\n 0 1e-19\n 10 1e-19\n-----\n";
        let mix = mixture(text, 300.0);
        let mesh = VelocityMesh::new(2.0, 0.01, 4).unwrap();
        let specs = build_processes(&mix, &mesh, ModelOptions::default()).unwrap();
        assert!(specs[0].gas_temperature_ev > 0.0);
        assert_eq!(
            specs[0].sigma_mt_edges.as_ref().unwrap().len(),
            mesh.n_eps - 1
        );
        let cold = build_processes(
            &mix,
            &mesh,
            ModelOptions {
                superelastic: true,
                gas_heating: false,
            },
        )
        .unwrap();
        assert_eq!(cold[0].gas_temperature_ev, 0.0);
    }
}
