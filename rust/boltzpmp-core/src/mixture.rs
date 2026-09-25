//! 気体の組成と、励起準位の占有。

use std::collections::HashSet;

use crate::{
    constants::{AMU, K_B, M_E, kelvin_to_ev},
    lxcat::CrossSection,
};

#[derive(Clone, Debug)]
pub struct Gas {
    pub name: String,
    /// 全数密度に対する割合。
    pub fraction: f64,
    pub cross_sections: Vec<CrossSection>,
    pub mass_amu: Option<f64>,
}

impl Gas {
    /// 弾性衝突の質量比 m/M。断面積データの値を優先し、なければ分子量から求める。
    pub fn mass_ratio(&self, section: &CrossSection) -> Result<f64, String> {
        if let Some(ratio) = section.mass_ratio {
            return Ok(ratio);
        }
        match self.mass_amu {
            Some(mass) => mass_ratio_from_amu(mass),
            None => Err(format!(
                "no mass ratio available for elastic process of {}",
                self.name
            )),
        }
    }
}

pub fn mass_ratio_from_amu(mass_amu: f64) -> Result<f64, String> {
    if !mass_amu.is_finite() || mass_amu <= 0.0 {
        return Err(format!(
            "mass_amu must be finite and positive, got {mass_amu}"
        ));
    }
    Ok(M_E / (mass_amu * AMU))
}

/// 割合の和が1であることを確かめ、和を返す。
pub fn validate_fractions(fractions: impl IntoIterator<Item = f64>) -> Result<f64, String> {
    let mut total = 0.0;
    for fraction in fractions {
        if !fraction.is_finite() || fraction < 0.0 {
            return Err(format!(
                "mole fractions must be finite and non-negative, got {fraction}"
            ));
        }
        total += fraction;
    }
    // numpy.isclose(total, 1.0, rtol=1e-6) と同じ判定
    if (total - 1.0).abs() > 1.0e-8 + 1.0e-6 {
        return Err(format!("mole fractions sum to {total}, expected 1"));
    }
    Ok(total)
}

/// 数密度 (m⁻³)。`number_density`を優先し、なければ圧力と気体温度から求める。
pub fn number_density(
    pressure_pa: Option<f64>,
    temperature_k: f64,
    number_density: Option<f64>,
) -> Result<f64, String> {
    let value = match (number_density, pressure_pa) {
        (Some(n), _) => n,
        (None, Some(p)) => p / (K_B * temperature_k),
        (None, None) => return Err("give either N or p_Pa (with T_K)".into()),
    };
    if !value.is_finite() || value <= 0.0 {
        return Err("number density must be finite and positive".into());
    }
    Ok(value)
}

/// 励起準位の占有の温度モデル（BOLSIG+のExcitation temperatureとTransition energy）。
///
/// 基底状態からのエネルギーが`transition_energy_ev`以下の準位は気体温度、それより上は
/// 励起温度のBoltzmann因子に従う。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Populations {
    pub gas_temperature_k: f64,
    pub excitation_temperature_k: f64,
    pub transition_energy_ev: f64,
}

impl Populations {
    /// 基底状態を1とした占有因子（統計重みを含まない）。
    pub fn factor(&self, energy_ev: f64) -> f64 {
        self.log_factor(energy_ev).exp()
    }

    /// 占有因子の対数（占有が0なら負の無限大）。
    pub fn log_factor(&self, energy_ev: f64) -> f64 {
        let kt_gas = kelvin_to_ev(self.gas_temperature_k);
        let kt_exc = kelvin_to_ev(self.excitation_temperature_k);
        let u_tr = self.transition_energy_ev;
        if energy_ev <= u_tr {
            log_boltzmann(energy_ev, kt_gas)
        } else {
            log_boltzmann(u_tr, kt_gas) + log_boltzmann(energy_ev - u_tr, kt_exc)
        }
    }
}

fn log_boltzmann(energy_ev: f64, kt_ev: f64) -> f64 {
    if energy_ev <= 0.0 {
        0.0
    } else if kt_ev <= 0.0 {
        f64::NEG_INFINITY
    } else {
        -energy_ev / kt_ev
    }
}

#[derive(Clone, Debug)]
pub struct Mixture {
    pub gases: Vec<Gas>,
    pub number_density: f64,
    pub temperature_k: f64,
    pub populations: Populations,
}

impl Mixture {
    /// `excitation_temperature_k`を省略すると気体温度と同じにする。
    pub fn new(
        gases: Vec<Gas>,
        number_density: f64,
        temperature_k: f64,
        excitation_temperature_k: Option<f64>,
        transition_energy_ev: f64,
    ) -> Result<Self, String> {
        validate_fractions(gases.iter().map(|gas| gas.fraction))?;
        if !number_density.is_finite() || number_density <= 0.0 {
            return Err("number density must be finite and positive".into());
        }
        let excitation_temperature_k = excitation_temperature_k.unwrap_or(temperature_k);
        for (label, value) in [
            ("T_K", temperature_k),
            ("T_exc_K", excitation_temperature_k),
            ("transition_energy_eV", transition_energy_ev),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "{label} must be finite and non-negative, got {value}"
                ));
            }
        }
        let mut names = HashSet::new();
        for gas in &gases {
            if !names.insert(gas.name.as_str()) {
                return Err(format!("gas name {:?} appears more than once", gas.name));
            }
        }
        Ok(Self {
            gases,
            number_density,
            temperature_k,
            populations: Populations {
                gas_temperature_k: temperature_k,
                excitation_temperature_k,
                transition_energy_ev,
            },
        })
    }

    pub fn gas(&self, name: &str) -> Option<&Gas> {
        self.gases.iter().find(|gas| gas.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn population_factor_follows_bolsig_rules() {
        let single = Populations {
            gas_temperature_k: 300.0,
            excitation_temperature_k: 300.0,
            transition_energy_ev: 0.0,
        };
        let kt = kelvin_to_ev(300.0);
        assert!((single.factor(0.1) - (-0.1 / kt).exp()).abs() < 1e-15);
        assert_eq!(single.factor(0.0), 1.0);
        let cold = Populations {
            excitation_temperature_k: 0.0,
            ..single
        };
        assert_eq!(cold.factor(0.1), 0.0);
        let bi = Populations {
            gas_temperature_k: 300.0,
            excitation_temperature_k: 3000.0,
            transition_energy_ev: 0.3,
        };
        let expected = (-0.3 / kt).exp() * (-0.2 / kelvin_to_ev(3000.0)).exp();
        assert!((bi.factor(0.5) - expected).abs() < 1e-15);
    }

    #[test]
    fn validates_fractions_and_density() {
        assert!(validate_fractions([0.5, 0.5]).is_ok());
        assert!(validate_fractions([0.5, 0.4]).is_err());
        assert!(validate_fractions([1.5, -0.5]).is_err());
        assert!(
            (number_density(Some(133.0), 300.0, None).unwrap() - 133.0 / (K_B * 300.0)).abs() < 1.0
        );
        assert!(number_density(None, 300.0, None).is_err());
    }
}
