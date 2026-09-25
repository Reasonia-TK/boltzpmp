pub const E_CHARGE: f64 = 1.602_176_634e-19;
pub const M_E: f64 = 9.109_383_701_5e-31;
pub const K_B: f64 = 1.380_649e-23;
pub const AMU: f64 = 1.660_539_066_60e-27;
pub const TOWNSEND: f64 = 1.0e-21;

#[inline]
pub fn speed_from_ev(energy_ev: f64) -> f64 {
    (2.0 * E_CHARGE * energy_ev / M_E).sqrt()
}

/// 温度 (K) を kT (eV) に換算する。
#[inline]
pub fn kelvin_to_ev(temperature_k: f64) -> f64 {
    K_B * temperature_k / E_CHARGE
}
