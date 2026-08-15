pub const E_CHARGE: f64 = 1.602_176_634e-19;
pub const M_E: f64 = 9.109_383_701_5e-31;
pub const TOWNSEND: f64 = 1.0e-21;

#[inline]
pub fn speed_from_ev(energy_ev: f64) -> f64 {
    (2.0 * E_CHARGE * energy_ev / M_E).sqrt()
}
