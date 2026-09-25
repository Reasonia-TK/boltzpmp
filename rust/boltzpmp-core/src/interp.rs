//! `numpy.interp`互換の区分線形補間。

/// `numpy.interp`と同じ演算順序で補間する。
///
/// x86-64では結果がビット単位で一致する。numpyが積和を融合演算（FMA）で計算する環境
/// （macOS arm64など）では、最後の1桁が異なることがある。
///
/// `xp`は非減少であること。`x < xp[0]`は`left`、`x > xp[last]`は`right`を返す。
/// 重複した`xp`があるときは、`xp[j] <= x`を満たす最後の`j`の区間を使う。
pub fn interp(x: f64, xp: &[f64], fp: &[f64], left: f64, right: f64) -> f64 {
    debug_assert_eq!(xp.len(), fp.len());
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if n == 1 {
        return if x < xp[0] {
            left
        } else if x > xp[0] {
            right
        } else {
            fp[0]
        };
    }
    if x < xp[0] {
        return left;
    }
    if x > xp[n - 1] {
        return right;
    }
    let j = xp.partition_point(|value| *value <= x) - 1;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut value = slope * (x - xp[j]) + fp[j];
    if value.is_nan() {
        value = slope * (x - xp[j + 1]) + fp[j + 1];
        if value.is_nan() && fp[j] == fp[j + 1] {
            value = fp[j];
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy_rules() {
        let xp = [0.0, 1.0, 1.0, 3.0];
        let fp = [0.0, 2.0, 4.0, 8.0];
        assert_eq!(interp(-1.0, &xp, &fp, -5.0, 9.0), -5.0);
        assert_eq!(interp(4.0, &xp, &fp, -5.0, 9.0), 9.0);
        assert_eq!(interp(0.5, &xp, &fp, 0.0, 0.0), 1.0);
        // 重複点では後ろ側の値
        assert_eq!(interp(1.0, &xp, &fp, 0.0, 0.0), 4.0);
        assert_eq!(interp(2.0, &xp, &fp, 0.0, 0.0), 6.0);
        assert_eq!(interp(3.0, &xp, &fp, 0.0, 0.0), 8.0);
        assert_eq!(interp(2.0, &[2.0], &[7.0], 1.0, 3.0), 7.0);
        assert!(interp(f64::NAN, &xp, &fp, 0.0, 0.0).is_nan());
    }
}
