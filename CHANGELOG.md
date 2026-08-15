# Changelog

## 0.1.0 - Unreleased

- 配布名、import名、PyO3モジュール、Rust crateを `boltzpmp` 系へ統一。
- Python非依存のRust数値コアを追加。
- PyO3/Maturinによる `abi3-py310` Python拡張を追加。
- メッシュ、移流、衝突、DC、RFソルバーを移植。
- 旧Python版から生成した固定fixtureとの互換試験を追加。
- 単一計算内の実験的Rayon並列を追加。
- 独立DC計算を並列化する `solve_dc_sweep` を追加。
- Windows向けwheel/sdistのローカル構築手順を追加。
