# Changelog

## 0.1.1 - 2026-08-15

- LXCat Morgan databaseのAr電子衝突断面積セットによる6条件の収束・物理量検証を追加。
- 入力パーサー、分布、EEDF、主要物理量、反応速度係数の再現性を確認。
- 検証条件と生の測定結果を公開文書へ追加。
- 配布メタデータとパッケージ文書を更新。

## 0.1.0 - 2026-08-15

- 配布名、import名、PyO3モジュール、Rust crateを `boltzpmp` 系へ統一。
- Rust数値コアとPyO3/Maturinによる `abi3-py310` Python拡張を追加。
- メッシュ、移流、衝突、DC、RFソルバーを実装。
- 固定fixtureによる数値回帰試験と物理試験を追加。
- 単一計算内のRayon並列と、独立DC計算を並列化する `solve_dc_sweep` を追加。
- Windows、Linux、macOS向けwheelとsdistのCIビルドを追加。
