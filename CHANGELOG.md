# Changelog

## 0.3.0 - 2026-09-25

### 既定の挙動の変更

- `solve_dc`、`solve_dc_many`、`solve_dc_sweep` の既定を陰解法（`method="implicit"`）にした。
  0.2.0と同じ時間発展で解くには `method="explicit"` を指定する。
- `tol` の既定は方法ごとに決まる（陽解法1e-6、陰解法1e-8）。

### 追加

- DC定常解の陰解法。風上差分の移流と衝突の損失を流れに沿った1回の走査（`UpwindSweep`）で解き、
  再注入と制限関数の高次補正を前の反復の値で扱うソース反復を、Anderson加速で速める。
  陽解法と同じ離散方程式の解を、HFで数十〜数千倍、同梱Arで数十倍速く求める。
- `DcMethod`（Rust）と `method` 引数（Python）。

## 0.2.0 - 2026-09-25

### 既定の挙動の変更

- 超弾性衝突と、気体温度による弾性衝突のエネルギー交換を既定で有効にした。0.1.3と同じ模型にするには
  `PMSolver(..., superelastic=False, gas_heating=False)` を指定する。
- 既定の移流スキームを `blending` から `limiter`（van Leer制限関数、2次精度）に変えた。
- LXCatの3行目に「しきい値 統計重み比」の2数がある過程で、しきい値が0になっていた不具合を直した。

### 追加

- Rustコア: LXCat/BOLSIG+パーサー、numpy.interp互換の補間、Gas・Mixture、励起準位の占有、
  衝突過程の組み立てを移し、Python側は公開APIを保った薄い層にした。
- ROTATIONブロック、反応式の `<->`、表の3列目（運動量移行断面積）の読み込み。
- BOLSIG+と同じ規則の超弾性衝突（ROTATIONの準位集団、2準位系、生成物の気体）。
  `Mixture(T_exc_K=..., transition_energy_eV=...)` で占有の温度を与える。
- 遮蔽Rutherford型の異方散乱（運動量移行断面積と積分断面積の比から角度分布を作る）。
- 非一様エネルギー格子（`PMSolver(energy_grid=...)`、`VelocityMesh.from_edges`、`graded_energy_grid`）。
- `solve_dc_sweep` をRustのスレッドプールで実行する `PMSolver.solve_dc_many`。
- 結果の `reduced_attachment_frequency`、`fractions`、`alpha_over_N`、`eta_over_N`、
  `PMSolver.processes()`。

### 注意

- `rate_coefficients` は過程の標的1個あたりの値（0.1.3と同じ）。混合気体全体の値は `fractions` を掛けて足す。
- ROTATIONブロックは0.1.3では読み飛ばされていた。

## 0.1.3 - 2026-08-15

- PyPI向けOIDC Trusted Publishingジョブを追加。
- 通常のPyPIインストール手順と公開先ごとのworkflow入力を文書化。

## 0.1.2 - 2026-08-15

- TestPyPIとPyPIを分離して使用するuvインストール手順へ更新。

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
