# Changelog

## 0.5.0 - 2026-09-26

### 既定の挙動の変更

- DCの陰解法（`solve_dc`、`solve_dc_many`）は、`init_n` を与えないとき、エネルギーだけの二項近似の定常解から始める
  （`init_T_eV` の Maxwell 分布は、その電子数の増加率を決めるのに使う）。
- 陽解法のRF（`method="explicit"`）の保存点を、陰解法と同じく周期全体に等間隔に取るようにした。0.4.0 までは
  1周期の段数が `n_store` の倍数でないと保存点が周期の終わりまで届かず、実効値が偏っていた。時間発展そのものは
  変わらない（最終状態は旧版と同じ）。

### 追加

- DCの陰解法の合成加速: ソース反復で遅いモード（エネルギー緩和）を、エネルギーだけの二項近似の演算子で直接解いて
  補正する（帯行列の LU 分解）。反復回数が数十分の1になり、0.4.0 では既定の初期状態から収束しなかった条件
  （HF の低い E/N など）も収束する。二項近似を使えなかったときは、その理由が `extra["two_term_error"]` に入る。
- RFの陰解法
  - 初期状態の実効電場の問題を、上の合成加速で解く（0.4.0 では低圧で収束していなかった）。その解の非等方成分の
    向きと大きさを、周期の始め（電場が最大）の値に直してから始める。
  - 遅いモードの補正: 1周期の流束から低次の演算子を合わせ（中性子輸送の非線形拡散加速と同じ考え方）、周期の残差の
    減りが鈍ったら補正する。補正のあと残差が増えたら使わない。
  - 収束の判定は、周期の残差と、上の低次の演算子による遅いモードの誤差の見積もりの両方。
- RFの周期平均の出力（`SwarmResultRF`）: `mean_energy_avg`、`eedf_avg`、`eepf_avg`、`rate_coefficients_avg`、
  `reduced_ionization_frequency_avg`、`reduced_attachment_frequency_avg`、保存点ごとの EEDF `eedf_t`。
  基底クラスの値（電場が最大の時刻の EEDF と速度係数、波形の実効値）は 0.4.0 と同じ。
- `PMSolver.solve_rf_many`（複数の E/N と周波数を Rust のスレッドで並列に解く）。
- `benchmarks/implicit_versions.py`（陰解法の反復回数と時間の比較）、`benchmarks/generate_rf_waveforms.py`。

### 修正

- RFの周期写像の Anderson 加速で、値がほぼ0のセル1つのために外挿の幅が0になり、加速が効かなくなっていた。

## 0.4.0 - 2026-09-26

### 既定の挙動の変更

- `solve_rf` の既定を陰解法（`method="implicit"`）にした。0.3.0と同じ時間発展で解くには
  `method="explicit"` を指定する。
- `solve_rf` の `tol` の既定は方法ごとに決まる（陽解法1e-4、陰解法1e-6）。

### 追加

- RF周期定常解の陰解法。
  - 各段をBDF2（1段目と、右辺が負になる段は後退Euler）で陰的に解く。解き方はDCの陰解法と同じ掃き出しとソース反復。
    時間刻みは安定条件に縛られず、1周期の既定は256段以上（`n_store` の倍数）。
  - 1周期の写像の不動点を、Anderson加速と実効電場の前処理で求める。
  - 前処理と初期状態には、実効値の電場に、エネルギーを変えない等方散乱 ω²/ν を加えたDC問題を使う
    （高周波の極限で時間平均の分布になる）。
  - 収束の判定は、周期写像の残差と前処理による誤差の見積もりの両方。
- `SolveMethod`（Rust。`DcMethod` はその別名）、`solve_rf` の `method` 引数（Python）。
- RFの結果の `extra` に `inner_iterations`（各段の反復の合計）と `cycle_residuals`（周期ごとの残差）を追加した。
- `benchmarks/rf_implicit.py`（陰解法と陽解法の比較）。

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
