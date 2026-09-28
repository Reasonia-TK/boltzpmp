# Changelog

## 0.6.0 - 2026-09-28

### 既定の挙動の変更（計算結果が変わる）

- 気体温度による弾性衝突のエネルギー交換（`gas_heating=True`）は、隣のエネルギーセルへ電子を移すとき向きを変えない
  ようにした。0.5.0 までは移るたびに等方に再注入していたので、その速さ（刻みの2乗に反比例）の分だけ運動量移行を
  余分に数え、熱平衡に近い低い E/N では格子を細かくするほどドリフト速度と平均エネルギーが下がっていた。
  - Ar（IST-Lisbon）の 0.0025 Td、1200 セル: 移動度が二項近似の厳密解（Davydov 分布）より 25% 小さかった。0.6.0 では
    格子を細かくすると厳密解に近づき、1200 セルで 0.05% 以内。
  - `benchmarks/implicit_versions.py` の弾性だけの気体（m/M = 1e-3、2 meV 刻み、0.01 Td）: ドリフト速度が 67.8 m/s
    （解析解 138.7 m/s の約半分）から 137.9 m/s になった。
  - 二項近似の演算子（合成加速、RF の実効電場の問題）の運動量移行の周波数にも、このエネルギー交換を入れない。
- EFFECTIVE（全運動量移行断面積）は、同じ標的の EXCITATION（しきい値が0以上）・IONIZATION・ATTACHMENT の断面積を
  引いた、弾性衝突の運動量移行断面積として使う（BOLSIG+、BOLOS と同じ）。0.5.0 までは ELASTIC と同じに扱い、
  非弾性衝突の運動量移行を二重に数えていた。引いた値が負になるところは0にして警告する。ROTATION は引かない。
  EFFECTIVE の過程の速度係数は、引いたあとの弾性衝突のもの。

### 修正

- DC の陰解法（RF の初期状態に使う実効電場の DC 問題も同じ）で、Ramsauer 極小の近くの低い E/N（Ar の 0.005 Td
  など）の粗い格子では、二項近似の合成加速の補正が行き過ぎて反復が周期的に振動し、収束しなかった。反復が発散したら
  （残差がそれまでの最小値の10倍を超えたら）、補正の係数 β を半分にして、残差が最小だった反復からやり直す
  （β は 0.7 から 0.0875 まで）。

### 追加

- `PMSolver.processes()` の各過程に `warning`（組み立てたときの注意。なければ `None`）。`PMSolver` を作るときに
  `UserWarning` としても出す。
- `examples/lxcat_to_comsol.ipynb`: 任意の LXCat 断面積から、平均電子エネルギーの範囲を指定して COMSOL 用の EEDF と
  換算電子移動度を作るノートブック（混合気体のモル分率、断面積の外挿、外挿した断面積の LXCat 形式での書き出し）。

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
