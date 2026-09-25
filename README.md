# boltzpmp

`boltzpmp` は、プロパゲータ法で電子ボルツマン方程式を解く、Rustで高速化された
Pythonパッケージです。DC・RF電場における電子エネルギー分布、平均エネルギー、
ドリフト速度、反応速度係数を計算できます。

## インストール

Python 3.10以降が必要です。

```powershell
uv pip install boltzpmp
```

## クイックスタート

```python
import boltzpmp as bp

mixture = bp.load_argon()
solver = bp.PMSolver(
    mixture,
    eps_max_eV=25.0,
    d_eps_eV=0.2,
    n_theta=90,
)

result = solver.solve_dc(EN_Td=10.0)
print(result.mean_energy)
print(result.drift_velocity)
print(result.rate_coefficients)
```

RF周期定常計算も同じソルバーから実行できます。

```python
result = solver.solve_rf(EN_rms_Td=10.0, freq_Hz=13.56e6)
print(result.mean_energy_rms, result.drift_velocity_rms)
```

独立した換算電場点は、Rust計算中にPythonインタープリタを解放して並列実行できます。

```python
results = bp.solve_dc_sweep(
    solver,
    [0.5, 1.0, 2.0, 5.0, 10.0],
    max_workers=4,
    scheme="upwind",
)
```

計算点の並列化には `solve_dc_sweep` を使用してください。単一計算内の並列化は
`PMSolver(..., parallel=True)` で明示的に有効化できます。

## 断面積データ（LXCat・BOLSIG+形式）

`parse_lxcat` はLXCatとBOLSIG+の書式を読みます。読み込みと検証はRustコアが行い、書式の誤りは
行番号付きの `ValueError` になります。

- EXCITATIONの3行目の「しきい値 統計重み比」
- 反応式の `<->`（逆過程の標的を、生成物の気体とする）
- ROTATIONブロック（3行目と4行目に下準位・上準位の「エネルギー 統計重み」）
- 表の3列目（運動量移行断面積。このとき2列目は積分断面積として扱う）

## 物理モデル

0.2.0から、次の2つを既定で有効にしています。0.1.3と同じ模型で計算するには
`PMSolver(..., superelastic=False, gas_heating=False)` とします。

- 超弾性衝突（`superelastic=True`）。逆過程の断面積は詳細釣り合いで求めます。
  - ROTATION: 同じ気体のROTATIONブロックに現れる全準位の占有をBoltzmann因子で求め（全体で規格化）、
    遷移ごとに下準位・上準位の占有を掛けます（BOLSIG+と同じ）。
  - EXCITATIONで `<->` を使うか、生成物が混合気体の成分にあるとき: 逆過程の標的は生成物の気体です。
  - それ以外のEXCITATION: 下準位と上準位の2準位系として占有を求めます（BOLSIG+と同じ）。
  - 占有の温度は `Mixture(T_K=..., T_exc_K=..., transition_energy_eV=...)` で与えます
    （BOLSIG+のGas temperature、Excitation temperature、Transition energy。`T_exc_K`の既定は`T_K`）。
- 気体温度による弾性衝突のエネルギー交換（`gas_heating=True`）。電場がなければ格子上のMaxwell分布が
  厳密な定常解になるように離散化しています。

断面積に運動量移行断面積（`CrossSection.mt_data` または表の3列目）があると、その比 σ_m/σ から
遮蔽Rutherford型の角度分布（Okhrimovskyy et al., Phys. Rev. E 65, 037402 (2002)）を作り、
散乱後の方向の再分配に使います。極性分子の回転励起のような前方散乱を、積分断面積のまま扱えます。

## DC定常解の陰解法

0.3.0から、`solve_dc` の既定は陰解法（`method="implicit"`）です。定常方程式を直接解くので、
時間発展（`method="explicit"`）より数十〜数千倍速く、同じ離散方程式の解になります。

- 風上差分の移流と衝突の損失を、流れに沿った1回の走査で厳密に解きます（(ε, θ) の流れは閉路を作らない）。
- 衝突による再注入と、`limiter` の高次補正は前の反復の値を使い（ソース反復と欠損補正）、Anderson加速で収束を速めます。
- 電離・付着による電子数の増減は、陽解法と同じく和を1に保つ規格化として扱います。
- `tol`（既定1e-8）は1反復の残差 ‖g(n) − n‖₁、`max_steps` は反復回数の上限です。
- `scheme="blending"`（ξの探索）は陽解法だけで使えます。RFは陽解法のままです。

| 条件 | 陽解法 | 陰解法 |
|---|---:|---:|
| 同梱Ar、10 Td（0.2 eV刻み、n_theta = 90） | 24 s | 0.7 s |
| HF（xsecsim、非一様格子2331セル）、10 Td | 15 s | 0.3 s |
| HF、30 Td | 400 s | 0.6 s |
| HF、50 Td（4183セル） | 1.5時間で未収束 | 1.3 s |

## 数値スキームと格子

`solve_dc` と `solve_rf` の `scheme` で移流の離散化を選びます。

| scheme | 内容 |
|---|---|
| `limiter`（既定） | van Leer制限関数による2次精度のTVDスキーム。負の値を作らない |
| `upwind` | 1次精度。刻みと電場に比例する数値拡散で平均エネルギーを高めに出す |
| `blending` | ξ = 1（中心差分）から始め、負の値が出るたびに ξ を下げてやり直す |

熱平衡の近く（0.01 Td）での平均エネルギーの誤差は、5 meV刻みで `upwind` が +5.9%、`limiter` が +0.4% でした。

低エネルギーに細かい構造がある分子（回転しきい値が数meVのHFなど）では、非一様格子を使うと
少ないセル数と大きな時間刻みで計算できます。

```python
edges = bp.graded_energy_grid(eps_max_eV=60.0, d_eps_min_eV=0.0025, eps_uniform_eV=0.5)
solver = bp.PMSolver(mixture, energy_grid=edges, n_theta=16)
```

`eps_uniform_eV` までは一様刻み、その上は刻みを `sqrt(eps)` に比例して広げます（速度の刻みが一定）。

## 計算結果

- `rate_coefficients`: 過程ごとの速度係数（その過程の標的1個あたり）。キーは `気体名:過程名` で、
  逆過程には ` (superelastic)` が付きます。
- `fractions`: 各過程の標的の、全数密度に対する割合。混合気体全体への寄与は速度係数に掛けて足します。
- `reduced_ionization_frequency`、`reduced_attachment_frequency`: その和。`alpha_over_N`、`eta_over_N` はドリフト速度で割った値です。
- `PMSolver.processes()`: 組み立てた衝突過程（逆過程を含む）の一覧。

## LXCat断面積による検証

2026-08-15に、LXCatのMorgan databaseから取得したAr電子衝突断面積セットを使い、
1、10、50、100 Tdのupwind計算と、10、100 Tdの自動ブレンディング計算を検証しました
（0.1.3の模型。超弾性衝突と気体温度の効果なし）。
6条件すべてが収束し、Python参照実装との比較は次の結果でした。0.2.0でも、両方を切ると
同じ結果になることを互換テスト（`tests/parity`）で確かめています。

| 指標 | 最大誤差 | 合格基準 |
|---|---:|---:|
| 状態分布のL1差 | `3.52e-14` | `1e-5` |
| EEDFの相対L1差 | `3.55e-14` | `1e-5` |
| 主要物理量の相対差 | `3.63e-14` | `1e-5` |
| 反応速度係数の相対差 | `4.06e-14` | `1e-5` |

入力ファイルのSHA-256は
`29c903d91e68bb0895f45b763c8c982ef09c2b2e0636fc75fd0545dc7d69abc3` です。
条件、収束ステップ数、各物理量、生の誤差は
[`VALIDATION.md`](https://github.com/Reasonia-TK/boltzpmp/blob/main/VALIDATION.md) と
[`reference/lxcat_morgan_argon_validation.json`](https://github.com/Reasonia-TK/boltzpmp/blob/main/reference/lxcat_morgan_argon_validation.json)
に記録しています。

## 開発とテスト

Windows PowerShellでは次のコマンドを実行します。

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --with maturin maturin develop --release
uv run --extra test pytest -q
cargo test --workspace
```

実データ検証は、断面積ファイルとPython参照実装の場所を指定して再実行できます。

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --with scipy --extra test python benchmarks\validate_lxcat.py `
  'C:\path\to\Ar-cross-sections.txt' `
  --reference-source 'C:\path\to\python-reference' `
  --output reference\lxcat_morgan_argon_validation.json
```

現在のテスト構成はRust単体テスト28件、Python API・物理テスト38件です。

## パッケージ公開

`.github/workflows/wheels.yml` を手動実行すると、Windows、Linux、macOS向けwheelと
sdistを作成します。全ビルド成功後、選択した公開先へOIDC Trusted Publishingで
配布します。

| workflow入力 | 公開先 | GitHub Environment |
|---|---|---|
| `publish_testpypi` | TestPyPI | `testpypi` |
| `publish_pypi` | PyPI | `pypi` |

TestPyPI版を確認する場合は、依存パッケージと本体の取得先を分けます。

```powershell
uv pip install "numpy>=1.22"
uv pip install --no-deps --index-url https://test.pypi.org/simple/ boltzpmp
```

Trusted PublisherにはOwner `Reasonia-TK`、Repository `boltzpmp`、Workflow
`wheels.yml` と、公開先に対応するEnvironmentを設定します。

## ライセンス

MIT License
