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

## LXCat断面積による検証

2026-08-15に、LXCatのMorgan databaseから取得したAr電子衝突断面積セットを使い、
1、10、50、100 Tdのupwind計算と、10、100 Tdの自動ブレンディング計算を検証しました。
6条件すべてが収束し、Python参照実装との比較は次の結果でした。

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

現在のテスト構成はRust単体テスト4件、Python API・物理テスト15件です。

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
