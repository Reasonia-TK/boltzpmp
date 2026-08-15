# boltzpmp

`boltzpmp` は、プロパゲータ法で電子ボルツマン方程式を解く `boltzpm` の
Rust移植版です。数値計算コアをRustで実装し、Pythonから既存版に近いAPIで利用
できることを目標としています。

> 現在は移植中のalpha版です。数値互換試験と性能評価が完了するまで、研究結果の
> 本計算には旧版との比較なしで使用しないでください。

## 開発環境

Windows PowerShellでは次のように開発用インストールと検証を行います。

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --with maturin maturin develop
uv run --extra test pytest -q
cargo test --workspace
```

## TestPyPI公開

`.github/workflows/wheels.yml` を手動実行し、`publish_testpypi` を有効にすると、
3 OS向けwheelとsdistの全ビルド成功後にTestPyPIへ公開します。公開ジョブは
GitHubの `testpypi` EnvironmentとOIDC Trusted Publishingを使用し、APIトークンを
リポジトリへ保存しません。TestPyPI側のpublisherには次を設定してください。

| 項目 | 値 |
|---|---|
| Owner | `Reasonia-TK` |
| Repository | `boltzpmp` |
| Workflow | `wheels.yml` |
| Environment | `testpypi` |

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

RF周期定常計算も旧版と同じ形で実行できます。

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

`PMSolver(..., parallel=True)` は単一計算内の実験的Rayon並列です。現在の代表
メッシュでは同期コストが上回ったため、既定は `False` です。通常は
`solve_dc_sweep` による計算点単位の並列化を使用してください。

## 現在の検証結果

- 旧Python版のテスト: 26件成功
- Rust単体テスト: 4件成功
- Python/Rust互換・物理試験: 14件成功
- 54,000セル固定ステップ: 旧版比 約3.2倍
- 54,000セル計算4件の並列掃引: 逐次Rust比 約2.1倍

値は2026-08-15に同一Windows環境で取得したもので、CPUや入力条件に依存します。
生の測定値は `reference/*.json` に保存しています。

## 既知の制約

- 低換算電場では物理的な緩和時間が長く、Rust化しても必要ステップ数自体は減りません。
- 単一計算内Rayon並列は実験機能であり、メッシュによって遅くなる場合があります。
- 同梱Ar/Ar*断面積は検証用の近似データです。本計算には信頼できるLXCatデータを
  指定してください。
- alpha期間中は重要な計算で旧 `boltzpm` との比較を行ってください。

移植方針と検証ゲートは [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) を参照してください。

## ライセンス

MIT License
