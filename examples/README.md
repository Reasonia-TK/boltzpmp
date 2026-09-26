# boltzpmp の動作例

boltzpmp 0.5.0 以降で動きます（`uv pip install "boltzpmp>=0.5"`）。図を保存する例（`--plot`）には
matplotlib が必要です（`uv pip install matplotlib`）。出力は既定で `examples/output/` に書きます
（`--out` で変えられます）。時間は Windows のノートPC（1〜4スレッド）での目安です。

| ファイル | 内容 | 時間 |
|---|---|---:|
| `01_dc_transport_table.py` | 同梱の Ar で、DC の輸送係数と速度係数の表を作る。`solve_dc_many` で 1〜300 Td を並列に解き、CSV（と `--plot` で EEPF と輸送係数の図）を書く | 1秒未満 |
| `02_rf_waveforms.py` | 13.56 MHz の RF 周期定常解。平均エネルギー・ドリフト速度・電離周波数の波形、位相遅れ、保存点ごとの EEDF（`eedf_t`）、電場が最大の時刻と周期平均の速度係数の比較 | 約5秒 |
| `03_rf_frequency_scan.py` | 周波数を 100 kHz〜1 GHz で変え（`solve_rf_many`）、準静的な応答から実効電場の極限への移り変わりを DC 解と比べる | 約1分 |
| `04_molecular_gas.py` | 架空の分子気体で、回転励起と逆過程（ROTATION ブロック）、気体温度、異方散乱（表の3列目）、付着を比べる。RF の周期平均の付着周波数も求める | 約30秒 |
| `dc_argon.py` | いちばん短い DC の例 | 1秒未満 |
| `export_comsol_eedf.py` | COMSOL の Plasma インターフェース用の2引数 EEDF 表を書く | 数秒 |
| `export_comsol_eedf_sweep.ipynb`、`lxcat_electron_swarm.ipynb` | LXCat の断面積を使うノートブック | — |
| `validate_against_references.py` | 他のソルバー（BOLOS など）の結果との比較（参照データが必要） | — |

## 実行例

```powershell
python examples/01_dc_transport_table.py --plot
python examples/02_rf_waveforms.py --en-rms 30 --pressure 133 --plot
python examples/03_rf_frequency_scan.py --plot
python examples/04_molecular_gas.py
```

## 使っている機能（0.5.0）

- DC と RF の陰解法（既定）。DC は二項近似の合成加速と、二項近似の解から始める初期状態で、数十反復で収束する。
  収束しなかったときは `result.converged` が `False` になるので、例ではそのとき例外を投げる。
- RF の周期平均の出力: `mean_energy_avg`、`eedf_avg`、`eepf_avg`、`rate_coefficients_avg`、
  `reduced_ionization_frequency_avg`、`reduced_attachment_frequency_avg`、保存点ごとの EEDF `eedf_t`。
  基底クラスの `eedf` や `rate_coefficients` は、電場の大きさが最大の時刻の値（0.4 までと同じ）。
- 並列計算: `solve_dc_many`、`solve_rf_many`（Rust のスレッド。`max_workers` で数を決める）。
- 非一様エネルギー格子: `bp.graded_energy_grid(eps_max_eV, d_eps_min_eV, eps_uniform_eV)`。

## 注意

- RF で ω がエネルギー緩和の周波数よりずっと大きい条件（低い気体密度、高い周波数）は、収束に百周期以上かかる
  ことがある。`03_rf_frequency_scan.py` の 1 GHz（133 Pa）は約180周期かかる。`result.converged` と
  `result.extra["cycle_residuals"]` を確かめること。
- `04_molecular_gas.py` の分子 M の断面積は、この例のために作った架空の値。
