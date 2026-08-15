# boltzpmp 検証記録

## 目的

公開API、数値計算、配布物が再現可能な条件で正しく動作することを確認します。
検証は単体テスト、物理テスト、実データ比較、wheel/sdistの導入確認で構成します。

## Ar電子衝突断面積セット

2026-08-15に次の入力を使用しました。

| 項目 | 値 |
|---|---|
| 気体 | Ar 100% |
| データ源 | Morgan database, LXCat |
| 取得日 | 2026-07-11 |
| SHA-256 | `29c903d91e68bb0895f45b763c8c982ef09c2b2e0636fc75fd0545dc7d69abc3` |
| 過程 | 弾性1、励起2、電離1 |

パーサーは4過程と各データ点を読み込み、Python参照実装と過程種別、しきい値、
質量比、エネルギー点、断面積値が一致することを確認しました。

## 計算条件

| 項目 | 値 |
|---|---:|
| 圧力 | 133 Pa |
| 気体温度 | 273 K |
| 最大エネルギー | 60 eV |
| エネルギー刻み | 0.5 eV |
| 角度分割 | 48 |
| 収束許容値 | `1e-5` |
| 最大ステップ数 | 1,000,000 |
| 判定間隔 | 200ステップ |

upwindは100、50、10、1 Tdを高電場側からwarm startで計算しました。自動
ブレンディングは100、10 Tdを独立に計算しました。

## 結果

6条件はすべて収束し、収束ステップ数とブレンディング係数も参照計算と一致しました。

| 指標 | 最大値 | 合格基準 |
|---|---:|---:|
| 規格化誤差 | `1.15e-14` | `1e-12` |
| EEPF末端比 | `1.31e-14` | `1e-6` |
| 状態分布のL1差 | `3.52e-14` | `1e-5` |
| EEDFの相対L1差 | `3.55e-14` | `1e-5` |
| 主要物理量の最大相対差 | `3.63e-14` | `1e-5` |
| 反応速度係数の最大相対差 | `4.06e-14` | `1e-5` |

平均エネルギー、ドリフト速度、電離速度係数は換算電場に対して単調に増加しました。
全6条件の合計実行時間では、この実行のRust計算はPython参照計算の約2.56倍の速度でした。

ブレンディング100 Tdで状態値に最大値比 `-1.23e-16` の負値がありましたが、
両実装で一致する浮動小数点丸めの範囲で、判定基準 `-1e-14` を満たしています。

各条件の収束ステップ、計算値、反応ごとの速度係数、実行時間は
[`reference/lxcat_morgan_argon_validation.json`](https://github.com/Reasonia-TK/boltzpmp/blob/main/reference/lxcat_morgan_argon_validation.json)
を参照してください。

## 再実行

Windows PowerShellで次を実行します。

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --with scipy --extra test python benchmarks\validate_lxcat.py `
  'C:\path\to\Ar-cross-sections.txt' `
  --reference-source 'C:\path\to\python-reference' `
  --output reference\lxcat_morgan_argon_validation.json
```

成功時は終了コード0となり、JSONの `passed` と全 `checks` が `true` になります。
