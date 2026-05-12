# basic-physics-lab

基礎物理実験のレポート作成や考察で使った解析スクリプトを保存するリポジトリです。
測定データから実験値を計算し、理論値やフィッティング結果と比較できるようにしています。

## 内容

| ディレクトリ | 実験 | 概要 |
| --- | --- | --- |
| `03-ac-circuits/` | 実験3: 交流回路の特性 | RLC回路の振幅特性・位相差・Q値・減衰振動を解析するスクリプト |
| `04-forced-and-damped-oscillations/` | 実験4: 強制振動・減衰振動 | 強制振動の半値幅・位相差・モンテカルロ不確かさ評価と、減衰振動の片対数フィッティングを行うスクリプト |
| `05-measurement-of-magnetic-flux-density/` | 実験5: 磁束密度の測定 | ホールプローブの角度依存性、コイル理論値との比較、距離不確かさ、モンテカルロ評価、両対数フィットを解析するスクリプト |

## 必要なもの

- uv
- Python 3.13

Python パッケージは `pyproject.toml` と `uv.lock` で管理します。
ローカルの仮想環境はリポジトリ直下の `.venv/` に作成されます。

## 環境構築

初回はリポジトリ直下で次のコマンドを実行します。

```bash
uv sync
```

これで NumPy, SciPy, Matplotlib, pandas が `.venv/` にインストールされます。

## 実行方法

実験3の解析スクリプトは次のコマンドで実行できます。

```bash
uv run python 03-ac-circuits/main.py
```

実験4の減衰振動フィッティングは次のコマンドで実行できます。

```bash
uv run python 04-forced-and-damped-oscillations/fit_damped_decay_semilog.py
```

実験4の強制振動に対するモンテカルロ不確かさ評価は次のコマンドで実行できます。

```bash
uv run python 04-forced-and-damped-oscillations/monte_carlo_forced_oscillation.py
```

実験5の磁束密度測定の解析は次のコマンドで実行できます。

```bash
uv run python 05-measurement-of-magnetic-flux-density/main.py
```

## 実験3で計算している値

`03-ac-circuits/main.py` では、交流回路の測定データを使って次の値を求めています。

- 半値幅法による共振周波数付近のQ値
- 振幅特性のフィッティングによる共振周波数とQ値
- フィッティング残差、RMSE、MAE
- 位相差のフィッティングによる共振周波数とQ値
- LCRの理論値から求めた共振周波数、Q値、減衰係数
- 減衰振動の振幅比から求めた減衰係数とQ値

## 実験4で計算している値

`04-forced-and-damped-oscillations/fit_damped_decay_semilog.py` では、減衰振動の電圧測定値を使って次の値を求めています。

- 片対数グラフ上での最小二乗フィッティング
- 自然対数・常用対数での傾き
- 初期電圧の推定値
- フィッティングの決定係数
- 減衰の時定数

`04-forced-and-damped-oscillations/monte_carlo_forced_oscillation.py` では、強制振動の測定データを使って次の値を求めています。

- 線形補間による半値幅 `f_-`, `f_+`, `Δf_r`
- 半値幅から求めた減衰率 `γ_B = πΔf_r`
- Q値
- 位相差が90度になる周波数
- 局所二次近似による共鳴周波数
- 測定値の読み取り不確かさを仮定したモンテカルロ評価

生成済みのグラフ画像として、初回測定の `first_measurement.png` と再測定点の `re_measured_points.png` を保存しています。

強制振動の測定結果は `04-forced-and-damped-oscillations/forced_oscillation_results.md` にまとめています。
モンテカルロ解析の代表的な実行結果は、スクリプト内のコメントにも記録しています。

## 実験5で計算している値

`05-measurement-of-magnetic-flux-density/main.py` では、磁束密度の測定データを使って次の値を求めています。

- 実験Aの角度依存性フィットによる磁場の主方向と大きさ
- 円形コイルの理論値と実測値の相対誤差
- 距離不確かさ \(\Delta h=0.4\,\mathrm{mm}\) による理論値の相対不確かさ
- 距離ずれを仮定したモンテカルロ評価
- コイルと磁石の両対数フィットによる \(B\propto h^{-n}\) の確認
- 遠方双極子近似による有効磁気モーメント

解析結果の文章は `05-measurement-of-magnetic-flux-density/magnetic_flux_density_results.md` にまとめています。
生成済みのグラフ画像とCSV表も同じディレクトリに保存しています。

## メモ

スクリプト内の測定データや回路定数は、実験レポートで使った値を直接書いています。
別の測定データで解析する場合は、各ファイル先頭付近の配列や定数を変更してください。
