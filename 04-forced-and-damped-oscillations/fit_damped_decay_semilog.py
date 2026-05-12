"""減衰振動の片対数フィッティング。

読み取りデータ
--------------
初回測定:
    時間 [ms]    電圧 [V]
    230          1.60
    400          1.20
    750          0.80
    1960         0.25
    2320         0.10

再測定:
    時間が経つと振幅差が小さくなり、読み取り誤差が大きくなると考えた。
    そこで、比較的読み取りやすい前半の点を中心に取り直した。

    時間 [ms]    電圧 [V]
    230          1.60
    400          1.20
    750          0.80
    1000         0.65
    1200         0.50
"""

import numpy as np
import matplotlib.pyplot as plt

datasets = [
    {
        "name": "dataset_1",
        "title": "Semilog plot: first measurement",
        "t_ms": np.array([230, 400, 750, 1960, 2320], dtype=float),
        "v": np.array([1.6, 1.2, 0.8, 0.25, 0.10], dtype=float),
    },
    {
        "name": "dataset_2",
        "title": "Semilog plot: re-measured points",
        "t_ms": np.array([230, 400, 750, 1000, 1200], dtype=float),
        "v": np.array([1.6, 1.2, 0.8, 0.65, 0.50], dtype=float),
    },
]

results = []

for d in datasets:
    t = d["t_ms"]
    v = d["v"]
    ln_v = np.log(v)

    # Least-squares fit: ln(V) = a + b t
    b, a = np.polyfit(t, ln_v, 1)
    ln_v_fit = a + b * t

    ss_res = np.sum((ln_v - ln_v_fit) ** 2)
    ss_tot = np.sum((ln_v - np.mean(ln_v)) ** 2)
    r2 = 1 - ss_res / ss_tot

    t_line = np.linspace(t.min(), t.max(), 300)
    v_line = np.exp(a + b * t_line)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(t, v, label="measured points")
    plt.plot(t_line, v_line, label="least-squares fit")
    plt.yscale("log")
    plt.xlabel("time t [ms]")
    plt.ylabel("voltage V [V] (log scale)")
    plt.title(d["title"])
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    results.append(
        {
            "dataset": d["name"],
            "ln_slope_per_ms": b,
            "ln_slope_per_s": b * 1000,
            "log10_slope_per_ms": b / np.log(10),
            "log10_slope_per_s": b * 1000 / np.log(10),
            "intercept_ln": a,
            "V0": np.exp(a),
            "R2_on_lnV": r2,
            "time_constant_ms": -1 / b,
        }
    )

for r in results:
    print(r)

# 実行結果メモ
#
# dataset_1: 初回測定
# ln_slope_per_ms = -0.0012124568903412304
# ln_slope_per_s = -1.2124568903412305
# log10_slope_per_ms = -0.0005265633370207725
# log10_slope_per_s = -0.5265633370207725
# intercept_ln = 0.7205616359885826
# V0 = 2.055587378351847 V
# R2_on_lnV = 0.977476363469548
# time_constant_ms = 824.771592265489 ms
#
# dataset_2: 再測定点
# ln_slope_per_ms = -0.0011523152794980535
# ln_slope_per_s = -1.1523152794980536
# log10_slope_per_ms = -0.0005004441672988079
# log10_slope_per_s = -0.5004441672988079
# intercept_ln = 0.6861080477352225
# V0 = 1.9859711676842047 V
# R2_on_lnV = 0.9913849621042616
# time_constant_ms = 867.8180510073582 ms
