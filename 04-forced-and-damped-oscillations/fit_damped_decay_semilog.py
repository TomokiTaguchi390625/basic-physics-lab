
# このように初めは読み取った。
# 時間(ms) 電圧(V)
# 230ms 1.6V
# 400ms 1.2V
# 750ms 0.8V
# 1960ms 0.25V
# 2320ms 0.10V

# しかし、時間が経つにつれ差分が小さくなって読み取り誤差が大きくなるのではないかと考えた。
# 時間(ms) 電圧(V)
# 230ms 1.6V
# 400ms 1.2V
# 750ms 0.8V
# 1000ms 0.65V
# 1200ms 0.50V

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

    results.append({
        "dataset": d["name"],
        "ln_slope_per_ms": b,
        "ln_slope_per_s": b * 1000,
        "log10_slope_per_ms": b / np.log(10),
        "log10_slope_per_s": b * 1000 / np.log(10),
        "intercept_ln": a,
        "V0": np.exp(a),
        "R2_on_lnV": r2,
        "time_constant_ms": -1 / b,
    })

for r in results:
    print(r)