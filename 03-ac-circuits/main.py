import numpy as np
from scipy.optimize import curve_fit

# 測定データ
f = np.array([10, 25, 35, 42, 46, 48, 50, 52, 54, 58, 65, 75, 90, 110, 150], dtype=float) # kHz
y = np.array([0.06097, 0.19512, 0.37073, 0.58536, 0.73170,
              0.85, 0.85, 0.8625, 0.80, 0.65, 0.4875,
              0.34, 0.24, 0.17, 0.1025], dtype=float)

# 半値幅法
peak = y.max()
half = peak / np.sqrt(2)

# 低周波側：42 kHz と 46 kHz の間
f1 = 42 + (half - 0.58536) * (46 - 42) / (0.73170 - 0.58536)

# 高周波側：58 kHz と 65 kHz の間
f2 = 58 + (half - 0.65) * (65 - 58) / (0.4875 - 0.65)

f0_exp = 50.4
Q_half = f0_exp / (f2 - f1)

print("half =", half)
print("f1 =", f1)
print("f2 =", f2)
print("Q_half =", Q_half)

# 振幅特性フィッティング
def amp_model(f, A, f0, Q):
    x = f / f0
    return A / np.sqrt(1 + (Q * (x - 1/x))**2)

popt, pcov = curve_fit(
    amp_model,
    f,
    y,
    p0=[0.86, 50.4, 3.0],
    bounds=([0, 40, 0.1], [2, 70, 20])
)

A_fit, f0_fit, Q_fit = popt

print("A_fit =", A_fit)
print("f0_fit =", f0_fit)
print("Q_fit =", Q_fit)

# 振幅特性フィッティングの残差
y_fit = amp_model(f, A_fit, f0_fit, Q_fit)
residual = y - y_fit

rmse = np.sqrt(np.mean(residual**2))
mae = np.mean(np.abs(residual))
max_abs_error = np.max(np.abs(residual))

print("amp RMSE =", rmse)
print("amp MAE =", mae)
print("amp residuals =", residual)

# 位相差フィッティング
phase = np.array([0.25, 0.225, 0.1785, 0.125, 0.088,
                  0.0394, 0, -0.039, -0.067, -0.114,
                  -0.145, -0.188, -0.205, -0.22, -0.24], dtype=float)

def phase_model(f, f0, Q):
    x = f / f0
    return -np.arctan(Q * (x - 1/x)) / (2*np.pi)

popt_phase, pcov_phase = curve_fit(
    phase_model,
    f,
    phase,
    p0=[50.4, 3.0],
    bounds=([40, 0.1], [70, 20])
)

f0_phase, Q_phase = popt_phase

print("f0_phase =", f0_phase)
print("Q_phase =", Q_phase)

# 理論値
L = 10e-3
C = 0.001e-6
R = 1050

f0_theory = 1 / (2*np.pi*np.sqrt(L*C))
Q_theory = (1/R) * np.sqrt(L/C)
gamma_theory = R / (2*L)

print("f0_theory =", f0_theory / 1000, "kHz")
print("Q_theory =", Q_theory)
print("gamma_theory =", gamma_theory)

# 減衰振動
T = 21e-6
a1 = 0.90
a2 = 0.30

gamma = (1/T) * np.log(a1/a2)
Q_decay = (2*np.pi*50.4e3) / (2*gamma)

print("gamma =", gamma)
print("Q_decay =", Q_decay)

# 出力結果:
# half = 0.6098795987733973
# f1 = 42.670209068563544
# f2 = 59.728263437453656
# Q_half = 2.9546159784739445
# A_fit = 0.8606544169461465
# f0_fit = 50.46893845207287
# Q_fit = 2.9023791912961707
# amp RMSE = 0.012171333154842716
# amp MAE = 0.0092426662601289
# amp residuals = [-3.29699283e-05  5.26283080e-03  1.08447954e-02 -1.63280006e-03
#  -2.59207026e-02  2.36840830e-02 -9.39358003e-03  1.45147814e-02
#  -1.05685436e-03 -1.87965415e-02  6.65967361e-03  4.22259397e-03
#   6.53316150e-03  1.02554657e-03 -9.05907908e-03]
# f0_phase = 50.173303364177315
# Q_phase = 2.9687420273592435
# f0_theory = 50.329212104487034 kHz
# Q_theory = 3.011693009684171
# gamma_theory = 52500.0
# gamma = 52314.87088895761
# Q_decay = 3.0266015580351264