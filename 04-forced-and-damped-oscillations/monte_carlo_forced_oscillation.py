"""Monte Carlo uncertainty analysis for experiment B forced oscillation data.

The default uncertainty model is intentionally simple:

- frequency: uniform +/- 0.005 Hz
- CH2 amplitude: uniform +/- 0.02 V
- phase: uniform +/- 1 deg

Change the constants near the top if the instrument resolution or reading
uncertainty should be treated differently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


N_TRIALS = 200_000
RNG_SEED = 20260512

FREQ_HALF_WIDTH_HZ = 0.005
AMP_HALF_WIDTH_V = 0.02
PHASE_HALF_WIDTH_DEG = 1.0

PHASE_RESONANCE_DEG = 90.0

# 実行結果メモ
#
# 実行コマンド:
#   uv run --with numpy python 04-forced-and-damped-oscillations/monte_carlo_forced_oscillation.py
#
# モンテカルロ設定:
#   試行回数: 200000
#   乱数シード: 20260512
#   周波数の読み取り不確かさ: 一様分布 +/- 0.005 Hz
#   振幅の読み取り不確かさ: 一様分布 +/- 0.02 V
#   位相の読み取り不確かさ: 一様分布 +/- 1.0 deg
#
# 線形補間からの名目値:
#   peak frequency           = 440.22000 Hz
#   quadratic peak frequency = 440.21580 Hz
#   peak amplitude           = 3.68000 V
#   half-power amplitude     = 2.60215 V
#   f_minus                  = 440.02532 Hz
#   f_plus                   = 440.40789 Hz
#   bandwidth                = 0.38257 Hz
#   gamma_B                  = 1.20186 s^-1
#   Q                        = 1150.70
#   half-width center        = 440.21661 Hz
#   phase 90 deg frequency   = 440.24150 Hz
#   phase at f_minus         = 143.41 deg
#   phase at f_plus          = 49.78 deg
#
# モンテカルロ結果:
#   f_minus                  = 440.02533 +/- 0.00270 Hz
#                              95% CI: 440.02008 to 440.03058 Hz
#   f_plus                   = 440.40786 +/- 0.00282 Hz
#                              95% CI: 440.40226 to 440.41303 Hz
#   bandwidth                = 0.38253 +/- 0.00425 Hz
#                              95% CI: 0.37412 to 0.39064 Hz
#   gamma_B                  = 1.20175 +/- 0.01334 s^-1
#                              95% CI: 1.17533 to 1.22722 s^-1
#   Q                        = 1150.94 +/- 12.79
#                              95% CI: 1126.92 to 1176.67
#   half-width center        = 440.21659 +/- 0.00176 Hz
#                              95% CI: 440.21314 to 440.21999 Hz
#   phase 90 deg frequency   = 440.24156 +/- 0.00267 Hz
#                              95% CI: 440.23661 to 440.24676 Hz
#   quadratic peak frequency = 440.21577 +/- 0.00213 Hz
#                              95% CI: 440.21162 to 440.21988 Hz
#   phase at f_minus         = 143.41 +/- 0.46 deg
#                              95% CI: 142.53 to 144.29 deg
#   phase at f_plus          = 49.78 +/- 0.51 deg
#                              95% CI: 48.86 to 50.76 deg


FREQ_HZ = np.array(
    [
        439.89,
        439.92,
        439.95,
        439.98,
        440.01,
        440.04,
        440.07,
        440.10,
        440.13,
        440.16,
        440.19,
        440.22,
        440.25,
        440.28,
        440.31,
        440.34,
        440.37,
        440.40,
        440.43,
        440.46,
        440.49,
        440.52,
        440.55,
    ],
    dtype=float,
)

AMP_V = np.array(
    [
        1.86,
        2.02,
        2.16,
        2.32,
        2.500,
        2.700,
        2.920,
        3.160,
        3.340,
        3.520,
        3.640,
        3.680,
        3.600,
        3.480,
        3.320,
        3.080,
        2.880,
        2.660,
        2.440,
        2.260,
        2.100,
        1.980,
        1.820,
    ],
    dtype=float,
)

PHASE_DEG = np.array(
    [
        158.6,
        154.9,
        152.7,
        149.0,
        145.2,
        141.7,
        135.7,
        129.0,
        122.6,
        115.7,
        105.7,
        96.55,
        87.41,
        79.08,
        70.89,
        63.45,
        56.94,
        50.86,
        46.75,
        42.88,
        39.56,
        38.10,
        35.04,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class DerivedValues:
    peak_frequency_hz: float
    quadratic_peak_frequency_hz: float
    peak_amplitude_v: float
    half_power_amplitude_v: float
    f_minus_hz: float
    f_plus_hz: float
    bandwidth_hz: float
    gamma_s_inv: float
    q_factor: float
    half_width_center_hz: float
    phase_90_frequency_hz: float
    phase_at_f_minus_deg: float
    phase_at_f_plus_deg: float


def interpolate_y_at_x(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    return float(np.interp(x0, x, y))


def crossing_by_linear_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    level: float,
    peak_index: int,
    side: str,
) -> float:
    """Find the nearest level crossing around the peak."""
    if side == "left":
        indices = range(peak_index - 1, -1, -1)
        for i in indices:
            j = i + 1
            if (y[i] - level) * (y[j] - level) <= 0:
                return float(x[i] + (level - y[i]) * (x[j] - x[i]) / (y[j] - y[i]))
    elif side == "right":
        indices = range(peak_index, len(y) - 1)
        for i in indices:
            j = i + 1
            if (y[i] - level) * (y[j] - level) <= 0:
                return float(x[i] + (level - y[i]) * (x[j] - x[i]) / (y[j] - y[i]))
    else:
        raise ValueError(f"unknown side: {side}")

    return math.nan


def phase_crossing_frequency(
    x: np.ndarray,
    phase_deg: np.ndarray,
    level_deg: float = PHASE_RESONANCE_DEG,
) -> float:
    y = phase_deg - level_deg
    for i in range(len(y) - 1):
        j = i + 1
        if y[i] * y[j] <= 0:
            return float(x[i] + (level_deg - phase_deg[i]) * (x[j] - x[i]) / (phase_deg[j] - phase_deg[i]))
    return math.nan


def quadratic_peak_frequency(x: np.ndarray, y: np.ndarray, peak_index: int, radius: int = 2) -> float:
    """Estimate the peak by fitting a local quadratic around the largest point."""
    start = max(0, peak_index - radius)
    stop = min(len(x), peak_index + radius + 1)
    x_fit = x[start:stop]
    y_fit = y[start:stop]
    if len(x_fit) < 3:
        return float(x[peak_index])

    x0 = x[peak_index]
    a, b, _ = np.polyfit(x_fit - x0, y_fit, deg=2)
    if a >= 0:
        return float(x[peak_index])
    return float(x0 - b / (2 * a))


def derive_values(freq_hz: np.ndarray, amp_v: np.ndarray, phase_deg: np.ndarray) -> DerivedValues:
    order = np.argsort(freq_hz)
    freq_hz = freq_hz[order]
    amp_v = amp_v[order]
    phase_deg = phase_deg[order]

    peak_index = int(np.argmax(amp_v))
    peak_frequency_hz = float(freq_hz[peak_index])
    peak_amplitude_v = float(amp_v[peak_index])
    half_power_amplitude_v = peak_amplitude_v / math.sqrt(2)

    f_minus_hz = crossing_by_linear_interpolation(
        freq_hz,
        amp_v,
        half_power_amplitude_v,
        peak_index,
        side="left",
    )
    f_plus_hz = crossing_by_linear_interpolation(
        freq_hz,
        amp_v,
        half_power_amplitude_v,
        peak_index,
        side="right",
    )
    bandwidth_hz = f_plus_hz - f_minus_hz
    gamma_s_inv = math.pi * bandwidth_hz
    half_width_center_hz = 0.5 * (f_minus_hz + f_plus_hz)
    q_factor = half_width_center_hz / bandwidth_hz

    phase_90_frequency_hz = phase_crossing_frequency(freq_hz, phase_deg)
    quadratic_peak_frequency_hz = quadratic_peak_frequency(freq_hz, amp_v, peak_index)
    phase_at_f_minus_deg = interpolate_y_at_x(freq_hz, phase_deg, f_minus_hz)
    phase_at_f_plus_deg = interpolate_y_at_x(freq_hz, phase_deg, f_plus_hz)

    return DerivedValues(
        peak_frequency_hz=peak_frequency_hz,
        quadratic_peak_frequency_hz=quadratic_peak_frequency_hz,
        peak_amplitude_v=peak_amplitude_v,
        half_power_amplitude_v=half_power_amplitude_v,
        f_minus_hz=f_minus_hz,
        f_plus_hz=f_plus_hz,
        bandwidth_hz=bandwidth_hz,
        gamma_s_inv=gamma_s_inv,
        q_factor=q_factor,
        half_width_center_hz=half_width_center_hz,
        phase_90_frequency_hz=phase_90_frequency_hz,
        phase_at_f_minus_deg=phase_at_f_minus_deg,
        phase_at_f_plus_deg=phase_at_f_plus_deg,
    )


def run_monte_carlo() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    samples: dict[str, list[float]] = {
        "f_minus_hz": [],
        "f_plus_hz": [],
        "bandwidth_hz": [],
        "gamma_s_inv": [],
        "q_factor": [],
        "half_width_center_hz": [],
        "phase_90_frequency_hz": [],
        "quadratic_peak_frequency_hz": [],
        "phase_at_f_minus_deg": [],
        "phase_at_f_plus_deg": [],
    }

    for _ in range(N_TRIALS):
        freq = FREQ_HZ + rng.uniform(-FREQ_HALF_WIDTH_HZ, FREQ_HALF_WIDTH_HZ, size=FREQ_HZ.shape)
        amp = AMP_V + rng.uniform(-AMP_HALF_WIDTH_V, AMP_HALF_WIDTH_V, size=AMP_V.shape)
        phase = PHASE_DEG + rng.uniform(-PHASE_HALF_WIDTH_DEG, PHASE_HALF_WIDTH_DEG, size=PHASE_DEG.shape)

        values = derive_values(freq, amp, phase)
        for key in samples:
            value = getattr(values, key)
            if math.isfinite(value):
                samples[key].append(value)

    return {key: np.array(value, dtype=float) for key, value in samples.items()}


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "p2.5": float(np.percentile(values, 2.5)),
        "p16": float(np.percentile(values, 16)),
        "median": float(np.percentile(values, 50)),
        "p84": float(np.percentile(values, 84)),
        "p97.5": float(np.percentile(values, 97.5)),
    }


def print_value(name: str, value: float, unit: str = "", decimals: int = 6) -> None:
    suffix = f" {unit}" if unit else ""
    print(f"{name:32s}: {value:.{decimals}f}{suffix}")


def print_summary(name: str, values: np.ndarray, unit: str = "", decimals: int = 6) -> None:
    stats = summarize(values)
    suffix = f" {unit}" if unit else ""
    print(
        f"{name:32s}: "
        f"{stats['mean']:.{decimals}f} +/- {stats['std']:.{decimals}f}{suffix} "
        f"(95% CI: {stats['p2.5']:.{decimals}f} to {stats['p97.5']:.{decimals}f}{suffix}; "
        f"median: {stats['median']:.{decimals}f}{suffix})"
    )


def main() -> None:
    nominal = derive_values(FREQ_HZ, AMP_V, PHASE_DEG)

    print("Nominal values from linear interpolation")
    print("----------------------------------------")
    print_value("peak frequency", nominal.peak_frequency_hz, "Hz", 5)
    print_value("quadratic peak frequency", nominal.quadratic_peak_frequency_hz, "Hz", 5)
    print_value("peak amplitude", nominal.peak_amplitude_v, "V", 5)
    print_value("half-power amplitude", nominal.half_power_amplitude_v, "V", 5)
    print_value("f_minus", nominal.f_minus_hz, "Hz", 5)
    print_value("f_plus", nominal.f_plus_hz, "Hz", 5)
    print_value("bandwidth", nominal.bandwidth_hz, "Hz", 5)
    print_value("gamma_B", nominal.gamma_s_inv, "s^-1", 5)
    print_value("Q", nominal.q_factor, "", 2)
    print_value("half-width center", nominal.half_width_center_hz, "Hz", 5)
    print_value("phase 90 deg frequency", nominal.phase_90_frequency_hz, "Hz", 5)
    print_value("phase at f_minus", nominal.phase_at_f_minus_deg, "deg", 2)
    print_value("phase at f_plus", nominal.phase_at_f_plus_deg, "deg", 2)

    print()
    print("Monte Carlo settings")
    print("--------------------")
    print(f"trials                          : {N_TRIALS}")
    print(f"random seed                     : {RNG_SEED}")
    print(f"frequency uncertainty           : uniform +/- {FREQ_HALF_WIDTH_HZ} Hz")
    print(f"amplitude uncertainty           : uniform +/- {AMP_HALF_WIDTH_V} V")
    print(f"phase uncertainty               : uniform +/- {PHASE_HALF_WIDTH_DEG} deg")

    samples = run_monte_carlo()

    print()
    print("Monte Carlo results")
    print("-------------------")
    print_summary("f_minus", samples["f_minus_hz"], "Hz", 5)
    print_summary("f_plus", samples["f_plus_hz"], "Hz", 5)
    print_summary("bandwidth", samples["bandwidth_hz"], "Hz", 5)
    print_summary("gamma_B", samples["gamma_s_inv"], "s^-1", 5)
    print_summary("Q", samples["q_factor"], "", 2)
    print_summary("half-width center", samples["half_width_center_hz"], "Hz", 5)
    print_summary("phase 90 deg frequency", samples["phase_90_frequency_hz"], "Hz", 5)
    print_summary("quadratic peak frequency", samples["quadratic_peak_frequency_hz"], "Hz", 5)
    print_summary("phase at f_minus", samples["phase_at_f_minus_deg"], "deg", 2)
    print_summary("phase at f_plus", samples["phase_at_f_plus_deg"], "deg", 2)


if __name__ == "__main__":
    main()
