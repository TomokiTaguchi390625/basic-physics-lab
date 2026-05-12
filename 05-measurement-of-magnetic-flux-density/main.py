"""Magnetic flux density analysis for experiment 5.

This script keeps the measured values in one place and regenerates the
figures/CSV tables used in the report notes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTDIR = Path(__file__).resolve().parent
N_MC = 200_000
RNG_SEED = 20260512

RADIUS_MM = 25.0
DELTA_H_MM = 0.4

MU0 = 4 * np.pi * 1e-7
N_TURNS = 100
CURRENT_A = 1.0
RADIUS_M = RADIUS_MM / 1000


theta_deg = np.array([0, 45, 90, 135, 180], dtype=float)
point_a_mT = np.array([0.0180, 0.1439, 0.1761, 0.1296, -0.0261], dtype=float)
point_b_mT = np.array([0.0045, 0.0334, 0.0450, 0.0320, -0.0037], dtype=float)

ruler_mm = np.array(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
    dtype=float,
)
h_mm = ruler_mm + 5.4

coil_exp_mT = np.array(
    [
        2.3255,
        1.5583,
        0.8133,
        0.4860,
        0.2830,
        0.1755,
        0.1142,
        0.0764,
        0.0542,
        0.0398,
        0.0300,
        0.0182,
        0.0117,
        0.0097,
        0.0054,
        0.0029,
    ],
    dtype=float,
)
coil_th_mT = np.array(
    [
        2.3471,
        1.5512,
        0.8675,
        0.4825,
        0.2821,
        0.1749,
        0.1144,
        0.0783,
        0.0557,
        0.0409,
        0.0309,
        0.0188,
        0.0122,
        0.0084,
        0.0060,
        0.0044,
    ],
    dtype=float,
)
magnet_mT = np.array(
    [
        8.488,
        35.94,
        23.02,
        13.121,
        7.692,
        4.791,
        3.140,
        2.139,
        1.5204,
        1.1157,
        0.8405,
        0.5083,
        0.3300,
        0.2257,
        0.1610,
        0.1185,
    ],
    dtype=float,
)


def fit_angle(y_mT: np.ndarray) -> dict[str, float | np.ndarray]:
    """Fit B(theta) = Bx cos(theta) + By sin(theta) + C."""
    rad = np.deg2rad(theta_deg)
    design = np.column_stack([np.cos(rad), np.sin(rad), np.ones_like(rad)])
    bx_mT, by_mT, offset_mT = np.linalg.lstsq(design, y_mT, rcond=None)[0]
    fitted = design @ np.array([bx_mT, by_mT, offset_mT])
    return {
        "Bx_mT": bx_mT,
        "By_mT": by_mT,
        "offset_C_mT": offset_mT,
        "amplitude_mT": float(np.hypot(bx_mT, by_mT)),
        "direction_deg": float(np.rad2deg(np.arctan2(by_mT, bx_mT)) % 360),
        "RMSE_mT": float(np.sqrt(np.mean((y_mT - fitted) ** 2))),
        "fitted": fitted,
    }


def coil_theory_from_h_mm(h_mm_array: np.ndarray) -> np.ndarray:
    h_m = np.asarray(h_mm_array) / 1000
    b_T = MU0 * N_TURNS * CURRENT_A * RADIUS_M**2
    b_T /= 2 * (h_m**2 + RADIUS_M**2) ** 1.5
    return b_T * 1000


def power_fit(
    source: str, field_mT: np.ndarray, mask: np.ndarray, label: str
) -> dict[str, float | str | int]:
    x = np.log(h_mm[mask])
    y = np.log(field_mT[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = intercept + slope * x
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "source": source,
        "fit_range": label,
        "n_points": int(mask.sum()),
        "intercept": float(intercept),
        "slope_beta": float(slope),
        "R2_loglog": float(1 - ss_res / ss_tot),
    }


def write_plots(
    fit_a: dict[str, float | np.ndarray],
    fit_b: dict[str, float | np.ndarray],
    coil_df: pd.DataFrame,
    coil_m_eff: np.ndarray,
    magnet_m_eff: np.ndarray,
    coil_m_theory: float,
) -> list[Path]:
    paths: list[Path] = []

    dense_theta = np.linspace(0, 180, 361)
    dense_rad = np.deg2rad(dense_theta)
    design_dense = np.column_stack(
        [np.cos(dense_rad), np.sin(dense_rad), np.ones_like(dense_rad)]
    )

    plt.figure(figsize=(7, 4.8))
    plt.scatter(theta_deg, point_a_mT, marker="o", label="A measured")
    plt.plot(
        dense_theta,
        design_dense @ np.array([fit_a["Bx_mT"], fit_a["By_mT"], fit_a["offset_C_mT"]]),
        label="A fit",
    )
    plt.scatter(theta_deg, point_b_mT, marker="x", label="B measured")
    plt.plot(
        dense_theta,
        design_dense @ np.array([fit_b["Bx_mT"], fit_b["By_mT"], fit_b["offset_C_mT"]]),
        label="B fit",
    )
    plt.xlabel("Probe angle theta (deg)")
    plt.ylabel("Measured magnetic flux density (mT)")
    plt.title("Experiment A: angular dependence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    paths.append(OUTDIR / "experiment_A_angular_fit.png")
    plt.savefig(paths[-1], dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    plt.plot(h_mm, coil_exp_mT, marker="o", label="Coil measured")
    plt.plot(h_mm, coil_th_mT, marker="x", label="Coil theory")
    plt.fill_between(
        h_mm,
        coil_df["MC_95_low_mT"],
        coil_df["MC_95_high_mT"],
        alpha=0.25,
        label="+/-0.4 mm h MC 95%",
    )
    plt.xlabel("Distance h (mm)")
    plt.ylabel("Magnetic flux density (mT)")
    plt.title("Experiment B: coil measured vs theory")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    paths.append(OUTDIR / "experiment_B_coil_measured_theory_MC.png")
    plt.savefig(paths[-1], dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    plt.axhline(0, linestyle="--", color="black", linewidth=1)
    plt.plot(
        h_mm, coil_df["relative_error_percent"], marker="o", label="Measured - theory"
    )
    plt.plot(
        h_mm,
        coil_df["distance_uncertainty_percent_if_dh_0p4mm"],
        marker="x",
        label="+ distance uncertainty",
    )
    plt.plot(
        h_mm,
        -coil_df["distance_uncertainty_percent_if_dh_0p4mm"],
        marker="x",
        label="- distance uncertainty",
    )
    plt.xlabel("Distance h (mm)")
    plt.ylabel("Relative difference / uncertainty (%)")
    plt.title("Experiment B: relative error and distance uncertainty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    paths.append(OUTDIR / "experiment_B_coil_relative_error_uncertainty.png")
    plt.savefig(paths[-1], dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    plt.loglog(h_mm, coil_exp_mT, marker="o", label="Coil measured")
    plt.loglog(h_mm, coil_th_mT, marker="x", label="Coil theory")
    plt.loglog(h_mm, magnet_mT, marker="^", label="Magnet measured")
    plt.xlabel("Distance h (mm)")
    plt.ylabel("Magnetic flux density (mT)")
    plt.title("Experiment B: log-log plot")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    paths.append(OUTDIR / "experiment_B_loglog_coil_magnet.png")
    plt.savefig(paths[-1], dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    plt.plot(h_mm, coil_m_eff, marker="o", label="Coil effective m")
    plt.axhline(coil_m_theory, linestyle="--", label="Coil theoretical m")
    plt.plot(h_mm, magnet_m_eff, marker="^", label="Magnet effective m")
    plt.xlabel("Distance h (mm)")
    plt.ylabel("Effective magnetic moment (A m^2)")
    plt.title("Far-field effective magnetic moment estimate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    paths.append(OUTDIR / "effective_magnetic_moment.png")
    plt.savefig(paths[-1], dpi=200)
    plt.close()

    return paths


def main() -> None:
    fit_a = fit_angle(point_a_mT)
    fit_b = fit_angle(point_b_mT)

    rel_err_percent = (coil_exp_mT - coil_th_mT) / coil_th_mT * 100
    rel_unc_from_h_percent = (3 * h_mm / (h_mm**2 + RADIUS_MM**2) * DELTA_H_MM) * 100

    rng = np.random.default_rng(RNG_SEED)
    h_samples = h_mm[:, None] + rng.uniform(
        -DELTA_H_MM, DELTA_H_MM, size=(len(h_mm), N_MC)
    )
    b_samples = coil_theory_from_h_mm(h_samples)
    mc_low = np.percentile(b_samples, 2.5, axis=1)
    mc_high = np.percentile(b_samples, 97.5, axis=1)

    coil_df = pd.DataFrame(
        {
            "h_mm": h_mm,
            "coil_exp_mT": coil_exp_mT,
            "coil_theory_mT": coil_th_mT,
            "residual_mT": coil_exp_mT - coil_th_mT,
            "relative_error_percent": rel_err_percent,
            "distance_uncertainty_percent_if_dh_0p4mm": rel_unc_from_h_percent,
            "abs_relative_error_over_distance_uncertainty": np.abs(rel_err_percent)
            / rel_unc_from_h_percent,
            "MC_theory_mean_mT": b_samples.mean(axis=1),
            "MC_95_low_mT": mc_low,
            "MC_95_high_mT": mc_high,
            "exp_inside_MC_95_h_only": (coil_exp_mT >= mc_low)
            & (coil_exp_mT <= mc_high),
        }
    )

    ranges = {
        "all": h_mm >= 0,
        "h >= 35.4 mm": h_mm >= 35.4,
        "h >= 55.4 mm": h_mm >= 55.4,
        "h >= 75.4 mm": h_mm >= 75.4,
        "h >= 105.4 mm": h_mm >= 105.4,
    }
    fit_rows = []
    for label, mask in ranges.items():
        fit_rows.append(power_fit("coil", coil_exp_mT, mask, label))
        fit_rows.append(power_fit("magnet", magnet_mT, mask, label))
    fit_df = pd.DataFrame(fit_rows)

    h_m = h_mm / 1000
    coil_m_eff = 2 * np.pi * (coil_exp_mT * 1e-3) * h_m**3 / MU0
    magnet_m_eff = 2 * np.pi * (magnet_mT * 1e-3) * h_m**3 / MU0
    coil_m_theory = N_TURNS * CURRENT_A * np.pi * RADIUS_M**2

    moment_df = pd.DataFrame(
        {
            "h_mm": h_mm,
            "coil_m_eff_A_m2": coil_m_eff,
            "magnet_m_eff_A_m2": magnet_m_eff,
            "coil_m_eff_over_theory": coil_m_eff / coil_m_theory,
            "magnet_over_coil_theory": magnet_m_eff / coil_m_theory,
        }
    )

    angle_df = pd.DataFrame(
        [
            {"point": "A", **{k: v for k, v in fit_a.items() if k != "fitted"}},
            {"point": "B", **{k: v for k, v in fit_b.items() if k != "fitted"}},
        ]
    )

    coil_df.to_csv(OUTDIR / "coil_error_uncertainty_table.csv", index=False)
    fit_df.to_csv(OUTDIR / "power_law_fit_table.csv", index=False)
    moment_df.to_csv(OUTDIR / "effective_moment_table.csv", index=False)
    plot_paths = write_plots(
        fit_a, fit_b, coil_df, coil_m_eff, magnet_m_eff, coil_m_theory
    )

    print("=== Experiment A fit ===")
    print(angle_df.round(5).to_string(index=False))

    print("\n=== Coil residual/uncertainty key rows ===")
    print(
        coil_df[
            [
                "h_mm",
                "coil_exp_mT",
                "coil_theory_mT",
                "relative_error_percent",
                "distance_uncertainty_percent_if_dh_0p4mm",
                "exp_inside_MC_95_h_only",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\n=== Power-law fits log(B)=a+b log(h) ===")
    print(
        fit_df[["source", "fit_range", "n_points", "slope_beta", "R2_loglog"]]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== Effective magnetic moment summary ===")
    print(f"coil_theoretical_moment_A_m2: {coil_m_theory}")
    print(
        f"coil_m_eff_median_far_75mm_plus_A_m2: {np.median(coil_m_eff[h_mm >= 75.4])}"
    )
    print(
        f"mag_m_eff_median_far_75mm_plus_A_m2: {np.median(magnet_m_eff[h_mm >= 75.4])}"
    )
    print(
        f"mag_over_coil_moment_median_far_75mm_plus: {np.median(magnet_m_eff[h_mm >= 75.4]) / coil_m_theory}"
    )

    print("\nSaved files:")
    for path in [
        *plot_paths,
        OUTDIR / "coil_error_uncertainty_table.csv",
        OUTDIR / "power_law_fit_table.csv",
        OUTDIR / "effective_moment_table.csv",
    ]:
        print(path)


if __name__ == "__main__":
    main()
