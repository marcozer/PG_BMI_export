"""Plots for BMI spline models: dose-response + calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.dataset import build_dataset
from lib.plotting import NordWhiteTheme, apply_theme

OUTPUT_DIR = Path("analysis/01_bmi_risk/outputs")
DOSE_FIG = OUTPUT_DIR / "bmi_bp_dose_response.png"
DOSE_FIG_SVG = OUTPUT_DIR / "bmi_bp_dose_response.svg"
CALIB_FIG = OUTPUT_DIR / "bmi_bp_calibration.png"
CALIB_FIG_SVG = OUTPUT_DIR / "bmi_bp_calibration.svg"


def plot_dose_response(curve_path: Path) -> None:
    df = pd.read_csv(curve_path)
    theme = NordWhiteTheme()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(df["bmi"], df["pred"], color=theme.palette[0], label="Probabilité ajustée")
    ax.fill_between(df["bmi"], df["ci_low"], df["ci_high"], color=theme.palette[0], alpha=0.2, label="IC 95%")
    # Rug plot
    data = build_dataset()
    ax.plot(data["bmi"], [0.01] * len(data), "|", color=theme.palette[3], alpha=0.4)
    # WHO cutoffs
    for x in [25, 30, 35, 40]:
        ax.axvline(x, color=theme.palette[2], linestyle="--", alpha=0.6)
    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Probabilité de Best Performer")
    ax.set_title("Dose-réponse BMI → Best Performer", color=theme.title)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    apply_theme(ax, theme)
    fig.tight_layout()
    fig.savefig(DOSE_FIG, dpi=150, facecolor=theme.background)
    fig.savefig(DOSE_FIG_SVG, dpi=150, facecolor=theme.background)
    plt.close(fig)


def plot_calibration(calib_path: Path) -> None:
    df = pd.read_csv(calib_path)
    theme = NordWhiteTheme()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot([0, 1], [0, 1], color=theme.palette[2], linestyle="--", label="Idéal")
    ax.scatter(df["mean_pred"], df["observed"], color=theme.palette[0])
    ax.set_xlabel("Probabilité prédite")
    ax.set_ylabel("Probabilité observée")
    ax.set_title("Calibration Best Performer (déciles)", color=theme.title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    apply_theme(ax, theme)
    fig.tight_layout()
    fig.savefig(CALIB_FIG, dpi=150, facecolor=theme.background)
    fig.savefig(CALIB_FIG_SVG, dpi=150, facecolor=theme.background)
    plt.close(fig)


def main() -> None:
    plot_dose_response(OUTPUT_DIR / "bmi_bp_curve.csv")
    plot_calibration(OUTPUT_DIR / "bmi_bp_calibration.csv")
    print(f"Saved {DOSE_FIG}, {DOSE_FIG_SVG}, {CALIB_FIG}, {CALIB_FIG_SVG}")


if __name__ == "__main__":
    main()
