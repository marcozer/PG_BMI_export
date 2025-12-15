"""Plots for BMI spline models: dose-response + calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "analysis"))
from lib.dataset import build_dataset
from lib.plotting import NordWhiteTheme, apply_theme
import numpy as np
from matplotlib import transforms
from statsmodels.stats.proportion import proportion_confint
import argparse
import patsy
import statsmodels.api as sm
from scipy.stats import norm, chi2

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DOSE_FIG = OUTPUT_DIR / "bmi_bp_dose_response.png"
DOSE_FIG_SVG = OUTPUT_DIR / "bmi_bp_dose_response.svg"
CALIB_FIG = OUTPUT_DIR / "bmi_bp_calibration.png"
CALIB_FIG_SVG = OUTPUT_DIR / "bmi_bp_calibration.svg"
POOLED_PNG = OUTPUT_DIR / "bmi_bp_volume_pooled.png"
POOLED_SVG = OUTPUT_DIR / "bmi_bp_volume_pooled.svg"
POOLED_ANN_PNG = OUTPUT_DIR / "bmi_bp_volume_pooled_annotated.png"
POOLED_ANN_SVG = OUTPUT_DIR / "bmi_bp_volume_pooled_annotated.svg"


def apply_volume_mode(df: pd.DataFrame, mode: str = "tertiles") -> pd.DataFrame:
    df = df.copy()
    if mode in {"annual_threshold", "mipd_annual_threshold"}:
        if "year" not in df.columns and "ANNEE" in df.columns:
            df["year"] = pd.to_numeric(df["ANNEE"], errors="coerce")
        if mode == "annual_threshold":
            counts_cy = df.groupby(["CENTRE", "year"], observed=False)["CODE"].count().reset_index(name="vol_cy")
            mean_per_centre = counts_cy.groupby("CENTRE", observed=False)["vol_cy"].mean().reset_index(name="vol_mean_per_year")
            df = df.merge(mean_per_centre, on="CENTRE", how="left")
            df["centre_volume"] = df["vol_mean_per_year"].astype(float)
            bins = [-np.inf, 5, 10, np.inf]
        else:  # mipd_annual_threshold = total MIPD / number of years observed
            df["NOMBRE_MIPD"] = pd.to_numeric(df.get("NOMBRE_MIPD"), errors="coerce")
            mipd_total = (
                df.groupby("CENTRE", observed=False)["NOMBRE_MIPD"]
                .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
                .reset_index(name="mipd_total")
            )
            years_per_centre = df.groupby("CENTRE", observed=False)["year"].nunique().reset_index(name="n_years")
            mean_per_centre = mipd_total.merge(years_per_centre, on="CENTRE", how="left")
            mean_per_centre["mipd_mean_per_year"] = mean_per_centre["mipd_total"] / mean_per_centre["n_years"]
            df = df.merge(mean_per_centre[["CENTRE", "mipd_mean_per_year"]], on="CENTRE", how="left")
            df["centre_volume"] = df["mipd_mean_per_year"].astype(float)
            bins = [-np.inf, 10, 20, np.inf]
        labels = ["Low", "Mid", "High"]
        df["centre_volume_cat"] = pd.cut(df["centre_volume"], bins=bins, labels=labels, right=True, include_lowest=True)
    else:
        df["centre_volume"] = df.groupby("CENTRE")["CODE"].transform("count")
        try:
            df["centre_volume_cat"] = pd.qcut(df["centre_volume"], q=[0, 0.33, 0.66, 1.0], labels=["Low", "Mid", "High"], duplicates="drop")
        except Exception:
            df["centre_volume_cat"] = pd.cut(df["centre_volume"], bins=3, labels=["Low", "Mid", "High"]) 
    return df


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


def plot_volume_pooled(df: pd.DataFrame) -> None:
    theme = NordWhiteTheme()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    palette = [theme.palette[0], theme.palette[2]]
    order = ["Low+Mid", "High"]

    # Build Low+Mid vs High
    df = df.dropna(subset=["bmi", "best_performer", "centre_volume_cat", "CENTRE"]).copy()
    df["best_performer"] = df["best_performer"].astype(int)
    df["tier2"] = df["centre_volume_cat"].map({"High": "High", "Mid": "Low+Mid", "Low": "Low+Mid"})

    # Rug
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    rug_levels = {"Low+Mid": -0.10, "High": -0.08}

    for idx, tier in enumerate(order):
        color = palette[idx]
        td = df[df["tier2"] == tier]
        ax.plot(td["bmi"], [rug_levels[tier]] * len(td), "|", color=color, alpha=0.8, markersize=9, linestyle="None", transform=trans, clip_on=False)

    # 2‑kg bins + Wilson CI + short smoothing
    bins = np.arange(15, 45, 2)
    df["bmi_bin"] = pd.cut(df["bmi"], bins=bins, right=False)
    agg = (
        df.groupby(["tier2", "bmi_bin"], observed=False)["best_performer"]
        .agg(["mean", "count", "sum"]).reset_index()
    )
    agg["bmi_mid"] = agg["bmi_bin"].apply(lambda x: x.left + 1 if pd.notna(x) else np.nan)
    for idx, tier in enumerate(order):
        color = palette[idx]
        grp = agg[agg["tier2"] == tier].dropna(subset=["bmi_mid"]).sort_values("bmi_mid")
        if grp.empty:
            continue
        means = grp["mean"].rolling(window=2, center=True, min_periods=1).mean()
        ci_low, ci_high = [], []
        for _, row in grp.iterrows():
            if row["count"] > 0:
                l, h = proportion_confint(count=row["sum"], nobs=row["count"], method="wilson")
            else:
                l, h = np.nan, np.nan
            ci_low.append(l); ci_high.append(h)
        ci_low = pd.Series(ci_low).rolling(window=2, center=True, min_periods=1).mean()
        ci_high = pd.Series(ci_high).rolling(window=2, center=True, min_periods=1).mean()
        ax.plot(grp["bmi_mid"], means, color=color, linewidth=2.5, label=tier)
        ax.fill_between(grp["bmi_mid"], ci_low, ci_high, color=color, alpha=0.12)

    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Taux de Best Performer")
    ax.set_title("BP vs BMI : High vs Low+Mid", color=theme.title)
    ax.set_xlim(18, 45)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(bottom=0.24)
    ax.legend(frameon=False, title="Volume")
    apply_theme(ax, theme)
    fig.tight_layout()
    fig.savefig(POOLED_PNG, dpi=300, facecolor=theme.background)
    fig.savefig(POOLED_SVG, dpi=300, facecolor=theme.background)
    plt.close(fig)

    # Annotated duplicates
    import shutil
    shutil.copyfile(POOLED_PNG, POOLED_ANN_PNG)
    shutil.copyfile(POOLED_SVG, POOLED_ANN_SVG)

    # Statistical comparison every 5 kg/m² (High vs Low+Mid) with Holm correction
    tests_csv = OUTPUT_DIR / "volume_pooled_bin_tests.csv"
    # Requested bins: [15–20), [20–25), [25–30), [30–35), [35–40), [40–45)
    bins5 = np.array([15, 20, 25, 30, 35, 40, 45])
    df["bmi_bin_test"] = pd.cut(df["bmi"], bins=bins5, right=False)
    agg5 = (
        df.groupby(["tier2", "bmi_bin_test"], observed=False)["best_performer"]
        .agg(["mean", "count", "sum"]).reset_index()
    )
    out = []
    for b in agg5["bmi_bin_test"].dropna().unique():
        hi = agg5[(agg5["tier2"] == "High") & (agg5["bmi_bin_test"] == b)]
        lo = agg5[(agg5["tier2"] == "Low+Mid") & (agg5["bmi_bin_test"] == b)]
        if hi.empty or lo.empty:
            continue
        p_h, n_h = float(hi["mean"].iloc[0]), int(hi["count"].iloc[0])
        p_l, n_l = float(lo["mean"].iloc[0]), int(lo["count"].iloc[0])
        if n_h < 10 or n_l < 20:
            continue
        diff = p_h - p_l
        # Standard error under independence (two-sample proportions)
        se = np.sqrt(p_h * (1 - p_h) / n_h + p_l * (1 - p_l) / n_l)
        z = diff / se if se > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        out.append({
            "bmi_bin": str(b),
            "bmi_mid": float(b.left + (b.right - b.left) / 2),
            "n_high": n_h,
            "n_pool": n_l,
            "p_high": p_h,
            "p_pool": p_l,
            "diff": diff,
            "se": se,
            "p_raw": p,
        })
    tests_df = pd.DataFrame(out)
    if not tests_df.empty:
        from statsmodels.stats.multitest import multipletests
        tests_df["p_adj"] = multipletests(tests_df["p_raw"], method="holm")[1]
        tests_df.to_csv(tests_csv, index=False)
        # annotate stars and p-values on the pooled figure
        # reload axis to add overlay
        fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=150)
        # Replot curves quickly for context
        for idx, tier in enumerate(order):
            color = palette[idx]
            grp = agg[agg["tier2"] == tier].dropna(subset=["bmi_mid"]).sort_values("bmi_mid")
            means = grp["mean"].rolling(window=2, center=True, min_periods=1).mean()
            ci_low = grp.apply(lambda r: proportion_confint(r["sum"], r["count"], method="wilson")[0] if r["count"]>0 else np.nan, axis=1).rolling(window=2, center=True, min_periods=1).mean()
            ci_high = grp.apply(lambda r: proportion_confint(r["sum"], r["count"], method="wilson")[1] if r["count"]>0 else np.nan, axis=1).rolling(window=2, center=True, min_periods=1).mean()
            ax2.plot(grp["bmi_mid"], means, color=color, linewidth=2.5, label=tier)
            ax2.fill_between(grp["bmi_mid"], ci_low, ci_high, color=color, alpha=0.12)
        # Add rug again
        trans2 = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        for idx, tier in enumerate(order):
            color = palette[idx]
            td = df[df["tier2"] == tier]
            y_rug = rug_levels[tier]
            ax2.plot(td["bmi"], [y_rug] * len(td), "|", color=color, alpha=0.8, markersize=9, linestyle="None", transform=trans2, clip_on=False)
        # Stars and p-values
        sig = tests_df.sort_values("bmi_mid")
        for _, r in sig.iterrows():
            x = r["bmi_mid"]
            pval = r["p_adj"] if np.isfinite(r["p_adj"]) else r["p_raw"]
            if np.isfinite(pval) and pval < 0.05:
                ax2.plot([x], [0.6], marker="*", color=palette[1], clip_on=False)
            if np.isfinite(pval):
                ax2.text(x, 0.62, f"p={pval:.3f}", ha="center", va="bottom", fontsize=8, color=palette[1])
        ax2.set_xlabel("BMI (kg/m²)")
        ax2.set_ylabel("Taux de Best Performer")
        ax2.set_title("BP vs BMI : High vs Low+Mid (tests par tranches)", color=theme.title)
        ax2.set_xlim(18, 45)
        ax2.set_ylim(0, 1)
        fig2.subplots_adjust(bottom=0.24)
        ax2.legend(frameon=False, title="Volume")
        apply_theme(ax2, theme)
        fig2.tight_layout()
        fig2.savefig(POOLED_ANN_PNG, dpi=300, facecolor=theme.background)
        fig2.savefig(POOLED_ANN_SVG, dpi=300, facecolor=theme.background)
        plt.close(fig2)
    else:
        pd.DataFrame(columns=["bmi_bin","bmi_mid","n_high","n_pool","p_high","p_pool","diff","se","p_raw","p_adj"]).to_csv(tests_csv, index=False)

    # Global interaction test (spline × tier), cluster‑robuste
    global_csv = OUTPUT_DIR / "volume_pooled_global_test.csv"
    df_model = df.copy()
    formula_int = "best_performer ~ bs(bmi, df=4, include_intercept=False) * C(tier2)"
    formula_add = "best_performer ~ bs(bmi, df=4, include_intercept=False) + C(tier2)"
    y_int, X_int = patsy.dmatrices(formula_int, df_model, return_type="dataframe")
    y_add, X_add = patsy.dmatrices(formula_add, df_model, return_type="dataframe")
    y_int = y_int.iloc[:, 0]; y_add = y_add.iloc[:, 0]
    groups = df_model.loc[y_int.index, "CENTRE"].astype("category").cat.codes.to_numpy()
    glm_int = sm.GLM(y_int, X_int, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    glm_add = sm.GLM(y_add, X_add, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    stat = 2 * (glm_int.llf - glm_add.llf)
    df_diff = len(glm_int.params) - len(glm_add.params)
    p_lrt = chi2.sf(stat, df_diff) if df_diff > 0 else np.nan
    pd.DataFrame([{ "model_int_llf": glm_int.llf, "model_add_llf": glm_add.llf, "df_diff": df_diff, "lrt_stat": stat, "p_lrt": p_lrt }]).to_csv(global_csv, index=False)


def main(volume_tier_mode: str = "tertiles") -> None:
    plot_dose_response(OUTPUT_DIR / "bmi_bp_curve.csv")
    plot_calibration(OUTPUT_DIR / "bmi_bp_calibration.csv")
    # Build dataset for descriptive pooled figure with requested volume mode
    df = build_dataset()
    df = apply_volume_mode(df, mode=volume_tier_mode)
    plot_volume_pooled(df)
    print(f"Saved {DOSE_FIG}, {DOSE_FIG_SVG}, {CALIB_FIG}, {CALIB_FIG_SVG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMI dose-response + pooled volume plots")
    parser.add_argument("--volume-tier-mode", choices=["tertiles", "annual_threshold", "mipd_annual_threshold"], default="tertiles")
    args = parser.parse_args()
    main(volume_tier_mode=args.volume_tier_mode)
