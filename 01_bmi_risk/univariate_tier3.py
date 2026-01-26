"""Univariate BP vs BMI by centre volume tier (Low/Mid/High).

This duplicates the pooled univariate approach (binning + descriptive smoothing + binwise tests)
but keeps three separate tiers instead of pooling Mid with Low.

Outputs are written to export/01_bmi_risk/outputs_univariate_tier3/<mode>/ so existing results
in export/01_bmi_risk/outputs/<mode>/ are not overwritten.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms

from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.dataset import build_dataset
from lib.plotting import NordWhiteTheme, apply_theme


BMI_MIN = 15
BMI_MAX = 45


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
        df["centre_volume_cat"] = pd.cut(
            df["centre_volume"],
            bins=bins,
            labels=["Low", "Mid", "High"],
            right=True,
            include_lowest=True,
        )
    else:
        df["centre_volume"] = df.groupby("CENTRE")["CODE"].transform("count")
        try:
            df["centre_volume_cat"] = pd.qcut(
                df["centre_volume"],
                q=[0, 0.33, 0.66, 1.0],
                labels=["Low", "Mid", "High"],
                duplicates="drop",
            )
        except Exception:
            df["centre_volume_cat"] = pd.cut(df["centre_volume"], bins=3, labels=["Low", "Mid", "High"])
    return df


def _smooth_reflect(values: pd.Series, kernel: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    pad = len(kernel) // 2
    if pad == 0 or v.size == 0:
        return v
    vpad = np.pad(v, pad_width=pad, mode="reflect")
    return np.convolve(vpad, kernel, mode="valid")


def build_bin_agg(df: pd.DataFrame) -> pd.DataFrame:
    bins2 = np.arange(BMI_MIN, BMI_MAX, 2)
    df = df.copy()
    df["bmi_bin"] = pd.cut(df["bmi"], bins=bins2, right=False)
    agg = (
        df.groupby(["centre_volume_cat", "bmi_bin"], observed=False)["best_performer"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    agg["bmi_mid"] = agg["bmi_bin"].apply(lambda x: float(x.left + 1) if pd.notna(x) else np.nan)
    return agg


def build_bin_tests(df: pd.DataFrame) -> pd.DataFrame:
    bins5 = np.array([15, 20, 25, 30, 35, 40, 45])
    df = df.copy()
    df["bmi_bin_test"] = pd.cut(df["bmi"], bins=bins5, right=False)
    out = []
    for b in sorted(df["bmi_bin_test"].dropna().unique(), key=lambda x: x.left):
        sub = df[df["bmi_bin_test"] == b]
        # tier x outcome table (3x2)
        tab = pd.crosstab(sub["centre_volume_cat"], sub["best_performer"])
        # ensure both outcome columns exist
        if 0 not in tab.columns:
            tab[0] = 0
        if 1 not in tab.columns:
            tab[1] = 0
        tab = tab.reindex(index=["Low", "Mid", "High"], fill_value=0)[[0, 1]]
        # skip bins with no data
        if tab.to_numpy().sum() == 0:
            continue
        chi2, p, _, _ = chi2_contingency(tab.to_numpy(), correction=False)
        out.append({
            "bmi_bin": str(b),
            "bmi_mid": float(b.left + (b.right - b.left) / 2),
            "n_low": int(tab.loc["Low"].sum()),
            "n_mid": int(tab.loc["Mid"].sum()),
            "n_high": int(tab.loc["High"].sum()),
            "bp_low": float(tab.loc["Low", 1] / tab.loc["Low"].sum()) if tab.loc["Low"].sum() else np.nan,
            "bp_mid": float(tab.loc["Mid", 1] / tab.loc["Mid"].sum()) if tab.loc["Mid"].sum() else np.nan,
            "bp_high": float(tab.loc["High", 1] / tab.loc["High"].sum()) if tab.loc["High"].sum() else np.nan,
            "chi2": float(chi2),
            "p_raw": float(p),
        })
    tests = pd.DataFrame(out)
    if not tests.empty:
        tests["p_adj"] = multipletests(tests["p_raw"].fillna(1.0), method="holm")[1]
    return tests


def plot_tier3(df: pd.DataFrame, out_dir: Path) -> None:
    theme = NordWhiteTheme()
    palette = list(theme.palette)
    tiers = ["Low", "Mid", "High"]
    kernel = np.array([0.25, 0.50, 0.25], dtype=float)

    agg = build_bin_agg(df)
    tests = build_bin_tests(df)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for i, tier in enumerate(tiers):
        grp = agg[agg["centre_volume_cat"] == tier].dropna(subset=["bmi_mid"]).sort_values("bmi_mid")
        if grp.empty:
            continue
        color = palette[i % len(palette)]
        ci_low_raw, ci_high_raw = [], []
        for _, row in grp.iterrows():
            if row["count"] > 0:
                l, h = proportion_confint(count=int(row["sum"]), nobs=int(row["count"]), method="wilson")
            else:
                l, h = np.nan, np.nan
            ci_low_raw.append(l)
            ci_high_raw.append(h)
        ci_low_raw = pd.Series(ci_low_raw)
        ci_high_raw = pd.Series(ci_high_raw)

        sum_s = _smooth_reflect(grp["sum"], kernel)
        n_s = _smooth_reflect(grp["count"], kernel)
        mean_s = np.divide(sum_s, n_s, out=np.full_like(sum_s, np.nan, dtype=float), where=n_s > 0)
        ci_l = _smooth_reflect(ci_low_raw, kernel)
        ci_h = _smooth_reflect(ci_high_raw, kernel)

        ax.plot(grp["bmi_mid"], np.clip(mean_s, 0, 1), color=color, linewidth=2.5, label=tier)
        ax.fill_between(grp["bmi_mid"], np.clip(ci_l, 0, 1), np.clip(ci_h, 0, 1), color=color, alpha=0.12)

    # Rug: each patient BMI by tier
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    rug_levels = {"Low": -0.12, "Mid": -0.10, "High": -0.08}
    for i, tier in enumerate(tiers):
        color = palette[i % len(palette)]
        td = df[df["centre_volume_cat"] == tier]
        ax.plot(
            td["bmi"],
            [rug_levels[tier]] * len(td),
            "|",
            color=color,
            alpha=0.8,
            markersize=9,
            linestyle="None",
            transform=trans,
            clip_on=False,
        )

    # annotate binwise global p-values (3-tier chi-square) on 5-kg bins
    for _, r in tests.sort_values("bmi_mid").iterrows():
        x = float(r["bmi_mid"])
        pval = float(r["p_adj"]) if np.isfinite(r.get("p_adj", np.nan)) else float(r["p_raw"])
        if np.isfinite(pval):
            if pval < 0.05:
                ax.plot([x], [0.6], marker="*", color=theme.palette[2], clip_on=False)
            ax.text(x, 0.62, f"p={pval:.3f}", ha="center", va="bottom", fontsize=8, color=theme.palette[2])

    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Taux de Best Performer")
    ax.set_title("BP vs BMI : Low vs Mid vs High (tests par tranches)", color=theme.title)
    ax.set_xlim(BMI_MIN, BMI_MAX)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(bottom=0.24)
    ax.legend(frameon=False, title="Volume")
    apply_theme(ax, theme)
    fig.tight_layout()

    out_png = out_dir / "bmi_bp_volume_tier3.png"
    out_svg = out_dir / "bmi_bp_volume_tier3.svg"
    fig.savefig(out_png, dpi=300, facecolor=theme.background)
    fig.savefig(out_svg, dpi=300, facecolor=theme.background)
    plt.close(fig)

    agg.to_csv(out_dir / "tier3_bin2_rates.csv", index=False)
    tests.to_csv(out_dir / "tier3_bin5_tests.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Univariate tier3 BP vs BMI plot (Low/Mid/High).")
    parser.add_argument(
        "--volume-tier-mode",
        choices=["tertiles", "annual_threshold", "mipd_annual_threshold"],
        default="tertiles",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "outputs_univariate_tier3" / args.volume_tier_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "volume_mode.txt").write_text(args.volume_tier_mode)

    df = build_dataset()
    df = apply_volume_mode(df, mode=args.volume_tier_mode)
    df = df.dropna(subset=["bmi", "best_performer", "centre_volume_cat"]).copy()
    df["best_performer"] = df["best_performer"].astype(int)

    plot_tier3(df, out_dir)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

