"""Univariate BP vs BMI comparing (High+Mid) vs Low centre volume.

This duplicates the pooled univariate approach (binning + descriptive smoothing + binwise tests)
but pools High+Mid together and compares them against Low.

Outputs are written to export/01_bmi_risk/outputs_univariate_highmid_vs_low/<mode>/ so existing
results in export/01_bmi_risk/outputs/<mode>/ are not overwritten.
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

from scipy.stats import norm
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
        else:  # mipd_annual_threshold
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


def plot_highmid_vs_low(df: pd.DataFrame, out_dir: Path) -> None:
    theme = NordWhiteTheme()
    palette = [theme.palette[0], theme.palette[2]]
    order = ["Low", "High+Mid"]
    kernel = np.array([0.25, 0.50, 0.25], dtype=float)

    df = df.copy()
    df["tier2"] = df["centre_volume_cat"].map({"Low": "Low", "Mid": "High+Mid", "High": "High+Mid"})

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    # Rug
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    rug_levels = {"Low": -0.12, "High+Mid": -0.08}
    for idx, tier in enumerate(order):
        color = palette[idx]
        td = df[df["tier2"] == tier]
        ax.plot(td["bmi"], [rug_levels[tier]] * len(td), "|", color=color, alpha=0.8, markersize=9, linestyle="None", transform=trans, clip_on=False)

    # 2‑kg bins + Wilson CI + descriptive smoothing
    bins2 = np.arange(BMI_MIN, BMI_MAX, 2)
    df["bmi_bin"] = pd.cut(df["bmi"], bins=bins2, right=False)
    agg = (
        df.groupby(["tier2", "bmi_bin"], observed=False)["best_performer"]
        .agg(["mean", "count", "sum"]).reset_index()
    )
    agg["bmi_mid"] = agg["bmi_bin"].apply(lambda x: float(x.left + 1) if pd.notna(x) else np.nan)

    # Statistical comparison every 5 kg/m² (High+Mid vs Low) with Holm correction
    bins5 = np.array([15, 20, 25, 30, 35, 40, 45])
    df["bmi_bin_test"] = pd.cut(df["bmi"], bins=bins5, right=False)
    agg5 = (
        df.groupby(["tier2", "bmi_bin_test"], observed=False)["best_performer"]
        .agg(["mean", "count", "sum"]).reset_index()
    )
    out = []
    for b in agg5["bmi_bin_test"].dropna().unique():
        hi = agg5[(agg5["tier2"] == "High+Mid") & (agg5["bmi_bin_test"] == b)]
        lo = agg5[(agg5["tier2"] == "Low") & (agg5["bmi_bin_test"] == b)]
        if hi.empty or lo.empty:
            continue
        p_h, n_h = float(hi["mean"].iloc[0]), int(hi["count"].iloc[0])
        p_l, n_l = float(lo["mean"].iloc[0]), int(lo["count"].iloc[0])
        # keep the same z-test as the existing pooled script for consistency
        se = np.sqrt(p_h * (1 - p_h) / n_h + p_l * (1 - p_l) / n_l) if (n_h > 0 and n_l > 0) else np.nan
        diff = p_h - p_l
        z = diff / se if se and se > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        out.append({
            "bmi_bin": str(b),
            "bmi_mid": float(b.left + (b.right - b.left) / 2),
            "n_highmid": n_h,
            "n_low": n_l,
            "p_highmid": p_h,
            "p_low": p_l,
            "diff": diff,
            "p_raw": p,
        })
    tests_df = pd.DataFrame(out)
    if not tests_df.empty:
        tests_df["p_adj"] = multipletests(tests_df["p_raw"].fillna(1.0), method="holm")[1]
    tests_df.to_csv(out_dir / "highmid_vs_low_bin5_tests.csv", index=False)

    for idx, tier in enumerate(order):
        color = palette[idx]
        grp = agg[agg["tier2"] == tier].dropna(subset=["bmi_mid"]).sort_values("bmi_mid")
        if grp.empty:
            continue

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

    # annotate per-bin p-values + stars (Holm-adjusted)
    for _, r in tests_df.sort_values("bmi_mid").iterrows():
        x = float(r["bmi_mid"])
        pval = float(r["p_adj"]) if np.isfinite(r.get("p_adj", np.nan)) else float(r["p_raw"])
        if not np.isfinite(pval):
            continue
        if pval < 0.05:
            ax.plot([x], [0.6], marker="*", color=palette[1], clip_on=False)
        ax.text(x, 0.62, f"p={pval:.3f}", ha="center", va="bottom", fontsize=8, color=palette[1])

    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Taux de Best Performer")
    ax.set_title("BP vs BMI : High+Mid vs Low (tests par tranches)", color=theme.title)
    ax.set_xlim(BMI_MIN, BMI_MAX)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(bottom=0.24)
    ax.legend(frameon=False, title="Volume")
    apply_theme(ax, theme)
    fig.tight_layout()

    out_png = out_dir / "bmi_bp_volume_highmid_vs_low.png"
    out_svg = out_dir / "bmi_bp_volume_highmid_vs_low.svg"
    fig.savefig(out_png, dpi=300, facecolor=theme.background)
    fig.savefig(out_svg, dpi=300, facecolor=theme.background)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Univariate BP vs BMI plot: High+Mid vs Low.")
    parser.add_argument(
        "--volume-tier-mode",
        choices=["tertiles", "annual_threshold", "mipd_annual_threshold"],
        default="tertiles",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "outputs_univariate_highmid_vs_low" / args.volume_tier_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "volume_mode.txt").write_text(args.volume_tier_mode)

    df = build_dataset()
    df = apply_volume_mode(df, mode=args.volume_tier_mode)
    df = df.dropna(subset=["bmi", "best_performer", "centre_volume_cat"]).copy()
    df["best_performer"] = df["best_performer"].astype(int)

    plot_highmid_vs_low(df, out_dir)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
