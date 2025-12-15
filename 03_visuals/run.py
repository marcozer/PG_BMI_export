"""Paragraph 3 – Component breakdown with stats and volume link."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "analysis"))
from lib.dataset import build_dataset
import argparse
from lib.plotting import NordWhiteTheme, apply_theme

COMPONENTS = [
    ("popf_bc", "CR-POPF B/C"),
    ("major_clavien", "Clavien ≥III"),
    ("conversion", "Conversion"),
    ("readmission", "Réadmission"),
    ("reoperation", "Réintervention"),
]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PLOT_STACK = OUTPUT_DIR / "component_stack.png"
PLOT_STACK_SVG = OUTPUT_DIR / "component_stack.svg"
PLOT_VOL = OUTPUT_DIR / "popf_bmi_volume.png"
PLOT_VOL_SVG = OUTPUT_DIR / "popf_bmi_volume.svg"


def component_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("bmi_class", observed=False)["CODE"].count().rename("n")
    rates = df.groupby("bmi_class", observed=False)[[c for c, _ in COMPONENTS]].mean()
    table = pd.concat([counts, rates], axis=1).reset_index()
    table.columns = ["bmi_class", "n"] + [label for _, label in COMPONENTS]
    return table


def chi_square_bmi(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for comp, label in COMPONENTS:
        cross = pd.crosstab(df["bmi_class"], df[comp])
        res = sm.stats.Table(cross).test_nominal_association()
        results.append({"component": label, "chi2": float(res.statistic), "p_value": float(res.pvalue)})
    return pd.DataFrame(results)


def popf_bmi_volume(df: pd.DataFrame) -> pd.DataFrame:
    df_use = df.dropna(subset=["bmi", "centre_volume_cat"]).copy()
    df_use["popf_bc"] = df_use["popf_bc"].astype(int)
    df_use["bmi30"] = (df_use["bmi"] >= 30).astype(int)
    df_use["highvol"] = df_use["centre_volume_cat"].eq("High").astype(int)
    cross = pd.crosstab([df_use["bmi30"], df_use["highvol"]], df_use["popf_bc"])
    res = sm.stats.Table(cross).test_nominal_association()
    chi2 = float(res.statistic)
    p = float(res.pvalue)
    # logistic regression for POPF ~ bmi30 * highvol
    formula = "popf_bc ~ bmi30 + highvol + bmi30:highvol"
    logit = sm.Logit.from_formula(formula, data=df_use).fit(disp=False)
    p_int = logit.pvalues.get("bmi30:highvol", np.nan)
    return pd.DataFrame({
        "table": [cross.to_string()],
        "chi2": [chi2],
        "p_value": [p],
        "logit_interaction_p": [p_int],
    })


def stack_plot(table: pd.DataFrame) -> None:
    theme = NordWhiteTheme()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    bottom = np.zeros(len(table))
    palette = list(theme.palette) + [theme.axis]
    for idx, label in enumerate(table.columns[2:]):
        color = palette[idx % len(palette)]
        ax.bar(table["bmi_class"], table[label], bottom=bottom, color=color, label=label)
        bottom = bottom + table[label]
    for i, n in enumerate(table["n"]):
        ax.text(i, 1.02, f"n={int(n)}", ha="center", color=theme.axis, fontsize=9)
    ax.set_xlabel("BMI (classes OMS)")
    ax.set_ylabel("Taux de complication")
    ax.set_ylim(0, 1.1)
    ax.set_title("Composantes empêchant l'Ideal Outcome", color=theme.title)
    ax.legend(frameon=False, loc="upper left")
    apply_theme(ax, theme)
    fig.tight_layout()
    fig.savefig(PLOT_STACK, dpi=150, facecolor=theme.background)
    fig.savefig(PLOT_STACK_SVG, dpi=150, facecolor=theme.background)
    plt.close(fig)


def popf_volume_plot(df: pd.DataFrame) -> None:
    df_use = df.dropna(subset=["bmi", "centre_volume_cat"]).copy()
    df_use["popf_bc"] = df_use["popf_bc"].astype(int)
    df_use["bmi30"] = (df_use["bmi"] >= 30).map({False: "BMI <30", True: "BMI ≥30"})
    df_use["vol"] = df_use["centre_volume_cat"].map({"Low": "Low", "Mid": "Mid", "High": "High"})
    tab = df_use.pivot_table(values="popf_bc", index="vol", columns="bmi30", aggfunc="mean", observed=False)
    theme = NordWhiteTheme()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    tab.plot(kind="bar", ax=ax, color=[theme.palette[0], theme.palette[2]])
    ax.set_ylabel("Taux CR-POPF B/C")
    ax.set_title("CR-POPF selon BMI et volume", color=theme.title)
    apply_theme(ax, theme)
    fig.tight_layout()
    fig.savefig(PLOT_VOL, dpi=150, facecolor=theme.background)
    fig.savefig(PLOT_VOL_SVG, dpi=150, facecolor=theme.background)
    plt.close(fig)


def rank_components(table: pd.DataFrame) -> pd.DataFrame:
    table2 = table.copy()
    table2["bmi30"] = table2["bmi_class"].apply(lambda x: "<30" if str(x) in ["<18.5", "18.5-24.9", "25-29.9"] else "≥30")
    agg = table2.groupby("bmi30", observed=False)[[label for _, label in COMPONENTS]].mean().reset_index()
    wide = agg.set_index("bmi30").T
    wide["difference"] = wide["≥30"] - wide["<30"]
    wide = wide.reset_index().rename(columns={"index": "component"})
    return wide


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
        else:  # mipd_annual_threshold = total MIPD / number of years
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
            # Expertise defined by mean annual minimally invasive pancreatectomies
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


def main(volume_tier_mode: str = "tertiles") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df = apply_volume_mode(df, mode=volume_tier_mode)
    # record mode used
    (OUTPUT_DIR / "volume_mode.txt").write_text(volume_tier_mode)

    table = component_table(df)
    table.to_csv(OUTPUT_DIR / "component_rates.csv", index=False)
    stack_plot(table)

    chi = chi_square_bmi(df)
    chi.to_csv(OUTPUT_DIR / "component_chi_square.csv", index=False)

    popf = popf_bmi_volume(df)
    popf.to_csv(OUTPUT_DIR / "popf_bmi_volume_tests.csv", index=False)
    popf_volume_plot(df)

    ranked = rank_components(table)
    ranked.to_csv(OUTPUT_DIR / "component_rank_bmi30.csv", index=False)

    print("Saved component stats, plots, and tables")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component breakdown export")
    parser.add_argument("--volume-tier-mode", choices=["tertiles", "annual_threshold", "mipd_annual_threshold"], default="tertiles", help="How to define volume tiers")
    args = parser.parse_args()
    main(volume_tier_mode=args.volume_tier_mode)
