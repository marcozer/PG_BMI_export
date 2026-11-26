#!/usr/bin/env python3
"""Build HBSN-ready figures with the final primary volume definition.

The source `export/` figures are useful historically, but some were produced
before the final HBSN convention was locked: Low <5, Mid 5-10, High >10 mean
annual MILP cases per centre. This script rebuilds the main figures directly
from `hbsn_revision.scripts.dataset` so the displayed rugs, curves, and group
definitions match the revised tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import transforms
from patsy import build_design_matrices
from scipy.special import expit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from scripts.dataset import BMI6_ORDER, VOLUME_ORDER, build_dataset  # noqa: E402

OUT = Path(__file__).resolve().parent
MAIN = OUT / "main"
SUPP = OUT / "supplementary"
MAIN.mkdir(parents=True, exist_ok=True)
SUPP.mkdir(parents=True, exist_ok=True)

NORD = {
    "axis": "#4C566A",
    "title": "#3B4252",
    "grid": "#D8DEE9",
    "blue": "#5E81AC",
    "green": "#A3BE8C",
    "orange": "#D08770",
    "red": "#BF616A",
    "dark": "#3B4252",
    "background": "#FFFFFF",
}


def style_ax(ax: plt.Axes) -> None:
    ax.figure.set_facecolor(NORD["background"])
    ax.set_facecolor(NORD["background"])
    ax.grid(color=NORD["grid"], alpha=0.45)
    ax.tick_params(colors=NORD["axis"], labelsize=10)
    ax.xaxis.label.set_color(NORD["axis"])
    ax.yaxis.label.set_color(NORD["axis"])
    ax.title.set_color(NORD["title"])
    for spine in ax.spines.values():
        spine.set_color(NORD["axis"])


def save(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(prefix.with_suffix(".png"), dpi=300, facecolor=NORD["background"], bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), dpi=300, facecolor=NORD["background"], bbox_inches="tight")
    plt.close(fig)


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def smooth_reflect(values: pd.Series | np.ndarray, kernel: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    pad = len(kernel) // 2
    vpad = np.pad(v, pad_width=pad, mode="reflect")
    return np.convolve(vpad, kernel, mode="valid")


def add_rug(ax: plt.Axes, df: pd.DataFrame, group_col: str, colors: dict[str, str], levels: dict[str, float]) -> None:
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for group, level in levels.items():
        sub = df[df[group_col].astype(str) == group]
        ax.plot(
            sub["bmi"],
            [level] * len(sub),
            "|",
            color=colors[group],
            alpha=0.75,
            markersize=8,
            linestyle="None",
            transform=trans,
            clip_on=False,
        )


def plot_component_stack(df: pd.DataFrame) -> None:
    components = [
        ("popf_bc", "CR-POPF B/C", NORD["blue"]),
        ("major_clavien", "Clavien-Dindo >=III", NORD["orange"]),
        ("conversion", "Conversion", "#A7BE95"),
        ("readmission", "Readmission", NORD["red"]),
        ("reoperation", "Reoperation", NORD["dark"]),
    ]
    grouped = []
    for bmi_cat in BMI6_ORDER:
        sub = df[df["bmi6"].astype(str) == bmi_cat]
        row = {"BMI category": bmi_cat, "n": len(sub)}
        for col, label, _color in components:
            row[label] = sub[col].astype(bool).mean() if len(sub) else np.nan
        grouped.append(row)
    rates = pd.DataFrame(grouped)
    rates.to_csv(SUPP / "component_stack_hbsn_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    x = np.arange(len(rates))
    bottom = np.zeros(len(rates))
    for _col, label, color in components:
        vals = rates[label].fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, label=label, width=0.72)
        bottom += vals
    for i, n in enumerate(rates["n"]):
        ax.text(i, min(bottom[i] + 0.025, 0.965), f"n={n}", ha="center", va="bottom", color=NORD["axis"], fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(rates["BMI category"])
    ax.set_xlabel("BMI category (kg/m2)")
    ax.set_ylabel("Complication rate")
    ax.set_title("Observed complication components by BMI category", fontsize=16)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper left")
    style_ax(ax)
    fig.tight_layout()
    save(fig, MAIN / "FIGURE_1_component_stack_BMI_components")


def plot_observed_bp_by_volume(df: pd.DataFrame) -> None:
    data = df.dropna(subset=["bmi", "best_performer", "centre_volume_tier"]).copy()
    data["bp"] = data["best_performer"].astype(int)
    data["tier2"] = np.where(data["centre_volume_tier"].astype(str).eq("High"), "High", "Low+Mid")
    order = ["Low+Mid", "High"]
    colors = {"Low+Mid": NORD["blue"], "High": NORD["green"]}

    fig, ax = plt.subplots(figsize=(7, 4.25), dpi=150)
    add_rug(ax, data, "tier2", colors, {"Low+Mid": -0.115, "High": -0.085})

    bins2 = np.arange(15, 47, 2)
    data["bmi_bin2"] = pd.cut(data["bmi"], bins=bins2, right=False)
    agg = (
        data.groupby(["tier2", "bmi_bin2"], observed=False)["bp"]
        .agg(["sum", "count"])
        .reset_index()
    )
    agg["bmi_mid"] = agg["bmi_bin2"].apply(lambda b: float(b.left + (b.right - b.left) / 2) if pd.notna(b) else np.nan)
    kernel = np.array([0.25, 0.50, 0.25])

    for tier in order:
        grp = agg[(agg["tier2"] == tier) & (agg["count"] > 0)].dropna(subset=["bmi_mid"]).sort_values("bmi_mid")
        if grp.empty:
            continue
        n_s = smooth_reflect(grp["count"], kernel)
        y_s = smooth_reflect(grp["sum"], kernel)
        mean = np.divide(y_s, n_s, out=np.full_like(y_s, np.nan, dtype=float), where=n_s > 0)
        lows, highs = [], []
        for _, row in grp.iterrows():
            lo, hi = proportion_confint(int(row["sum"]), int(row["count"]), method="wilson")
            lows.append(lo)
            highs.append(hi)
        lows = np.clip(smooth_reflect(pd.Series(lows), kernel), 0, 1)
        highs = np.clip(smooth_reflect(pd.Series(highs), kernel), 0, 1)
        ax.plot(grp["bmi_mid"], mean, color=colors[tier], linewidth=2.5, label=tier)
        ax.fill_between(grp["bmi_mid"], lows, highs, color=colors[tier], alpha=0.12)

    bins5 = np.array([15, 20, 25, 30, 35, 40, 45])
    data["bmi_bin5"] = pd.cut(data["bmi"], bins=bins5, right=False)
    tests = []
    for b in data["bmi_bin5"].dropna().cat.categories:
        sub = data[data["bmi_bin5"].eq(b)]
        h = sub[sub["tier2"].eq("High")]
        lm = sub[sub["tier2"].eq("Low+Mid")]
        if len(h) < 10 or len(lm) < 20:
            continue
        ph = h["bp"].mean()
        pl = lm["bp"].mean()
        se = np.sqrt(ph * (1 - ph) / len(h) + pl * (1 - pl) / len(lm))
        p = 2 * (1 - norm.cdf(abs((ph - pl) / se))) if se > 0 else np.nan
        tests.append(
            {
                "bmi_bin": str(b),
                "bmi_mid": float(b.left + (b.right - b.left) / 2),
                "n_high": len(h),
                "n_low_mid": len(lm),
                "bp_high": ph,
                "bp_low_mid": pl,
                "difference": ph - pl,
                "p_raw": p,
            }
        )
    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        tests_df["p_adj"] = multipletests(tests_df["p_raw"], method="holm")[1]
        tests_df.to_csv(SUPP / "observed_bp_volume_bin5_tests_hbsn.csv", index=False)
        for _, row in tests_df.iterrows():
            ax.text(row["bmi_mid"], 0.64, fmt_p(row["p_adj"]), ha="center", color=NORD["green"], fontsize=8)
            if row["p_adj"] < 0.05:
                ax.plot(row["bmi_mid"], 0.60, marker="*", color=NORD["green"], markersize=7)

    ax.set_xlabel("BMI (kg/m2)", labelpad=30)
    ax.set_ylabel("Best Performer rate")
    ax.set_title("Observed Best Performer rate by BMI and centre volume", fontsize=16)
    ax.set_xlim(15, 45)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, title="Centre volume", loc="upper right")
    style_ax(ax)
    fig.subplots_adjust(bottom=0.30)
    save(fig, MAIN / "FIGURE_2A_BP_observed_by_BMI_and_volume")


def marginal_prediction(result, design_info, pred_df: pd.DataFrame) -> tuple[float, float, float]:
    exog = np.asarray(build_design_matrices([design_info], pred_df, return_type="dataframe")[0], dtype=float)
    beta = np.asarray(result.params, dtype=float)
    cov = np.asarray(result.cov_params(), dtype=float)
    p = expit(exog @ beta)
    mean = float(np.mean(p))
    grad = np.mean((p * (1 - p))[:, None] * exog, axis=0)
    var = float(grad @ cov @ grad)
    se = np.sqrt(var) if var >= 0 else np.nan
    return mean, max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)


def plot_adjusted_bmi_bp(df: pd.DataFrame) -> None:
    covars = ["age", "asa_ge3", "sex_male", "malignant", "robotic", "splenectomy", "centre_volume_annual"]
    data = df.dropna(subset=["bmi", "best_performer", "CENTRE"] + covars).copy()
    for col in ["best_performer", "asa_ge3", "sex_male", "malignant", "robotic", "splenectomy"]:
        data[col] = data[col].astype(int)
    formula = "best_performer ~ bs(bmi, df=4, include_intercept=False) + age + asa_ge3 + sex_male + malignant + robotic + splenectomy + centre_volume_annual"
    result = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": data["CENTRE"]}
    )
    design_info = result.model.data.design_info
    grid = np.linspace(max(15, data["bmi"].min()), min(45, data["bmi"].max()), 220)
    rows = []
    for bmi_value in grid:
        pred_df = data.copy()
        pred_df["bmi"] = bmi_value
        pred, lo, hi = marginal_prediction(result, design_info, pred_df)
        rows.append({"bmi": bmi_value, "pred": pred, "ci_low": lo, "ci_high": hi})
    curve = pd.DataFrame(rows)
    curve.to_csv(SUPP / "adjusted_bmi_bp_curve_hbsn.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(curve["bmi"], curve["pred"], color=NORD["blue"], linewidth=2.2, label="Adjusted probability")
    ax.fill_between(curve["bmi"], curve["ci_low"], curve["ci_high"], color=NORD["blue"], alpha=0.18, label="95% CI")
    ax.plot(df["bmi"], [0.012] * len(df), "|", color=NORD["red"], alpha=0.35, markersize=7, linestyle="None")
    for x, color in [(18.5, NORD["grid"]), (20, NORD["grid"]), (25, NORD["green"]), (30, NORD["green"]), (35, NORD["green"]), (40, NORD["green"])]:
        ax.axvline(x, color=color, linestyle="--", alpha=0.65, linewidth=1.2)
    ax.set_xlabel("BMI (kg/m2)")
    ax.set_ylabel("Best Performer probability")
    ax.set_title("Adjusted association between BMI and Best Performer", fontsize=16)
    ax.set_xlim(15, 45)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    save(fig, MAIN / "FIGURE_2B_adjusted_BMI_BP_association")


def plot_adjusted_interaction(df: pd.DataFrame) -> None:
    covars = ["age", "asa_ge3", "sex_male", "malignant", "robotic", "splenectomy"]
    data = df.dropna(subset=["bmi", "centre_volume_annual", "best_performer", "CENTRE"] + covars).copy()
    for col in ["best_performer", "asa_ge3", "sex_male", "malignant", "robotic", "splenectomy"]:
        data[col] = data[col].astype(int)
    data["bmi_per5"] = data["bmi"] / 5
    data["volume_per5"] = data["centre_volume_annual"] / 5
    formula = "best_performer ~ bmi_per5 * volume_per5 + age + asa_ge3 + sex_male + malignant + robotic + splenectomy"
    model = smf.gee(
        formula=formula,
        groups="CENTRE",
        data=data,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit()
    design_info = result.model.data.design_info
    p_interaction = float(result.pvalues["bmi_per5:volume_per5"])

    centre_values = (
        data.drop_duplicates("CENTRE")
        .groupby("centre_volume_tier", observed=False)["centre_volume_annual"]
        .median()
        .reindex(VOLUME_ORDER)
    )
    grid = np.linspace(15, min(45, data["bmi"].max()), 220)
    rows = []
    for tier in VOLUME_ORDER:
        vol = float(centre_values.loc[tier])
        for bmi_value in grid:
            pred_df = data.copy()
            pred_df["bmi_per5"] = bmi_value / 5
            pred_df["volume_per5"] = vol / 5
            pred, lo, hi = marginal_prediction(result, design_info, pred_df)
            rows.append({"tier": tier, "bmi": bmi_value, "prob": pred, "ci_low": lo, "ci_high": hi, "volume_value": vol})
    curves = pd.DataFrame(rows)
    curves.to_csv(SUPP / "adjusted_bmi_volume_interaction_curves_hbsn.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.35), dpi=150)
    colors = {"Low": NORD["blue"], "Mid": NORD["orange"], "High": NORD["green"]}
    for tier in VOLUME_ORDER:
        grp = curves[curves["tier"].eq(tier)]
        ax.plot(grp["bmi"], grp["prob"], color=colors[tier], linewidth=2.3, label=f"{tier}")
        ax.fill_between(grp["bmi"], grp["ci_low"], grp["ci_high"], color=colors[tier], alpha=0.13)
    add_rug(ax, data, "centre_volume_tier", colors, {"Low": -0.13, "Mid": -0.105, "High": -0.08})
    ax.text(0.03, 0.92, f"BMI x volume interaction {fmt_p(p_interaction)}", transform=ax.transAxes, color=NORD["title"], fontsize=9, va="top")
    ax.set_xlabel("BMI (kg/m2)", labelpad=30)
    ax.set_ylabel("Best Performer probability")
    ax.set_title("Adjusted BMI-volume interaction for Best Performer", fontsize=16)
    ax.set_xlim(15, 45)
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False, title="Centre volume", loc="lower left")
    style_ax(ax)
    fig.subplots_adjust(bottom=0.31)
    save(fig, MAIN / "FIGURE_2C_adjusted_BMI_volume_interaction_BP")


def main() -> None:
    df = build_dataset()
    plot_component_stack(df)
    plot_observed_bp_by_volume(df)
    plot_adjusted_bmi_bp(df)
    plot_adjusted_interaction(df)


if __name__ == "__main__":
    main()
