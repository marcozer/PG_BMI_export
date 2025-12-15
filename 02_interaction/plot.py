"""Enhanced interaction plots: dose-response by volume tier with CI bands."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "analysis"))
from lib.plotting import NordWhiteTheme, apply_theme
from lib.dataset import build_dataset
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PLOT_TIER = OUTPUT_DIR / "bmi_interaction_tiers.png"
PLOT_TIER_SVG = OUTPUT_DIR / "bmi_interaction_tiers.svg"
PLOT_TIER_ANN = OUTPUT_DIR / "bmi_interaction_tiers_annotated.png"
PLOT_TIER_ANN_SVG = OUTPUT_DIR / "bmi_interaction_tiers_annotated.svg"
PLOT_DIFF = OUTPUT_DIR / "bmi_interaction_diff.png"
PLOT_DIFF_SVG = OUTPUT_DIR / "bmi_interaction_diff.svg"
PLOT_MAIN = OUTPUT_DIR / "bmi_interaction_main.png"
PLOT_MAIN_SVG = OUTPUT_DIR / "bmi_interaction_main.svg"
PLOT_COMBINED = OUTPUT_DIR / "bmi_interaction_combined.png"
PLOT_COMBINED_SVG = OUTPUT_DIR / "bmi_interaction_combined.svg"
PLOT_FACETS = OUTPUT_DIR / "bmi_interaction_facets.png"
PLOT_FACETS_SVG = OUTPUT_DIR / "bmi_interaction_facets.svg"
PLOT_POINTS = OUTPUT_DIR / "bmi_interaction_points.png"
PLOT_POINTS_SVG = OUTPUT_DIR / "bmi_interaction_points.svg"
PLOT_ORBARS = OUTPUT_DIR / "or_per5_bars.png"
PLOT_ORBARS_SVG = OUTPUT_DIR / "or_per5_bars.svg"
PLOT_CALIB = OUTPUT_DIR / "calibration_tier.png"
PLOT_CALIB_SVG = OUTPUT_DIR / "calibration_tier.svg"

def _read_volume_mode(default: str = "tertiles") -> str:
    """Read the volume tiering mode recorded during run.py execution."""
    try:
        txt = (OUTPUT_DIR / "volume_mode.txt").read_text().strip()
        if txt in {"tertiles", "annual_threshold"}:
            return txt
    except Exception:
        pass
    return default

def apply_volume_mode(df: pd.DataFrame, mode: str = "tertiles") -> pd.DataFrame:
    """Compute centre volume metric and tiers according to the requested mode.

    - tertiles (default): total cases per centre across the whole dataset, tertile cut.
    - annual_threshold: mean annual cases per centre; Low <5, Mid 5–10, High >10.
    - mipd_annual_threshold: mean annual minimally invasive pancreatectomies; Low <10, Mid 10–20, High >20.
    Returns a modified copy of df with updated 'centre_volume' and 'centre_volume_cat'.
    """
    df = df.copy()
    if mode in {"annual_threshold", "mipd_annual_threshold"}:
        if "year" not in df.columns and "ANNEE" in df.columns:
            df["year"] = pd.to_numeric(df["ANNEE"], errors="coerce")
        if mode == "annual_threshold":
            counts_cy = (
                df.groupby(["CENTRE", "year"], observed=False)["CODE"].count().reset_index(name="vol_cy")
            )
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
        labels = ["Low", "Mid", "High"]
        df["centre_volume_cat"] = pd.cut(df["centre_volume"], bins=bins, labels=labels, right=True, include_lowest=True)
    else:
        df["centre_volume"] = df.groupby("CENTRE")["CODE"].transform("count")
        try:
            df["centre_volume_cat"] = pd.qcut(
                df["centre_volume"], q=[0, 0.33, 0.66, 1.0], labels=["Low", "Mid", "High"], duplicates="drop"
            )
        except Exception:
            df["centre_volume_cat"] = pd.cut(df["centre_volume"], bins=3, labels=["Low", "Mid", "High"]) 
    return df


def plot_tier_curves(curve_path: Path, pairwise_path: Path, interaction_p_path: Path) -> None:
    df = pd.read_csv(curve_path)
    # Safety: if linear curves file is stale (probabilities > 0.5),
    # try the spline file as a fallback to avoid empty-looking plots.
    if df["prob"].max() > 0.5:
        alt = curve_path.parent / "bmi_volume_curves_spline.csv"
        if alt.exists():
            try:
                df_alt = pd.read_csv(alt)
                if not df_alt.empty:
                    df = df_alt
            except Exception:
                pass
    pair = pd.read_csv(pairwise_path) if pairwise_path.exists() else pd.DataFrame()
    pint = None
    if interaction_p_path.exists():
        pint_df = pd.read_csv(interaction_p_path)
        pint = pint_df.iloc[0].get("gee_interaction_p", None)
    theme = NordWhiteTheme()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    palette = list(theme.palette)
    for idx, tier in enumerate(["Low", "Mid", "High"]):
        grp = df[df["tier"] == tier].sort_values("bmi")
        if grp.empty:
            continue
        color = palette[idx % len(palette)]
        ax.plot(grp["bmi"], grp["prob"], color=color, label=tier)
        ax.fill_between(grp["bmi"], grp["ci_low"], grp["ci_high"], color=color, alpha=0.15)
    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Probabilité de Best Performer")
    ax.set_title("Interaction BMI × volume : tiers de centre", color=theme.title)
    ax.set_xlim(18, 45)
    ax.set_ylim(0, 0.5)
    ax.legend(frameon=False, title="Volume", loc="lower left")
    note = []
    if pint is not None and pd.notna(pint):
        note.append(f"Interaction p={pint:.3f}")
    row40 = pair[pair.get("bmi", pd.Series(dtype=float)) == 40]
    if not row40.empty:
        diff = float(row40.iloc[0]["diff_high_minus_low"])
        p40 = float(row40.iloc[0]["p_value"])
        ci_l = float(row40.iloc[0].get("ci_low", float("nan")))
        ci_h = float(row40.iloc[0].get("ci_high", float("nan")))
        if pd.notna(ci_l) and pd.notna(ci_h):
            note.append(f"BMI 40: ΔHigh-Low={diff:.02f} ({ci_l:.02f},{ci_h:.02f}); p={p40:.3f}")
        else:
            note.append(f"BMI 40: ΔHigh-Low={diff:.02f} (p={p40:.3f})")
    if note:
        ax.text(0.02, 0.05, "\n".join(note), transform=ax.transAxes, color=theme.title, fontsize=10, ha="left", va="bottom")
    apply_theme(ax, theme)
    fig.tight_layout()
    fig.savefig(PLOT_TIER, dpi=150, facecolor=theme.background)
    fig.savefig(PLOT_TIER_SVG, dpi=150, facecolor=theme.background)
    fig.savefig(PLOT_TIER_ANN, dpi=150, facecolor=theme.background)
    fig.savefig(PLOT_TIER_ANN_SVG, dpi=150, facecolor=theme.background)
    plt.close(fig)

def plot_diff_curve(diff_path: Path, interaction_p_path: Path) -> None:
    df = pd.read_csv(diff_path)
    if df["diff"].abs().max() == 0 and (diff_path.parent / "bmi_interaction_diff_spline.csv").exists():
        try:
            df2 = pd.read_csv(diff_path.parent / "bmi_interaction_diff_spline.csv")
            if not df2.empty:
                df = df2
        except Exception:
            pass
    pint = None
    if interaction_p_path.exists():
        pint_df = pd.read_csv(interaction_p_path)
        pint = pint_df.iloc[0].get("gee_interaction_p", None)
    theme = NordWhiteTheme()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for comp, color in zip(["High-Low", "High-Mid"], [theme.palette[2], theme.palette[1]]):
        grp = df[df["comparison"] == comp].sort_values("bmi")
        if grp.empty:
            continue
        ax.plot(grp["bmi"], grp["diff"], color=color, label=comp)
        ax.fill_between(grp["bmi"], grp["ci_low"], grp["ci_high"], color=color, alpha=0.15)
    ax.axhline(0, color=theme.grid, linewidth=1)
    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Δ Probabilité (High − autre)")
    ax.set_title("Interaction BMI×volume – différence de probabilité", color=theme.title)
    ax.set_xlim(18, 45)
    ax.set_ylim(-0.2, 0.2)
    ax.legend(frameon=False, title="Comparaison")
    if pint is not None and pd.notna(pint):
        ax.text(0.02, 0.05, f"Interaction p={pint:.3f}", transform=ax.transAxes, color=theme.title)
    apply_theme(ax, theme)
    fig.tight_layout()
    fig.savefig(PLOT_DIFF, dpi=150, facecolor=theme.background)
    fig.savefig(PLOT_DIFF_SVG, dpi=150, facecolor=theme.background)
    plt.close(fig)

def plot_calibration(calib_path: Path) -> None:
    if not calib_path.exists():
        return
    df = pd.read_csv(calib_path)
    theme = NordWhiteTheme()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    palette = list(theme.palette)
    for idx, tier in enumerate(["Low", "Mid", "High"]):
        grp = df[df["tier"] == tier].sort_values("decile")
        if grp.empty:
            continue
        color = palette[idx % len(palette)]
        ax.plot(grp["mean_pred"], grp["mean_obs"], marker="o", color=color, label=tier)
    ax.plot([0, 1], [0, 1], color=theme.grid, linestyle="--", linewidth=1)
    ax.set_xlabel("Prédit (moyenne par décile)")
    ax.set_ylabel("Observé (moyenne par décile)")
    ax.set_title("Calibration par tiers de volume", color=theme.title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, title="Volume", loc="lower right")
    apply_theme(ax, theme)
    fig.tight_layout()
    fig.savefig(PLOT_CALIB, dpi=150, facecolor=theme.background)
    fig.savefig(PLOT_CALIB_SVG, dpi=150, facecolor=theme.background)
    plt.close(fig)

def main() -> None:
    # Primary analysis = LINEAR (BMI × volume)
    curves_csv = OUTPUT_DIR / "bmi_volume_curves.csv"
    diff_csv = OUTPUT_DIR / "bmi_interaction_diff.csv"
    pair_csv = OUTPUT_DIR / "pairwise_bmi_comparisons.csv"
    pint_csv = OUTPUT_DIR / "interaction_p.csv"
    # Fallback to spline if linear files absent
    if not curves_csv.exists():
        curves_csv = OUTPUT_DIR / "bmi_volume_curves_spline.csv"
    if not diff_csv.exists():
        diff_csv = OUTPUT_DIR / "bmi_interaction_diff_spline.csv"
    if not pair_csv.exists():
        pair_csv = OUTPUT_DIR / "pairwise_bmi_comparisons_spline.csv"
    if not pint_csv.exists():
        pint_csv = OUTPUT_DIR / "interaction_p_spline.csv"

    plot_tier_curves(curves_csv, pair_csv, pint_csv)
    plot_diff_curve(diff_csv, pint_csv)
    # calibration figure prefers spline calibration file
    calib_csv = OUTPUT_DIR / "calibration_tier_spline.csv"
    if not calib_csv.exists():
        calib_csv = OUTPUT_DIR / "calibration_tier.csv"
    plot_calibration(calib_csv)
    # Single, publication-ready main figure: 3 tiers + model CIs + rug + p-interaction
    try:
        curves = pd.read_csv(curves_csv)
        pint_df = pd.read_csv(pint_csv) if pint_csv.exists() else None
        pair_df = pd.read_csv(pair_csv) if pair_csv.exists() else pd.DataFrame()
        or_df = pd.read_csv(OUTPUT_DIR / "or_per5_by_volume_tier.csv") if (OUTPUT_DIR / "or_per5_by_volume_tier.csv").exists() else None
        pint = None if pint_df is None else pint_df.iloc[0].get("gee_interaction_p", None)
        theme = NordWhiteTheme()
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        palette = list(theme.palette)
        for idx, tier in enumerate(["Low", "Mid", "High"]):
            grp = curves[curves["tier"] == tier].sort_values("bmi")
            if grp.empty:
                continue
            color = palette[idx % len(palette)]
            ax.plot(grp["bmi"], grp["prob"], color=color, linewidth=2.3, label=tier)
            ax.fill_between(grp["bmi"], grp["ci_low"], grp["ci_high"], color=color, alpha=0.15)
        # Rug: show BMI distribution by tier under axis, computed with SAME volume mode as curves
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        rug_levels = {"Low": -0.12, "Mid": -0.10, "High": -0.08}
        vm = _read_volume_mode(default="tertiles")
        df = build_dataset()
        df = apply_volume_mode(df, mode=vm)
        # Optionally restrict to complete cases used by the model to avoid misleading rugs
        needed = ["bmi", "centre_volume_cat"]
        df = df.dropna(subset=needed)
        for idx, tier in enumerate(["Low", "Mid", "High"]):
            t = df[df["centre_volume_cat"] == tier]
            if t.empty:
                continue
            ax.plot(t["bmi"], [rug_levels[tier]] * len(t), "|", color=palette[idx % len(palette)], alpha=0.85, markersize=9, linestyle="None", transform=trans, clip_on=False)
        ax.set_xlabel("BMI (kg/m²)")
        ax.set_ylabel("Probabilité de Best Performer")
        ax.set_title("BMI × volume: centres experts atténuent la pénalité", color=theme.title)
        ax.set_xlim(18, 45)
        ax.set_ylim(0, 0.5)
        ax.legend(frameon=False, title="Volume", loc="lower left")
        note = ["Courbes = prédictions GLM (IC 95%), SE cluster centre"]
        if pint is not None and pd.notna(pint):
            note.insert(0, f"Interaction GEE p={pint:.3f}")
        # Add OR per +5 BMI by tier
        if or_df is not None and not or_df.empty:
            try:
                or_lines = []
                for tier in ["Low", "Mid", "High"]:
                    row = or_df[or_df["tier"] == tier].iloc[0]
                    or_lines.append(f"{tier} OR(+5) {row['OR']:.2f} ({row['ci_low']:.2f};{row['ci_high']:.2f})")
                note.append("; ".join(or_lines))
            except Exception:
                pass
        # Add delta at BMI 40
        if not pair_df.empty and (pair_df["bmi"] == 40).any():
            r40 = pair_df[pair_df["bmi"] == 40].iloc[0]
            note.append(f"BMI40 ΔH-L={r40['diff_high_minus_low']:.02f} ({r40['ci_low']:.02f};{r40['ci_high']:.02f})")
        ax.text(0.02, 0.05, "\n".join(note), transform=ax.transAxes, color=theme.title, fontsize=9, ha="left", va="bottom")
        apply_theme(ax, theme)
        fig.tight_layout()
        fig.savefig(PLOT_MAIN, dpi=150, facecolor=theme.background)
        fig.savefig(PLOT_MAIN_SVG, dpi=150, facecolor=theme.background)
        plt.close(fig)
    except Exception as e:
        print("Warning: failed to produce main figure:", e)
    # Combined figure (Panel A: curves; Panel B: difference)
    try:
        df_curves = pd.read_csv(curves_csv)
        df_diff = pd.read_csv(diff_csv)
        theme = NordWhiteTheme()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150, sharex=False)
        # Panel A
        palette = list(theme.palette)
        for idx, tier in enumerate(["Low", "Mid", "High"]):
            grp = df_curves[df_curves["tier"] == tier].sort_values("bmi")
            if grp.empty:
                continue
            color = palette[idx % len(palette)]
            ax1.plot(grp["bmi"], grp["prob"], color=color, label=tier)
            ax1.fill_between(grp["bmi"], grp["ci_low"], grp["ci_high"], color=color, alpha=0.15)
        ax1.set_xlabel("BMI (kg/m²)"); ax1.set_ylabel("Probabilité BP"); ax1.set_xlim(18, 45); ax1.set_ylim(0, 0.5)
        ax1.set_title("A. Courbes ajustées", color=theme.title)
        ax1.legend(frameon=False, title="Volume", loc="lower left")
        apply_theme(ax1, theme)
        # Panel B
        for comp, color in zip(["High-Low", "High-Mid"], [theme.palette[2], theme.palette[1]]):
            grp = df_diff[df_diff["comparison"] == comp].sort_values("bmi")
            if grp.empty:
                continue
            ax2.plot(grp["bmi"], grp["diff"], color=color, label=comp)
            ax2.fill_between(grp["bmi"], grp["ci_low"], grp["ci_high"], color=color, alpha=0.15)
        ax2.axhline(0, color=theme.grid, linewidth=1)
        ax2.set_xlabel("BMI (kg/m²)"); ax2.set_ylabel("Δ Prob (High−autre)"); ax2.set_xlim(18, 45); ax2.set_ylim(-0.2, 0.2)
        ax2.set_title("B. Différence ajustée", color=theme.title)
        ax2.legend(frameon=False, loc="lower left")
        apply_theme(ax2, theme)
        fig.tight_layout()
        fig.savefig(PLOT_COMBINED, dpi=150, facecolor=theme.background)
        fig.savefig(PLOT_COMBINED_SVG, dpi=150, facecolor=theme.background)
        plt.close(fig)
    except Exception as e:
        print("Warning: failed to produce combined figure:", e)
    # Additional readable formats: facets and dot‑whiskers + OR bars
    try:
        curves = pd.read_csv(curves_csv)
        theme = NordWhiteTheme()
        # Facets: un panneau par tier
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=150, sharey=True)
        for ax, tier, color in zip(axes, ["Low", "Mid", "High"], list(theme.palette)):
            grp = curves[curves["tier"] == tier].sort_values("bmi")
            if grp.empty:
                continue
            ax.plot(grp["bmi"], grp["prob"], color=color)
            ax.fill_between(grp["bmi"], grp["ci_low"], grp["ci_high"], color=color, alpha=0.15)
            ax.set_title(tier, color=theme.title)
            ax.set_xlim(18, 45); ax.set_ylim(0, 0.5)
            ax.set_xlabel("BMI (kg/m²)")
            if tier == "Low":
                ax.set_ylabel("Probabilité BP")
            apply_theme(ax, theme)
        fig.tight_layout()
        fig.savefig(PLOT_FACETS, dpi=150, facecolor=theme.background)
        fig.savefig(PLOT_FACETS_SVG, dpi=150, facecolor=theme.background)
        plt.close(fig)
        # Points & whiskers aux BMI 25/30/35/40
        targets = [25, 30, 35, 40]
        rows = []
        for tier in ["Low", "Mid", "High"]:
            grp = curves[curves["tier"] == tier]
            for t in targets:
                row = grp.iloc[(grp["bmi"] - t).abs().argmin()]
                rows.append({"tier": tier, "bmi": int(round(t)), "prob": row["prob"], "ci_low": row["ci_low"], "ci_high": row["ci_high"]})
        pts = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        palette = list(theme.palette)
        xs = sorted(pts["bmi"].unique())
        offsets = {"Low": -0.25, "Mid": 0.0, "High": 0.25}
        for idx, tier in enumerate(["Low", "Mid", "High"]):
            sub = pts[pts["tier"] == tier]
            ax.errorbar([xi + offsets[tier] for xi in sub["bmi"]], sub["prob"],
                        yerr=[sub["prob"] - sub["ci_low"], sub["ci_high"] - sub["prob"]],
                        fmt='o', color=palette[idx], capsize=3, label=tier)
        ax.set_xticks(xs); ax.set_xlabel("BMI (kg/m²)"); ax.set_ylabel("Probabilité BP")
        ax.set_ylim(0, 0.5); ax.legend(frameon=False, title="Volume")
        apply_theme(ax, theme)
        fig.tight_layout()
        fig.savefig(PLOT_POINTS, dpi=150, facecolor=theme.background)
        fig.savefig(PLOT_POINTS_SVG, dpi=150, facecolor=theme.background)
        plt.close(fig)
        # Barres OR(+5 BMI) par tier
        or_path = OUTPUT_DIR / "or_per5_by_volume_tier.csv"
        if or_path.exists():
            odf = pd.read_csv(or_path)
            fig, ax = plt.subplots(figsize=(5.5, 4), dpi=150)
            tiers = ["Low", "Mid", "High"]
            vals = [float(odf[odf["tier"] == t]["OR"].iloc[0]) for t in tiers]
            ci_l = [float(odf[odf["tier"] == t]["ci_low"].iloc[0]) for t in tiers]
            ci_h = [float(odf[odf["tier"] == t]["ci_high"].iloc[0]) for t in tiers]
            for i, (v, l, h, color) in enumerate(zip(vals, ci_l, ci_h, palette)):
                ax.bar(i, v, color=color, alpha=0.85, width=0.6)
                ax.plot([i, i], [l, h], color=color)
                ax.plot([i-0.08, i+0.08], [l, l], color=color)
                ax.plot([i-0.08, i+0.08], [h, h], color=color)
            ax.set_xticks(range(len(tiers))); ax.set_xticklabels(tiers)
            ax.set_ylabel("OR (+5 BMI)"); ax.set_ylim(0.6, 1.0)
            ax.set_title("Effet du BMI (+5) par tier", color=theme.title)
            apply_theme(ax, theme)
            fig.tight_layout()
            fig.savefig(PLOT_ORBARS, dpi=150, facecolor=theme.background)
            fig.savefig(PLOT_ORBARS_SVG, dpi=150, facecolor=theme.background)
            plt.close(fig)
    except Exception as e:
        print("Warning: failed to produce alternative plots:", e)
    print(f"Saved {PLOT_TIER}, {PLOT_TIER_SVG}, {PLOT_TIER_ANN}, {PLOT_TIER_ANN_SVG}, {PLOT_DIFF}, {PLOT_DIFF_SVG}, {PLOT_CALIB}, {PLOT_CALIB_SVG}, {PLOT_MAIN}, {PLOT_MAIN_SVG}, {PLOT_COMBINED}, {PLOT_COMBINED_SVG}, {PLOT_FACETS}, {PLOT_FACETS_SVG}, {PLOT_POINTS}, {PLOT_POINTS_SVG}, {PLOT_ORBARS}, {PLOT_ORBARS_SVG}")


if __name__ == "__main__":
    main()
