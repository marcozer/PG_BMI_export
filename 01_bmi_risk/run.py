"""Paragraph 1 – BMI intrinsic effect with model comparisons and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import bs, build_design_matrices  # noqa: F401
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "analysis"))
from lib.dataset import build_dataset  # noqa: E402
import argparse

OUTCOMES = ["ideal_outcome", "best_performer", "major_clavien", "popf_bc", "conversion"]
COVARIATES = ["age", "asa_ge3", "sex_male", "malignant", "robotic", "splenectomy", "centre_volume"]
BMI_POINTS = [22, 27, 32, 37, 42]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def fit_models(df: pd.DataFrame, outcome: str) -> Dict[str, sm.GLM]:
    df_model = df.copy()
    df_model[outcome] = df_model[outcome].astype(int)
    family = sm.families.Binomial()
    covars = " + ".join(COVARIATES)
    models = {
        "spline": sm.GLM.from_formula(
            f"{outcome} ~ bs(bmi, df=4, include_intercept=False) + {covars}",
            data=df_model,
            family=family,
        ).fit(),
        "linear": sm.GLM.from_formula(
            f"{outcome} ~ bmi + {covars}",
            data=df_model,
            family=family,
        ).fit(),
        "categorical": sm.GLM.from_formula(
            f"{outcome} ~ C(bmi_class) + {covars}",
            data=df_model,
            family=family,
        ).fit(),
    }
    return models


def compare_models(models: Dict[str, sm.GLM], outcome: str) -> pd.DataFrame:
    spline, linear, categorical = models["spline"], models["linear"], models["categorical"]
    rows = []
    for name, mod in models.items():
        rows.append({
            "outcome": outcome,
            "model": name,
            "aic": mod.aic,
            "bic": mod.bic,
            "llf": mod.llf,
            "n_params": len(mod.params),
        })
    df_diff = len(spline.params) - len(linear.params)
    stat = 2 * (spline.llf - linear.llf)
    p_lrt = float(1 - chi2.cdf(stat, df_diff)) if df_diff > 0 else np.nan
    rows.append({"outcome": outcome, "model": "lrt_spline_vs_linear", "aic": stat, "bic": df_diff, "llf": p_lrt, "n_params": df_diff})
    return pd.DataFrame(rows)


def marginal_probs(models: Dict[str, sm.GLM], df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    base = df[COVARIATES + ["bmi"]].dropna().copy()
    rows: List[Dict[str, float]] = []
    for bmi_value in BMI_POINTS:
        sample = base.copy()
        sample["bmi"] = bmi_value
        prob = float(models["spline"].predict(sample).mean())
        rows.append({
            "outcome": outcome,
            "bmi": bmi_value,
            "predicted_prob": prob,
        })
    return pd.DataFrame(rows)


def or_vs_ref(model: sm.GLM, ref: float, targets: List[float], df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    design_info = model.model.data.design_info
    base = df[COVARIATES + ["bmi"]].dropna().median().to_dict()
    base["bmi"] = ref
    ref_exog = np.asarray(build_design_matrices([design_info], pd.DataFrame([base]))[0])
    cov = model.cov_params()
    rows = []
    for bmi_val in targets:
        row = base.copy()
        row["bmi"] = bmi_val
        exog_target = np.asarray(build_design_matrices([design_info], pd.DataFrame([row]))[0])
        delta = (exog_target - ref_exog)[0]
        lp_diff = float(delta @ model.params)
        var = float(delta @ cov @ delta)
        se = np.sqrt(var) if var > 0 else np.nan
        or_val = float(np.exp(lp_diff))
        ci_low = float(np.exp(lp_diff - 1.96 * se)) if np.isfinite(se) else np.nan
        ci_high = float(np.exp(lp_diff + 1.96 * se)) if np.isfinite(se) else np.nan
        rows.append({
            "outcome": outcome,
            "bmi_ref": ref,
            "bmi_target": bmi_val,
            "OR": or_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    return pd.DataFrame(rows)


def bp_curve_with_ci(model: sm.GLM, df: pd.DataFrame) -> pd.DataFrame:
    base = df[COVARIATES + ["bmi"]].dropna().median().to_dict()
    bmi_grid = np.linspace(18, 45, 200)
    rows = []
    for bmi_val in bmi_grid:
        row = base.copy()
        row["bmi"] = bmi_val
        pred = model.get_prediction(pd.DataFrame([row])).summary_frame()
        rows.append({
            "bmi": bmi_val,
            "pred": float(pred["mean"].iloc[0]),
            "ci_low": float(pred["mean_ci_lower"].iloc[0]),
            "ci_high": float(pred["mean_ci_upper"].iloc[0]),
        })
    return pd.DataFrame(rows)


def bp_diagnostics(model: sm.GLM, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    needed = ["best_performer", "bmi"] + COVARIATES
    data = df.dropna(subset=needed)
    y_true = data["best_performer"].astype(int)
    y_pred = model.predict(data)
    auc = roc_auc_score(y_true, y_pred)
    bins = pd.qcut(y_pred, 10, duplicates="drop")
    calib = data.assign(pred=y_pred, bin=bins).groupby("bin").agg(
        mean_pred=("pred", "mean"),
        observed=("best_performer", "mean"),
        n=("pred", "count"),
    ).reset_index()
    calib.insert(0, "bin_id", range(1, len(calib) + 1))
    diag = pd.DataFrame([{"metric": "AUC", "value": auc}])
    return calib, diag


def vif_table(model: sm.GLM) -> pd.DataFrame:
    exog = model.model.exog
    names = model.model.exog_names
    vif_rows = []
    for i, name in enumerate(names):
        if name.lower() == "intercept":
            continue
        vif_rows.append({"term": name, "vif": variance_inflation_factor(exog, i)})
    return pd.DataFrame(vif_rows)


def apply_volume_mode(df: pd.DataFrame, mode: str = "tertiles") -> pd.DataFrame:
    df = df.copy()
    if mode == "annual_threshold":
        if "year" not in df.columns and "ANNEE" in df.columns:
            df["year"] = pd.to_numeric(df["ANNEE"], errors="coerce")
        counts_cy = df.groupby(["CENTRE", "year"], observed=False)["CODE"].count().reset_index(name="vol_cy")
        mean_per_centre = counts_cy.groupby("CENTRE", observed=False)["vol_cy"].mean().reset_index(name="vol_mean_per_year")
        df = df.merge(mean_per_centre, on="CENTRE", how="left")
        df["centre_volume"] = df["vol_mean_per_year"].astype(float)
        bins = [-np.inf, 5, 10, np.inf]
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

    compare_rows = []
    marginal_rows = []
    or_rows = []
    param_tables = []

    for outcome in OUTCOMES:
        models = fit_models(df, outcome)
        compare_rows.append(compare_models(models, outcome))
        marginal_rows.append(marginal_probs(models, df, outcome))
        param_table = models["spline"].summary2().tables[1]
        param_table["outcome"] = outcome
        param_tables.append(param_table)
        if outcome == "best_performer":
            or_rows.append(or_vs_ref(models["spline"], ref=25, targets=[30, 35, 40], df=df, outcome=outcome))
            curve = bp_curve_with_ci(models["spline"], df)
            curve.to_csv(OUTPUT_DIR / "bmi_bp_curve.csv", index=False)
            calib, diag = bp_diagnostics(models["spline"], df)
            calib.to_csv(OUTPUT_DIR / "bmi_bp_calibration.csv", index=False)
            diag.to_csv(OUTPUT_DIR / "bmi_bp_diagnostics.csv", index=False)
            vif = vif_table(models["spline"])
            vif.to_csv(OUTPUT_DIR / "bmi_bp_vif.csv", index=False)
        else:
            or_rows.append(or_vs_ref(models["spline"], ref=25, targets=[30, 35, 40], df=df, outcome=outcome))

    pd.concat(compare_rows).to_csv(OUTPUT_DIR / "bmi_model_compare.csv", index=False)
    pd.concat(marginal_rows).to_csv(OUTPUT_DIR / "bmi_risk_predictions.csv", index=False)
    pd.concat(or_rows).to_csv(OUTPUT_DIR / "bmi_or_vs25.csv", index=False)
    pd.concat(param_tables).to_csv(OUTPUT_DIR / "bmi_risk_model_params.csv")
    print("Saved model comparisons, predictions, ORs, and diagnostics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMI intrinsic effect export")
    parser.add_argument("--volume-tier-mode", choices=["tertiles", "annual_threshold"], default="tertiles", help="How to define volume tiers")
    args = parser.parse_args()
    main(volume_tier_mode=args.volume_tier_mode)
