"""Paragraph 2 – BMI × centre expertise interaction with interpretability extensions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import patsy
from patsy import build_design_matrices
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from patsy import build_design_matrices

import sys
# Allow importing analysis/lib when running from export/
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "analysis"))
from lib.dataset import build_dataset  # noqa: E402
from lib.imputation import build_dataset_imputed  # noqa: E402
import argparse

OUTCOME = "best_performer"
BASE_COVARS = ["age", "asa_ge3", "sex_male", "malignant", "robotic", "splenectomy"]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

VOLUME_LEVELS = [10, 30, 60, 100]
BMI_LEVELS = [22, 27, 32, 37, 42]
VOLUME_THRESHOLDS = [20, 50, 75]
PAIRWISE_BMIS = [30, 35, 40]
BMI_GRID = np.linspace(18, 45, 200)


def build_design(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, np.ndarray, pd.DataFrame, patsy.DesignInfo]:
    mask = df[OUTCOME].notna() & df["bmi"].notna() & df["centre_volume"].notna()
    df = df.loc[mask].copy()
    df[OUTCOME] = df[OUTCOME].astype(int)
    df["centre_volume_scaled"] = df["centre_volume"] / 10.0
    formula = (
        f"{OUTCOME} ~ bmi + centre_volume_scaled + bmi:centre_volume_scaled + "
        + " + ".join(BASE_COVARS)
    )
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    y = y.iloc[:, 0]
    aligned = df.loc[y.index]
    groups = aligned["CENTRE"].astype("category").cat.codes.to_numpy()
    return y, X, groups, aligned, X.design_info


def apply_volume_mode(df: pd.DataFrame, mode: str = "tertiles") -> pd.DataFrame:
    """Compute centre volume metric and tiers according to the requested mode.

    - tertiles (default): total cases per centre across the whole dataset, tertile cut.
    - annual_threshold: cases per centre per year; tiers by thresholds: Low <5, Mid 5–10, High >10.
    - mipd_annual_threshold: mean annual minimally invasive pancreatectomies; tiers: Low <10, Mid 10–20, High >20 per year.
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
        # Tertiles on total cases per centre (default behavior)
        df["centre_volume"] = df.groupby("CENTRE")["CODE"].transform("count")
        try:
            df["centre_volume_cat"] = pd.qcut(df["centre_volume"], q=[0, 0.33, 0.66, 1.0], labels=["Low", "Mid", "High"], duplicates="drop")
        except Exception:
            # Fallback to equal-width if qcut fails
            df["centre_volume_cat"] = pd.cut(df["centre_volume"], bins=3, labels=["Low", "Mid", "High"]) 
    return df


def build_design_spline(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, np.ndarray, pd.DataFrame, patsy.DesignInfo]:
    """Same as build_design but with a spline on BMI and its interaction with volume.

    Formula: best_performer ~ bs(bmi, df=4, include_intercept=False) * centre_volume_scaled + covariates
    """
    mask = df[OUTCOME].notna() & df["bmi"].notna() & df["centre_volume"].notna()
    df = df.loc[mask].copy()
    # Ensure binary coding 0/1 for stability and to avoid any bool quirks
    df[OUTCOME] = df[OUTCOME].astype(int)
    df["centre_volume_scaled"] = df["centre_volume"] / 10.0
    formula = (
        f"{OUTCOME} ~ bs(bmi, df=4, include_intercept=False) * centre_volume_scaled + "
        + " + ".join(BASE_COVARS)
    )
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    y = y.iloc[:, 0]
    aligned = df.loc[y.index]
    groups = aligned["CENTRE"].astype("category").cat.codes.to_numpy()
    return y, X, groups, aligned, X.design_info


def fit_models(y, X, groups):
    glm = sm.GLM(y, X, family=Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    gee = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable()).fit()
    return glm, gee


def scenario_predictions(model, design_info: patsy.DesignInfo, base_covars: pd.Series, label: str) -> pd.DataFrame:
    """Predict on a small BMI×volume grid using the model design_info to build exog.

    base_covars: medians for BASE_COVARS (Series/dict).
    """
    rows: List[dict] = []
    base = base_covars.to_dict() if hasattr(base_covars, "to_dict") else dict(base_covars)
    for vol in VOLUME_LEVELS:
        vol_scaled = vol / 10.0
        for bmi in BMI_LEVELS:
            row = base.copy()
            row["bmi"] = bmi
            row["centre_volume_scaled"] = vol_scaled
            row["bmi:centre_volume_scaled"] = bmi * vol_scaled
            exog = np.asarray(build_design_matrices([design_info], pd.DataFrame([row]))[0])
            prob = float(model.get_prediction(exog).predicted_mean[0])
            rows.append({
                "model": label,
                "centre_volume": vol,
                "bmi": bmi,
                "prob_best": prob,
            })
    return pd.DataFrame(rows)


def pairwise_at_bmi(glm: sm.GLM, df: pd.DataFrame, design_info: patsy.DesignInfo) -> pd.DataFrame:
    base = df[BASE_COVARS + ["bmi"]].dropna().median().to_dict()
    # median volumes per tier
    vols = df.dropna(subset=["centre_volume_cat"]).groupby("centre_volume_cat")["centre_volume"].median().to_dict()
    low_vol = vols.get("Low", np.nan)
    high_vol = vols.get("High", np.nan)
    cov = glm.cov_params()
    results = []
    for bmi_val in PAIRWISE_BMIS:
        for tier, vol in [("Low", low_vol), ("High", high_vol)]:
            row = base.copy()
            row["bmi"] = bmi_val
            row["centre_volume_scaled"] = vol / 10.0 if pd.notna(vol) else np.nan
            row["bmi:centre_volume_scaled"] = row["bmi"] * row["centre_volume_scaled"]
            exog = np.asarray(build_design_matrices([design_info], pd.DataFrame([row]))[0])
            pred = glm.get_prediction(exog).summary_frame()
            results.append({
                "bmi": bmi_val,
                "tier": tier,
                "prob": float(pred["mean"].iloc[0]),
                "prob_ci_low": float(pred["mean_ci_lower"].iloc[0]),
                "prob_ci_high": float(pred["mean_ci_upper"].iloc[0]),
                "exog": exog[0],
            })
    out_rows = []
    for bmi_val in PAIRWISE_BMIS:
        low = next(r for r in results if r["bmi"] == bmi_val and r["tier"] == "Low")
        high = next(r for r in results if r["bmi"] == bmi_val and r["tier"] == "High")
        delta_prob = high["prob"] - low["prob"]
        # delta method for prob difference
        grad_low = low["prob"] * (1 - low["prob"]) * low["exog"]
        grad_high = high["prob"] * (1 - high["prob"]) * high["exog"]
        grad = grad_high - grad_low
        var = float(grad @ cov @ grad)
        se = np.sqrt(var) if var > 0 else np.nan
        z = delta_prob / se if se and se > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        ci_low = delta_prob - 1.96 * se if np.isfinite(se) else np.nan
        ci_high = delta_prob + 1.96 * se if np.isfinite(se) else np.nan
        out_rows.append({
            "bmi": bmi_val,
            "prob_low": low["prob"],
            "prob_high": high["prob"],
            "diff_high_minus_low": delta_prob,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p,
        })
    res = pd.DataFrame(out_rows)
    if not res.empty and res["p_value"].notna().any():
        res["p_adj"] = multipletests(res["p_value"].fillna(1.0), method="holm")[1]
    return res


def difference_curve(glm: sm.GLM, df: pd.DataFrame, design_info: patsy.DesignInfo) -> pd.DataFrame:
    """Delta method (cluster-robust) CIs for the marginal difference of probabilities.

    For each BMI grid value, compute mean predicted probability across the empirical
    covariate distribution for two volume tiers, then take the difference. The
    gradient for the mean is the average of p_i(1−p_i) x_i, so the gradient for the
    difference is (g_high − g_other). Use glm.cov_params() (cluster-robust) to get CIs.
    """
    tiers = df.dropna(subset=["centre_volume_cat"]).groupby("centre_volume_cat")["centre_volume"].median()
    cov = glm.cov_params()
    base_cov = df[BASE_COVARS].copy()

    def mean_pred_and_grad(bmi_val: float, vol: float) -> tuple[float, np.ndarray]:
        tmp = base_cov.copy()
        tmp["bmi"] = bmi_val
        tmp["centre_volume_scaled"] = vol / 10.0
        tmp["bmi:centre_volume_scaled"] = tmp["bmi"] * tmp["centre_volume_scaled"]
        exog = np.asarray(build_design_matrices([design_info], tmp)[0])
        lin = exog @ glm.params.to_numpy()
        p = 1.0 / (1.0 + np.exp(-lin))
        mean_p = float(p.mean())
        w = (p * (1 - p))[:, None]
        g = (w * exog).mean(axis=0)
        return mean_p, g

    rows: list[dict] = []
    vol_low = float(tiers.get("Low", np.nan))
    vol_mid = float(tiers.get("Mid", np.nan))
    vol_high = float(tiers.get("High", np.nan))
    for bmi_val in BMI_GRID:
        # High vs Low
        p_h, g_h = mean_pred_and_grad(bmi_val, vol_high)
        p_l, g_l = mean_pred_and_grad(bmi_val, vol_low)
        diff_hl = p_h - p_l
        var_hl = float((g_h - g_l) @ cov @ (g_h - g_l))
        se_hl = np.sqrt(max(var_hl, 0))
        rows.append({
            "bmi": bmi_val,
            "comparison": "High-Low",
            "diff": diff_hl,
            "ci_low": diff_hl - 1.96 * se_hl,
            "ci_high": diff_hl + 1.96 * se_hl,
        })
        # High vs Mid
        if not np.isnan(vol_mid):
            p_m, g_m = mean_pred_and_grad(bmi_val, vol_mid)
            diff_hm = p_h - p_m
            var_hm = float((g_h - g_m) @ cov @ (g_h - g_m))
            se_hm = np.sqrt(max(var_hm, 0))
            rows.append({
                "bmi": bmi_val,
                "comparison": "High-Mid",
                "diff": diff_hm,
                "ci_low": diff_hm - 1.96 * se_hm,
                "ci_high": diff_hm + 1.96 * se_hm,
            })
    return pd.DataFrame(rows)


def calibration_by_tier(glm: sm.GLM, df: pd.DataFrame, design_info: patsy.DesignInfo) -> pd.DataFrame:
    """Calibration: within each volume tier, group by deciles of predicted risk
    (using observed BMI/volume/covariates), report mean predicted, observed and n.
    """
    df_use = df.dropna(subset=[OUTCOME, "bmi", "centre_volume_cat"]).copy()
    exog = np.asarray(build_design_matrices([design_info], df_use[BASE_COVARS + ["bmi", "centre_volume_scaled"]])[0])
    df_use["pred"] = glm.predict(exog)
    rows = []
    for tier, grp in df_use.groupby("centre_volume_cat", observed=False):
        if grp.empty:
            continue
        # deciles of predicted risk
        grp = grp.copy()
        grp["decile"] = pd.qcut(grp["pred"], 10, labels=False, duplicates="drop")
        agg = grp.groupby("decile").agg(mean_pred=("pred", "mean"), mean_obs=(OUTCOME, "mean"), n=(OUTCOME, "count")).reset_index()
        agg.insert(0, "tier", tier)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["tier", "decile", "mean_pred", "mean_obs", "n"])

def or_per5_by_tier(glm: sm.GLM, df: pd.DataFrame, design_info: patsy.DesignInfo) -> pd.DataFrame:
    tiers = df.dropna(subset=["centre_volume_cat"]).groupby("centre_volume_cat")["centre_volume"].median()
    base = df[BASE_COVARS + ["bmi"]].dropna().median().to_dict()
    ref = base.copy()
    ref["bmi"] = ref_bmi = 25
    cov = glm.cov_params()
    rows = []
    for tier, vol_med in tiers.items():
        vol_scaled = vol_med / 10.0
        ref_row = ref.copy()
        ref_row["centre_volume_scaled"] = vol_scaled
        ref_row["bmi:centre_volume_scaled"] = ref_bmi * vol_scaled
        ref_exog = np.asarray(patsy.build_design_matrices([design_info], pd.DataFrame([ref_row]))[0])
        for delta_bmi in [5]:
            tgt_row = ref_row.copy()
            tgt_row["bmi"] = ref_bmi + delta_bmi
            tgt_row["bmi:centre_volume_scaled"] = (ref_bmi + delta_bmi) * vol_scaled
            tgt_exog = np.asarray(patsy.build_design_matrices([design_info], pd.DataFrame([tgt_row]))[0])
            delta = (tgt_exog - ref_exog)[0]
            lp = float(delta @ glm.params)
            var = float(delta @ cov @ delta)
            se = np.sqrt(var) if var > 0 else np.nan
            or_val = float(np.exp(lp))
            ci_low = float(np.exp(lp - 1.96 * se)) if np.isfinite(se) else np.nan
            ci_high = float(np.exp(lp + 1.96 * se)) if np.isfinite(se) else np.nan
            rows.append({"tier": tier, "vol_median": vol_med, "bmi_ref": ref_bmi, "delta_bmi": delta_bmi, "OR": or_val, "ci_low": ci_low, "ci_high": ci_high})
    return pd.DataFrame(rows)


def ard_bmi25_40(glm: sm.GLM, df: pd.DataFrame, design_info: patsy.DesignInfo) -> pd.DataFrame:
    tiers = df.dropna(subset=["centre_volume_cat"]).groupby("centre_volume_cat")["centre_volume"].median()
    base = df[BASE_COVARS + ["bmi"]].dropna().median().to_dict()
    rows = []
    for tier, vol_med in tiers.items():
        vol_scaled = vol_med / 10.0
        for bmi in [25, 40]:
            row = base.copy()
            row["bmi"] = bmi
            row["centre_volume_scaled"] = vol_scaled
            row["bmi:centre_volume_scaled"] = bmi * vol_scaled
            pred = glm.get_prediction(patsy.build_design_matrices([design_info], pd.DataFrame([row]))[0]).predicted_mean[0]
            rows.append({"tier": tier, "bmi": bmi, "prob": float(pred)})
    out = []
    for tier in tiers.index:
        p25 = next(r["prob"] for r in rows if r["tier"] == tier and r["bmi"] == 25)
        p40 = next(r["prob"] for r in rows if r["tier"] == tier and r["bmi"] == 40)
        out.append({"tier": tier, "prob_bmi25": p25, "prob_bmi40": p40, "ard_40_vs_25": p40 - p25})
    return pd.DataFrame(out)


def volume_threshold_tests(df: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    family = sm.families.Binomial()
    covars = " + ".join(c for c in BASE_COVARS if c != "centre_volume")
    for thr in VOLUME_THRESHOLDS:
        df_thr = df.copy()
        df_thr["highvol"] = (df_thr["centre_volume"] >= thr).astype(int)
        formula = f"{OUTCOME} ~ bmi + highvol + bmi:highvol + {covars}"
        y, X = patsy.dmatrices(formula, df_thr, return_type="dataframe")
        y = y.iloc[:, 0]
        groups_aligned = df_thr.loc[y.index, "CENTRE"].astype("category").cat.codes.to_numpy()
        try:
            glm = sm.GLM(y, X, family=family).fit(cov_type="cluster", cov_kwds={"groups": groups_aligned})
            p_int = glm.pvalues.get("bmi:highvol", np.nan)
        except Exception:
            # Fallback if robust covariance fails (e.g., singular matrix in small cells)
            glm = sm.GLM(y, X, family=family).fit()
            p_int = glm.pvalues.get("bmi:highvol", np.nan)
        rows.append({"threshold": thr, "p_interaction": float(p_int) if p_int is not None else np.nan})
    return pd.DataFrame(rows)


def three_way_tests(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    family = sm.families.Binomial()
    covars = " + ".join(c for c in BASE_COVARS if c not in {"malignant", "robotic"})
    df_use = df.dropna(subset=[OUTCOME, "bmi", "centre_volume", "malignant", "robotic"]).copy()
    df_use["centre_volume_scaled"] = df_use["centre_volume"] / 10.0
    groups = df_use["CENTRE"].astype("category").cat.codes.to_numpy()
    # Malignant
    formula_m = f"{OUTCOME} ~ bmi * centre_volume_scaled * malignant + {covars}"
    y, X = patsy.dmatrices(formula_m, df_use, return_type="dataframe")
    y = y.iloc[:, 0]
    glm_m = sm.GLM(y, X, family=family).fit(cov_type="cluster", cov_kwds={"groups": groups})
    p3_m = glm_m.pvalues.filter(like=":malignant").get("bmi:centre_volume_scaled:malignant[T.True]", np.nan)
    results.append({"interaction": "bmi*volume*malignant", "p_value": float(p3_m) if p3_m is not None else np.nan})
    # Robotic
    formula_r = f"{OUTCOME} ~ bmi * centre_volume_scaled * robotic + {covars}"
    y, X = patsy.dmatrices(formula_r, df_use, return_type="dataframe")
    y = y.iloc[:, 0]
    glm_r = sm.GLM(y, X, family=family).fit(cov_type="cluster", cov_kwds={"groups": groups})
    p3_r = glm_r.pvalues.filter(like=":robotic").get("bmi:centre_volume_scaled:robotic[T.True]", np.nan)
    results.append({"interaction": "bmi*volume*robotic", "p_value": float(p3_r) if p3_r is not None else np.nan})
    return pd.DataFrame(results)


def curve_by_tier(glm: sm.GLM, df: pd.DataFrame, design_info: patsy.DesignInfo) -> pd.DataFrame:
    """Marginal standardized curves with delta-method cluster-robust CIs.

    For each BMI grid value and volume tier, compute the average predicted
    probability across the empirical covariate mix, then use the delta method
    with the cluster-robust covariance of the GLM to derive CIs.
    """
    tiers = df.dropna(subset=["centre_volume_cat"]).groupby("centre_volume_cat")["centre_volume"].median()
    bmi_grid = np.linspace(18, 45, 200)
    cov = glm.cov_params()
    rows = []
    base_cov = df[BASE_COVARS].copy()
    for tier, vol_med in tiers.items():
        vol_scaled = vol_med / 10.0
        for bmi_val in bmi_grid:
            tmp = base_cov.copy()
            tmp["bmi"] = bmi_val
            tmp["centre_volume_scaled"] = vol_scaled
            tmp["bmi:centre_volume_scaled"] = bmi_val * vol_scaled
            exog = np.asarray(patsy.build_design_matrices([design_info], tmp)[0])
            lin = exog @ glm.params.to_numpy()
            p = 1.0 / (1.0 + np.exp(-lin))
            mean_p = float(p.mean())
            w = (p * (1 - p))[:, None]
            g = (w * exog).mean(axis=0)
            var = float(g @ cov @ g)
            se = np.sqrt(max(var, 0))
            rows.append({
                "tier": tier,
                "bmi": bmi_val,
                "prob": mean_p,
                "ci_low": mean_p - 1.96 * se,
                "ci_high": mean_p + 1.96 * se,
            })
    return pd.DataFrame(rows)


def main(use_imputed: bool = False, volume_tier_mode: str = "tertiles") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset_imputed() if use_imputed else build_dataset()
    df = apply_volume_mode(df, mode=volume_tier_mode)
    # record mode used
    (OUTPUT_DIR / "volume_mode.txt").write_text(volume_tier_mode)
    # Helper to run a full pipeline for a given design
    def run_pipeline(y, X, groups, aligned, design_info, suffix: str = "") -> None:
        glm, gee = fit_models(y, X, groups)
        # Params tables
        pd.DataFrame({"coef": glm.params, "std_err": glm.bse, "z": glm.tvalues, "p_value": glm.pvalues}).to_csv(OUTPUT_DIR / f"glm_parameters{suffix}.csv")
        pd.DataFrame({"coef": gee.params, "std_err": gee.bse, "z": gee.tvalues, "p_value": gee.pvalues}).to_csv(OUTPUT_DIR / f"gee_parameters{suffix}.csv")
        # Scenario predictions (legacy grid)
        base_covars = aligned[BASE_COVARS].median()
        glm_pred = scenario_predictions(glm, design_info, base_covars, "GLM")
        gee_pred = scenario_predictions(gee, design_info, base_covars, "GEE")
        pd.concat([glm_pred, gee_pred]).to_csv(OUTPUT_DIR / f"interaction_predictions{suffix}.csv", index=False)
        # Interpretability
        or_tier = or_per5_by_tier(glm, aligned, design_info)
        or_tier.to_csv(OUTPUT_DIR / f"or_per5_by_volume_tier{suffix}.csv", index=False)
        ard = ard_bmi25_40(glm, aligned, design_info)
        ard.to_csv(OUTPUT_DIR / f"ard_bmi25_40_by_tier{suffix}.csv", index=False)
        # Curves
        curves = curve_by_tier(glm, aligned, design_info)
        curves.to_csv(OUTPUT_DIR / f"bmi_volume_curves{suffix}.csv", index=False)
        diff = difference_curve(glm, aligned, design_info)
        diff.to_csv(OUTPUT_DIR / f"bmi_interaction_diff{suffix}.csv", index=False)
        pairwise = pairwise_at_bmi(glm, aligned, design_info)
        pairwise.to_csv(OUTPUT_DIR / f"pairwise_bmi_comparisons{suffix}.csv", index=False)
        # Calibration per tier
        calib = calibration_by_tier(glm, aligned, design_info)
        calib.to_csv(OUTPUT_DIR / f"calibration_tier{suffix}.csv", index=False)
        # p‑value interaction
        # Note: in spline design, the interaction is a block; we report the p‑value of the BMI×volume joint test via LRT is not here; keep GEE coef p if present.
        p_key = "bmi:centre_volume_scaled"
        gee_int_p = gee.pvalues.get(p_key, np.nan)
        # If the key is absent (e.g., spline interaction expands into multiple terms),
        # fall back to a Wald block test on GLM for all interaction parameters.
        if not np.isfinite(gee_int_p):
            names = list(glm.params.index)
            inter_idx = [i for i, nm in enumerate(names) if ":centre_volume_scaled" in nm and "bs(" in nm]
            if inter_idx:
                R = np.eye(len(names))[inter_idx]
                wt = glm.wald_test(R)
                gee_int_p = float(wt.pvalue)
        pd.DataFrame([{"gee_interaction_p": float(gee_int_p) if gee_int_p is not None else np.nan}]).to_csv(OUTPUT_DIR / f"interaction_p{suffix}.csv", index=False)

    # Linear design (original)
    y_lin, X_lin, groups_lin, aligned_lin, design_info_lin = build_design(df)
    run_pipeline(y_lin, X_lin, groups_lin, aligned_lin, design_info_lin, suffix="")

    # Spline design (new): files suffixed with _spline
    y_sp, X_sp, groups_sp, aligned_sp, design_info_sp = build_design_spline(df)
    run_pipeline(y_sp, X_sp, groups_sp, aligned_sp, design_info_sp, suffix="_spline")

    # Sensitivities not design‑specific (volume thresholds, 3‑way): keep from linear aligned
    thr = volume_threshold_tests(aligned_lin, groups_lin)
    thr.to_csv(OUTPUT_DIR / "volume_threshold_tests.csv", index=False)
    threeway = three_way_tests(aligned_lin)
    threeway.to_csv(OUTPUT_DIR / "threeway_interactions.csv", index=False)

    print("Saved interaction parameters (linear + spline), interpretability tables, and curves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMI × Volume interaction export")
    parser.add_argument("--use-imputed", action="store_true", help="Use MICE-imputed dataset")
    parser.add_argument("--volume-tier-mode", choices=["tertiles", "annual_threshold", "mipd_annual_threshold"], default="tertiles", help="How to define volume tiers")
    args = parser.parse_args()
    main(use_imputed=args.use_imputed, volume_tier_mode=args.volume_tier_mode)
