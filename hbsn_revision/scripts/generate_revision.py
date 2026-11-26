"""Generate HBSN revision tables, figures, and response drafts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .dataset import (
    APPROACH_ORDER,
    BMI3_ORDER,
    BMI6_ORDER,
    BMI_EXT_ORDER,
    LOS_75TH_THRESHOLD,
    REPO_ROOT,
    VOLUME_ORDER,
    build_dataset,
)


REVISION_ROOT = REPO_ROOT / "hbsn_revision"
TABLE_DIR = REVISION_ROOT / "tables"
FIGURE_DIR = REVISION_ROOT / "figures"
METHODS_DIR = REVISION_ROOT / "methods"
TEXT_DIR = REVISION_ROOT / "manuscript_text"
RESPONSE_DIR = REVISION_ROOT / "response_to_reviewers"
REVIEW_DOCX = REPO_ROOT / "HBSN-2026-0394-Reveiw Comments" / "HBSN-2026-0394-Reveiw Comments.docx"
AUTHOR = "Marc-Anthony Chouillard"
RNG_SEED = 20260612
NORD = {
    "background": "#FFFFFF",
    "axis": "#4C566A",
    "title": "#3B4252",
    "grid": "#D8DEE9",
    "blue": "#5E81AC",
    "cyan": "#88C0D0",
    "green": "#A3BE8C",
    "orange": "#D08770",
    "red": "#BF616A",
    "purple": "#B48EAD",
}

BASE_COVARS = ["age", "asa_ge3", "sex_male", "malignant", "robotic", "splenectomy"]
ANATOMIC_COVARS = BASE_COVARS + ["soft_pancreas", "duct_diameter"]
PS_COVARS = [
    "age",
    "bmi",
    "sex_male",
    "asa_ge3",
    "malignant",
    "robotic",
    "splenectomy",
    "soft_pancreas",
    "duct_diameter",
    "texture_missing",
    "duct_missing",
]
BINARY_OUTCOMES = [
    ("best_performer", "Best Performer", "favourable"),
    ("ideal_outcome", "Ideal Outcome", "favourable"),
    ("popf_bc", "CR-POPF B/C", "adverse"),
    ("conversion", "Conversion", "adverse"),
    ("major_clavien", "Clavien-Dindo >=III", "adverse"),
    ("major_morbidity_no_los", "Major morbidity excluding LOS", "adverse"),
]
OUTCOME_LABELS = {var: label for var, label, _direction in BINARY_OUTCOMES}


def apply_nord_theme(ax: plt.Axes) -> None:
    ax.figure.set_facecolor(NORD["background"])
    ax.set_facecolor(NORD["background"])
    ax.tick_params(colors=NORD["axis"], labelsize=10)
    ax.xaxis.label.set_color(NORD["axis"])
    ax.yaxis.label.set_color(NORD["axis"])
    ax.title.set_color(NORD["title"])
    for spine in ax.spines.values():
        spine.set_color(NORD["axis"])
    ax.grid(color=NORD["grid"], alpha=0.45)


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIGURE_DIR, METHODS_DIR, TEXT_DIR, RESPONSE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fmt_p(p: float | None) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_n_pct(events: int, total: int) -> str:
    if total == 0:
        return ""
    return f"{events}/{total} ({100 * events / total:.1f})"


def fmt_median_iqr(values: pd.Series, decimals: int = 1) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return ""
    return (
        f"{vals.median():.{decimals}f} "
        f"[{vals.quantile(0.25):.{decimals}f}-{vals.quantile(0.75):.{decimals}f}]"
    )


def categorical_p(table: np.ndarray) -> str:
    chi2, p, _dof, expected = stats.chi2_contingency(table)
    if table.shape == (2, 2) and np.min(expected) < 5:
        return fmt_p(stats.fisher_exact(table)[1])
    return fmt_p(p)


def kruskal_p(groups: list[pd.Series]) -> str:
    clean = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    clean = [g for g in clean if not g.empty]
    if len(clean) < 2:
        return ""
    return fmt_p(stats.kruskal(*clean).pvalue)


def add_binary_row(rows: list[dict[str, object]], df: pd.DataFrame, group_col: str, order: list[str], var: str, label: str) -> None:
    row = {"Metric": label}
    table = []
    for group in order:
        sub = df[df[group_col].astype(str) == group]
        total = int(sub.shape[0])
        events = int(sub[var].astype(bool).sum())
        row[group] = fmt_n_pct(events, total)
        table.append([events, total - events])
    row["p-value"] = categorical_p(np.asarray(table))
    rows.append(row)


def add_cont_row(rows: list[dict[str, object]], df: pd.DataFrame, group_col: str, order: list[str], var: str, label: str, decimals: int = 1) -> None:
    row = {"Metric": label}
    groups = []
    for group in order:
        vals = df.loc[df[group_col].astype(str) == group, var]
        row[group] = fmt_median_iqr(vals, decimals)
        groups.append(vals)
    row["p-value"] = kruskal_p(groups)
    rows.append(row)


def add_binary_threshold_row(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    group_col: str,
    order: list[str],
    var: str,
    threshold: float,
    label: str,
) -> None:
    row = {"Metric": label}
    table = []
    for group in order:
        sub = df[df[group_col].astype(str) == group]
        vals = pd.to_numeric(sub[var], errors="coerce").dropna()
        total = int(vals.shape[0])
        events = int(vals.ge(threshold).sum())
        row[group] = fmt_n_pct(events, total)
        table.append([events, total - events])
    row["p-value"] = categorical_p(np.asarray(table))
    rows.append(row)


def grouped_table(
    df: pd.DataFrame,
    group_col: str,
    order: list[str],
    include_baseline: bool = False,
    include_outcomes: bool = True,
    include_cci: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_row = {"Metric": "Number of patients", "p-value": ""}
    for group in order:
        n_row[group] = int((df[group_col].astype(str) == group).sum())
    rows.append(n_row)
    if include_baseline:
        add_cont_row(rows, df, group_col, order, "age", "Age, median [IQR], years")
        add_binary_row(rows, df, group_col, order, "female", "Female sex")
        add_binary_row(rows, df, group_col, order, "asa_ge3", "ASA >=III")
        add_binary_row(rows, df, group_col, order, "malignant", "Malignant indication")
        add_binary_row(rows, df, group_col, order, "robotic", "Robotic approach")
        add_binary_row(rows, df, group_col, order, "splenectomy", "Splenectomy")
        add_binary_row(rows, df, group_col, order, "soft_pancreas", "Soft pancreas")
        add_cont_row(rows, df, group_col, order, "duct_diameter", "Pancreatic duct diameter, median [IQR], mm", 1)
    if not include_outcomes:
        return pd.DataFrame(rows)
    add_cont_row(rows, df, group_col, order, "los_days", "Length of stay, median [IQR], days")
    if include_cci:
        add_cont_row(rows, df, group_col, order, "cci_postop", "CCI, median [IQR]")
    for var, label, _direction in BINARY_OUTCOMES:
        add_binary_row(rows, df, group_col, order, var, label)
    add_binary_row(rows, df, group_col, order, "mortality", "Postoperative mortality")
    return pd.DataFrame(rows)


def fit_clustered_glm(formula: str, data: pd.DataFrame, family=None, weights: pd.Series | None = None):
    family = family or sm.families.Binomial()
    model = smf.glm(formula=formula, data=data, family=family, freq_weights=weights)
    return model.fit(cov_type="cluster", cov_kwds={"groups": data["CENTRE"]})


def or_from_result(result, term: str) -> tuple[float, float, float, float]:
    beta = float(result.params[term])
    se = float(result.bse[term])
    p = float(result.pvalues[term])
    return np.exp(beta), np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se), p


def endpoint_robustness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = df.copy()
    for col in ["asa_ge3", "sex_male", "malignant", "robotic", "splenectomy", "high_volume"] + [v for v, _, _ in BINARY_OUTCOMES]:
        data[col] = data[col].astype(int)
    data["bmi_per5"] = data["bmi"] / 5
    data["centre_volume_per5"] = data["centre_volume_annual"] / 5
    covars = " + ".join(BASE_COVARS)
    needed = ["bmi_per5", "centre_volume_per5", "CENTRE"] + BASE_COVARS

    for endpoint, label, direction in BINARY_OUTCOMES:
        use = data.dropna(subset=needed + [endpoint]).copy()
        main_fit = fit_clustered_glm(f"{endpoint} ~ bmi_per5 + centre_volume_per5 + {covars}", use)
        interaction_fit = fit_clustered_glm(f"{endpoint} ~ bmi_per5 * centre_volume_per5 + {covars}", use)
        bmi_or, bmi_lo, bmi_hi, bmi_p = or_from_result(main_fit, "bmi_per5")
        int_or, int_lo, int_hi, int_p = or_from_result(interaction_fit, "bmi_per5:centre_volume_per5")
        rows.append(
            {
                "Endpoint": label,
                "Endpoint type": direction,
                "n": int(use.shape[0]),
                "events": int(use[endpoint].sum()),
                "BMI OR per +5 kg/m2": f"{bmi_or:.2f}",
                "BMI 95% CI": f"{bmi_lo:.2f}-{bmi_hi:.2f}",
                "BMI p-value": fmt_p(bmi_p),
                "BMI x volume OR per +5 BMI and +5 annual cases": f"{int_or:.2f}",
                "Interaction 95% CI": f"{int_lo:.2f}-{int_hi:.2f}",
                "Interaction p-value": fmt_p(int_p),
            }
        )

    cci_use = data.dropna(subset=needed + ["cci_postop"]).copy()
    cci_use["log_cci"] = np.log1p(cci_use["cci_postop"])
    cci_main_fit = fit_clustered_glm(
        f"log_cci ~ bmi_per5 + centre_volume_per5 + {covars}",
        cci_use,
        family=sm.families.Gaussian(),
    )
    cci_interaction_fit = fit_clustered_glm(
        f"log_cci ~ bmi_per5 * centre_volume_per5 + {covars}",
        cci_use,
        family=sm.families.Gaussian(),
    )
    beta = float(cci_main_fit.params["bmi_per5"])
    beta_se = float(cci_main_fit.bse["bmi_per5"])
    beta_int = float(cci_interaction_fit.params["bmi_per5:centre_volume_per5"])
    beta_int_se = float(cci_interaction_fit.bse["bmi_per5:centre_volume_per5"])
    bmi_lo = 100 * (np.exp(beta - 1.96 * beta_se) - 1)
    bmi_hi = 100 * (np.exp(beta + 1.96 * beta_se) - 1)
    int_lo = 100 * (np.exp(beta_int - 1.96 * beta_int_se) - 1)
    int_hi = 100 * (np.exp(beta_int + 1.96 * beta_int_se) - 1)
    rows.append(
        {
            "Endpoint": "CCI (log1p)",
            "Endpoint type": "continuous",
            "n": int(cci_use.shape[0]),
            "events": "",
            "BMI OR per +5 kg/m2": f"{100 * (np.exp(beta) - 1):.1f}% change",
            "BMI 95% CI": f"{bmi_lo:.1f}% to {bmi_hi:.1f}%",
            "BMI p-value": fmt_p(float(cci_main_fit.pvalues["bmi_per5"])),
            "BMI x volume OR per +5 BMI and +5 annual cases": f"{100 * (np.exp(beta_int) - 1):.1f}% change",
            "Interaction 95% CI": f"{int_lo:.1f}% to {int_hi:.1f}%",
            "Interaction p-value": fmt_p(float(cci_interaction_fit.pvalues["bmi_per5:centre_volume_per5"])),
        }
    )
    return pd.DataFrame(rows)


def anatomic_complete_case(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = df.dropna(subset=["soft_pancreas", "duct_diameter", "age", "bmi", "centre_volume_annual"]).copy()
    data = data[data["texture_available"]].copy()
    for col in ["asa_ge3", "sex_male", "malignant", "robotic", "splenectomy", "soft_pancreas"] + [v for v, _, _ in BINARY_OUTCOMES]:
        data[col] = data[col].astype(int)
    data["bmi_per5"] = data["bmi"] / 5
    data["centre_volume_per5"] = data["centre_volume_annual"] / 5
    covars = " + ".join(ANATOMIC_COVARS)
    needed = ["bmi_per5", "centre_volume_per5", "CENTRE"] + ANATOMIC_COVARS
    for endpoint, label, _direction in [
        ("best_performer", "Best Performer", "favourable"),
        ("popf_bc", "CR-POPF B/C", "adverse"),
        ("major_morbidity_no_los", "Major morbidity excluding LOS", "adverse"),
    ]:
        use = data.dropna(subset=needed + [endpoint]).copy()
        main_fit = fit_clustered_glm(f"{endpoint} ~ bmi_per5 + centre_volume_per5 + {covars}", use)
        interaction_fit = fit_clustered_glm(f"{endpoint} ~ bmi_per5 * centre_volume_per5 + {covars}", use)
        bmi_or, bmi_lo, bmi_hi, bmi_p = or_from_result(main_fit, "bmi_per5")
        int_or, int_lo, int_hi, int_p = or_from_result(interaction_fit, "bmi_per5:centre_volume_per5")
        rows.append(
            {
                "Endpoint": label,
                "n complete cases": int(use.shape[0]),
                "events": int(use[endpoint].sum()),
                "BMI OR per +5": f"{bmi_or:.2f}",
                "BMI 95% CI": f"{bmi_lo:.2f}-{bmi_hi:.2f}",
                "BMI p-value": fmt_p(bmi_p),
                "Interaction OR": f"{int_or:.2f}",
                "Interaction 95% CI": f"{int_lo:.2f}-{int_hi:.2f}",
                "Interaction p-value": fmt_p(int_p),
            }
        )
    return pd.DataFrame(rows)


def approach_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = grouped_table(df, "approach", APPROACH_ORDER, include_baseline=True)
    data = df.copy()
    for col in ["robotic", "asa_ge3", "sex_male", "malignant", "splenectomy", "best_performer", "major_morbidity_no_los"]:
        data[col] = data[col].astype(int)
    data["bmi_per5"] = data["bmi"] / 5
    data["centre_volume_per5"] = data["centre_volume_annual"] / 5
    covars = "age + asa_ge3 + sex_male + malignant + splenectomy"
    rows = []
    use = data.dropna(subset=["best_performer", "bmi_per5", "centre_volume_per5", "robotic", "CENTRE", "age"]).copy()
    fit = fit_clustered_glm(f"best_performer ~ bmi_per5 * centre_volume_per5 * robotic + {covars}", use)
    term = "bmi_per5:centre_volume_per5:robotic"
    three_or, three_lo, three_hi, three_p = or_from_result(fit, term)
    rows.append(
        {
            "Model": "Three-way interaction",
            "Subgroup": "All patients",
            "Endpoint": "Best Performer",
            "n": int(use.shape[0]),
            "Term": "BMI x volume x robotic",
            "Interaction estimate": f"{three_or:.2f}",
            "95% CI": f"{three_lo:.2f}-{three_hi:.2f}",
            "p-value": fmt_p(three_p),
        }
    )
    for approach in APPROACH_ORDER:
        sub = data[data["approach"].astype(str) == approach].dropna(subset=["best_performer", "bmi_per5", "centre_volume_per5", "CENTRE", "age"]).copy()
        if sub.shape[0] < 100 or sub["best_performer"].sum() < 10:
            rows.append(
                {
                    "Model": "Stratified",
                    "Subgroup": approach,
                    "Endpoint": "Best Performer",
                    "n": int(sub.shape[0]),
                    "Term": "BMI x volume",
                    "Interaction estimate": "",
                    "95% CI": "",
                    "p-value": "underpowered",
                }
            )
            continue
        fit_sub = fit_clustered_glm(f"best_performer ~ bmi_per5 * centre_volume_per5 + {covars}", sub)
        int_or, int_lo, int_hi, int_p = or_from_result(fit_sub, "bmi_per5:centre_volume_per5")
        rows.append(
            {
                "Model": "Stratified",
                "Subgroup": approach,
                "Endpoint": "Best Performer",
                "n": int(sub.shape[0]),
                "Term": "BMI x volume",
                "Interaction estimate": f"{int_or:.2f}",
                "95% CI": f"{int_lo:.2f}-{int_hi:.2f}",
                "p-value": fmt_p(int_p),
            }
        )
    return table, pd.DataFrame(rows)


def _prepare_ps_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["texture_missing"] = ~data["texture_available"]
    data["duct_missing"] = ~data["duct_available"]
    data["duct_diameter"] = data["duct_diameter"].fillna(data["duct_diameter"].median())
    data["soft_pancreas"] = np.where(data["texture_available"], data["soft_pancreas"].astype(int), 0)
    for col in ["sex_male", "asa_ge3", "malignant", "robotic", "splenectomy", "texture_missing", "duct_missing", "high_volume"]:
        data[col] = data[col].astype(int)
    data = data.dropna(subset=["age", "bmi"] + PS_COVARS + ["high_volume"]).copy()
    return data


def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
    return float(np.sum(x * w) / np.sum(w))


def _smd(x: pd.Series, treat: pd.Series, weights: pd.Series | None = None) -> float:
    x = pd.to_numeric(x, errors="coerce")
    treat = treat.astype(bool)
    if weights is None:
        m1 = float(x[treat].mean())
        m0 = float(x[~treat].mean())
        v1 = float(x[treat].var())
        v0 = float(x[~treat].var())
    else:
        w = weights
        m1 = _weighted_mean(x[treat], w[treat])
        m0 = _weighted_mean(x[~treat], w[~treat])
        v1 = _weighted_mean((x[treat] - m1) ** 2, w[treat])
        v0 = _weighted_mean((x[~treat] - m0) ** 2, w[~treat])
    pooled = np.sqrt((v1 + v0) / 2)
    return float((m1 - m0) / pooled) if pooled > 0 else 0.0


def _matched_pairs(data: pd.DataFrame, caliper: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated = data[data["high_volume"] == 1].copy().sort_values("ps")
    control = data[data["high_volume"] == 0].copy().sort_values("ps")
    available = set(control.index)
    pairs = []
    for treated_idx, treated_row in treated.iterrows():
        if not available:
            break
        control_subset = control.loc[list(available)]
        distances = (control_subset["logit_ps"] - treated_row["logit_ps"]).abs()
        control_idx = distances.idxmin()
        distance = float(distances.loc[control_idx])
        if distance <= caliper:
            pair_id = len(pairs) + 1
            pairs.append(
                {
                    "pair_id": pair_id,
                    "treated_index": treated_idx,
                    "control_index": control_idx,
                    "abs_logit_ps_distance": distance,
                }
            )
            available.remove(control_idx)
    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        return pd.DataFrame(), pair_df
    matched_rows = []
    for row in pair_df.itertuples(index=False):
        high = data.loc[row.treated_index].copy()
        lowmid = data.loc[row.control_index].copy()
        high["match_pair_id"] = int(row.pair_id)
        lowmid["match_pair_id"] = int(row.pair_id)
        matched_rows.extend([high, lowmid])
    matched = pd.DataFrame(matched_rows)
    return matched, pair_df


def _matched_binary_effect(matched: pd.DataFrame, endpoint: str) -> dict[str, object]:
    high_events = 0
    high_total = 0
    control_events = 0
    control_total = 0
    high_only = 0
    control_only = 0
    complete_pairs = 0
    for _pair_id, pair in matched.groupby("match_pair_id"):
        if pair.shape[0] != 2 or pair[endpoint].isna().any():
            continue
        high = pair[pair["high_volume"] == 1]
        control = pair[pair["high_volume"] == 0]
        if high.empty or control.empty:
            continue
        high_val = int(bool(high.iloc[0][endpoint]))
        control_val = int(bool(control.iloc[0][endpoint]))
        high_events += high_val
        control_events += control_val
        high_total += 1
        control_total += 1
        complete_pairs += 1
        if high_val == 1 and control_val == 0:
            high_only += 1
        elif high_val == 0 and control_val == 1:
            control_only += 1
    if high_only == 0 or control_only == 0:
        b = high_only + 0.5
        c = control_only + 0.5
    else:
        b = high_only
        c = control_only
    matched_or = b / c
    se = np.sqrt(1 / b + 1 / c)
    lo = np.exp(np.log(matched_or) - 1.96 * se)
    hi = np.exp(np.log(matched_or) + 1.96 * se)
    discordant = high_only + control_only
    p = stats.binomtest(min(high_only, control_only), discordant, 0.5).pvalue if discordant else np.nan
    return {
        "Matched pairs": int(complete_pairs),
        "High-volume": fmt_n_pct(high_events, high_total),
        "Matched Low/Mid-volume": fmt_n_pct(control_events, control_total),
        "Matched-pair OR": f"{matched_or:.2f}",
        "95% CI": f"{lo:.2f}-{hi:.2f}",
        "McNemar p-value": fmt_p(p),
        "High-only discordant pairs": int(high_only),
        "Low/Mid-only discordant pairs": int(control_only),
    }


def propensity_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _prepare_ps_data(df)
    X = data[PS_COVARS].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    y = data["high_volume"].astype(int).to_numpy()
    ps_model = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=RNG_SEED)
    ps_model.fit(Xs, y)
    ps = np.clip(ps_model.predict_proba(Xs)[:, 1], 0.01, 0.99)
    p_treat = y.mean()
    weights = np.where(y == 1, p_treat / ps, (1 - p_treat) / (1 - ps))
    lo, hi = np.quantile(weights, [0.01, 0.99])
    data["iptw"] = np.clip(weights, lo, hi)
    data["ps"] = ps
    data["logit_ps"] = np.log(ps / (1 - ps))
    caliper = 0.2 * float(np.std(data["logit_ps"], ddof=1))
    matched, pair_df = _matched_pairs(data, caliper)

    balance_rows = []
    for cov in PS_COVARS:
        row = {
            "Covariate": cov,
            "SMD before adjustment": f"{_smd(data[cov], data['high_volume']):.3f}",
            "SMD after IPTW": f"{_smd(data[cov], data['high_volume'], data['iptw']):.3f}",
            "SMD after PSM": "",
        }
        if not matched.empty:
            row["SMD after PSM"] = f"{_smd(matched[cov], matched['high_volume']):.3f}"
        balance_rows.append(row)
    balance = pd.DataFrame(balance_rows)

    outcome_rows = []
    ps_endpoints = ["best_performer", "major_morbidity_no_los", "popf_bc", "major_clavien", "conversion"]
    for endpoint in ps_endpoints:
        data[endpoint] = data[endpoint].astype(int)
        fit = fit_clustered_glm(
            f"{endpoint} ~ high_volume + bmi + age + asa_ge3 + sex_male + malignant + robotic + splenectomy",
            data.dropna(subset=[endpoint, "CENTRE"]),
        )
        fit_w = fit_clustered_glm(
            f"{endpoint} ~ high_volume + bmi + age + asa_ge3 + sex_male + malignant + robotic + splenectomy",
            data.dropna(subset=[endpoint, "CENTRE"]),
            weights=data.dropna(subset=[endpoint, "CENTRE"])["iptw"],
        )
        for label, model in [("Unweighted", fit), ("IPTW", fit_w)]:
            orv, lo_ci, hi_ci, p = or_from_result(model, "high_volume")
            outcome_rows.append(
                {
                    "Endpoint": OUTCOME_LABELS.get(endpoint, endpoint),
                    "Model": label,
                    "n": int(data.shape[0]),
                    "High-volume OR": f"{orv:.2f}",
                    "95% CI": f"{lo_ci:.2f}-{hi_ci:.2f}",
                    "p-value": fmt_p(p),
                }
            )
    outcomes = pd.DataFrame(outcome_rows)

    psm_rows = []
    if not matched.empty:
        for endpoint in ps_endpoints:
            matched[endpoint] = matched[endpoint].astype(int)
            effect = _matched_binary_effect(matched, endpoint)
            effect["Endpoint"] = OUTCOME_LABELS.get(endpoint, endpoint)
            psm_rows.append(effect)
    psm_outcomes = pd.DataFrame(psm_rows)
    if not psm_outcomes.empty:
        columns = [
            "Endpoint",
            "Matched pairs",
            "High-volume",
            "Matched Low/Mid-volume",
            "Matched-pair OR",
            "95% CI",
            "McNemar p-value",
            "High-only discordant pairs",
            "Low/Mid-only discordant pairs",
        ]
        psm_outcomes = psm_outcomes[columns]

    treated = data[data["high_volume"] == 1]
    matched_high = int(matched[matched["high_volume"] == 1].shape[0]) if not matched.empty else 0
    matched_control = int(matched[matched["high_volume"] == 0].shape[0]) if not matched.empty else 0
    match_summary = pd.DataFrame(
        [
            {"Metric": "IPTW analytic n", "Value": int(data.shape[0])},
            {"Metric": "Treated high-volume patients", "Value": int(treated.shape[0])},
            {"Metric": "Matching method", "Value": "1:1 nearest-neighbour without replacement"},
            {"Metric": "Caliper on logit propensity score", "Value": f"{caliper:.3f}"},
            {"Metric": "Matched high-volume patients", "Value": matched_high},
            {"Metric": "Matched Low/Mid-volume controls", "Value": matched_control},
            {"Metric": "Unmatched high-volume patients", "Value": int(treated.shape[0] - matched_high)},
            {
                "Metric": "Median absolute logit PS distance",
                "Value": "" if pair_df.empty else f"{float(pair_df['abs_logit_ps_distance'].median()):.3f}",
            },
        ]
    )
    return balance, outcomes, psm_outcomes, match_summary


def cci_group_tables(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col, order, label in [
        ("bmi3", BMI3_ORDER, "BMI category"),
        ("centre_volume_tier", VOLUME_ORDER, "Centre volume"),
    ]:
        groups = []
        row = {"Stratum": label, "p-value": ""}
        for group in order:
            vals = df.loc[df[group_col].astype(str) == group, "cci_postop"]
            row[group] = fmt_median_iqr(vals)
            groups.append(vals)
        row["p-value"] = kruskal_p(groups)
        rows.append(row)
    return pd.DataFrame(rows)


def centre_volume_definition(df: pd.DataFrame) -> pd.DataFrame:
    centre = (
        df[["CENTRE", "centre_volume_tier", "centre_volume_annual", "mipd_mean_annual"]]
        .drop_duplicates("CENTRE")
        .dropna(subset=["centre_volume_tier", "centre_volume_annual"])
        .copy()
    )
    rows = []
    for tier in VOLUME_ORDER:
        sub = centre[centre["centre_volume_tier"].astype(str) == tier]
        rows.append(
            {
                "Centre-volume tier": tier,
                "Definition": {"Low": "<5 MILP/year", "Mid": "5-10 MILP/year", "High": ">10 MILP/year"}[tier],
                "Number of centres": int(sub.shape[0]),
                "Annual MILP volume, median [IQR]": fmt_median_iqr(sub["centre_volume_annual"], 2),
                "Annual MILP volume, range": (
                    ""
                    if sub.empty
                    else f"{sub['centre_volume_annual'].min():.2f}-{sub['centre_volume_annual'].max():.2f}"
                ),
                "Annual MIPD volume, median [IQR]": fmt_median_iqr(sub["mipd_mean_annual"], 2),
            }
        )
    return pd.DataFrame(rows)


def case_mix_by_volume(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_row = {"Metric": "Number of patients", "p-value": ""}
    for group in VOLUME_ORDER:
        n_row[group] = int((df["centre_volume_tier"].astype(str) == group).sum())
    rows.append(n_row)
    add_cont_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "bmi", "BMI, median [IQR], kg/m2", 1)
    add_binary_threshold_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "bmi", 30, "BMI >=30 kg/m2")
    add_binary_threshold_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "bmi", 35, "BMI >=35 kg/m2")
    add_cont_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "age", "Age, median [IQR], years", 1)
    add_binary_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "female", "Female sex")
    add_binary_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "asa_ge3", "ASA >=III")
    add_binary_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "malignant", "Malignant indication")
    add_binary_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "robotic", "Robotic approach")
    add_binary_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "splenectomy", "Splenectomy")
    add_binary_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "soft_pancreas", "Soft pancreas")
    add_cont_row(rows, df, "centre_volume_tier", VOLUME_ORDER, "duct_diameter", "Pancreatic duct diameter, median [IQR], mm", 1)
    return pd.DataFrame(rows)


def volume_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, mask in [
        ("Primary High >10 annual MILP", df["centre_volume_tier"].astype(str).eq("High")),
        ("Strict High >20 annual MIPD", df["strict_mipd_high"]),
    ]:
        high = df[mask]
        other = df[~mask]
        rows.append(
            {
                "Definition": label,
                "High-volume centres": int(high["CENTRE"].nunique()),
                "High-volume patients": int(high.shape[0]),
                "Other patients": int(other.shape[0]),
                "Best Performer high": fmt_n_pct(int(high["best_performer"].sum()), int(high.shape[0])),
                "Best Performer other": fmt_n_pct(int(other["best_performer"].sum()), int(other.shape[0])),
                "Best Performer p-value": categorical_p(
                    np.asarray(
                        [
                            [int(high["best_performer"].sum()), int(high.shape[0] - high["best_performer"].sum())],
                            [int(other["best_performer"].sum()), int(other.shape[0] - other["best_performer"].sum())],
                        ]
                    )
                ),
                "CR-POPF B/C high": fmt_n_pct(int(high["popf_bc"].sum()), int(high.shape[0])),
                "CR-POPF B/C other": fmt_n_pct(int(other["popf_bc"].sum()), int(other.shape[0])),
                "CR-POPF B/C p-value": categorical_p(
                    np.asarray(
                        [
                            [int(high["popf_bc"].sum()), int(high.shape[0] - high["popf_bc"].sum())],
                            [int(other["popf_bc"].sum()), int(other.shape[0] - other["popf_bc"].sum())],
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def bmi_extremes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in BMI_EXT_ORDER:
        sub = df[df["bmi_ext"].astype(str) == group]
        rows.append(
            {
                "BMI category": group,
                "n": int(sub.shape[0]),
                "Best Performer": fmt_n_pct(int(sub["best_performer"].sum()), int(sub.shape[0])),
                "Ideal Outcome": fmt_n_pct(int(sub["ideal_outcome"].sum()), int(sub.shape[0])),
                "CCI, median [IQR]": fmt_median_iqr(sub["cci_postop"]),
                "CR-POPF B/C": fmt_n_pct(int(sub["popf_bc"].sum()), int(sub.shape[0])),
                "Mortality": fmt_n_pct(int(sub["mortality"].sum()), int(sub.shape[0])),
            }
        )
    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    variables = [
        ("Pancreatic texture", "texture_available"),
        ("Duct diameter", "duct_available"),
        ("CCI", "cci_postop"),
        ("Robotic/laparoscopic approach", "approach"),
    ]
    rows = []
    for label, col in variables:
        if col.endswith("_available"):
            available = int(df[col].sum())
        else:
            available = int(df[col].notna().sum())
        rows.append({"Variable": label, "Available n/N (%)": fmt_n_pct(available, int(df.shape[0]))})
    return pd.DataFrame(rows)


def _parse_ci(ci: str) -> tuple[float, float]:
    lo, hi = str(ci).split("-", maxsplit=1)
    return float(lo), float(hi)


def _parse_percent_estimate(value: str) -> float:
    return 1 + float(str(value).replace("% change", "").strip()) / 100


def _parse_percent_ci(ci: str) -> tuple[float, float]:
    lo, hi = str(ci).replace("%", "").split(" to ", maxsplit=1)
    return 1 + float(lo) / 100, 1 + float(hi) / 100


def sensitivity_interaction_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    endpoint_table = tables["endpoint_robustness_models"]
    for _, row in endpoint_table.iterrows():
        if row["Endpoint"] == "CCI (log1p)":
            estimate = _parse_percent_estimate(row["BMI x volume OR per +5 BMI and +5 annual cases"])
            ci_low, ci_high = _parse_percent_ci(row["Interaction 95% CI"])
            scale = "Relative ratio from log(CCI+1)"
        else:
            estimate = float(row["BMI x volume OR per +5 BMI and +5 annual cases"])
            ci_low, ci_high = _parse_ci(row["Interaction 95% CI"])
            scale = "OR"
        rows.append(
            {
                "Analysis": "Endpoint robustness",
                "Endpoint": row["Endpoint"],
                "Term": "BMI x annual MILP volume",
                "Estimate": f"{estimate:.3f}",
                "95% CI": f"{ci_low:.3f}-{ci_high:.3f}",
                "p-value": row["Interaction p-value"],
                "Scale": scale,
            }
        )

    anatomic = tables["anatomic_complete_case_sensitivity"]
    for _, row in anatomic.iterrows():
        ci_low, ci_high = _parse_ci(row["Interaction 95% CI"])
        rows.append(
            {
                "Analysis": "Anatomic complete-case",
                "Endpoint": row["Endpoint"],
                "Term": "BMI x annual MILP volume",
                "Estimate": f"{float(row['Interaction OR']):.3f}",
                "95% CI": f"{ci_low:.3f}-{ci_high:.3f}",
                "p-value": row["Interaction p-value"],
                "Scale": "OR",
            }
        )

    approach = tables["approach_interactions"]
    for _, row in approach.iterrows():
        if not str(row.get("Interaction estimate", "")).strip():
            continue
        ci_low, ci_high = _parse_ci(row["95% CI"])
        rows.append(
            {
                "Analysis": f"Approach: {row['Subgroup']}",
                "Endpoint": row["Endpoint"],
                "Term": row["Term"],
                "Estimate": f"{float(row['Interaction estimate']):.3f}",
                "95% CI": f"{ci_low:.3f}-{ci_high:.3f}",
                "p-value": row["p-value"],
                "Scale": "OR",
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, index=False)


def make_forest_plot(endpoint_table: pd.DataFrame) -> None:
    plot = endpoint_table[endpoint_table["Endpoint"] != "CCI (log1p)"].copy()
    vals = plot["BMI x volume OR per +5 BMI and +5 annual cases"].astype(float)
    ci = plot["Interaction 95% CI"].str.split("-", expand=True).astype(float)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(vals, y, xerr=[vals - ci[0], ci[1] - vals], fmt="o", color=NORD["blue"], ecolor=NORD["cyan"], capsize=3)
    ax.axvline(1, color=NORD["axis"], linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["Endpoint"])
    ax.set_xlabel("BMI x volume interaction OR")
    ax.set_title("Endpoint robustness: BMI x volume interaction")
    apply_nord_theme(ax)
    ax.grid(axis="x", color=NORD["grid"], alpha=0.45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "endpoint_interaction_forest.png", dpi=300, facecolor=NORD["background"])
    fig.savefig(FIGURE_DIR / "endpoint_interaction_forest.svg", facecolor=NORD["background"])
    plt.close(fig)


def make_sensitivity_interaction_plot(summary: pd.DataFrame) -> None:
    plot = summary.copy()
    plot["estimate"] = plot["Estimate"].astype(float)
    ci = plot["95% CI"].str.split("-", expand=True).astype(float)
    plot["ci_low"] = ci[0]
    plot["ci_high"] = ci[1]
    plot["label"] = plot["Analysis"] + " | " + plot["Endpoint"]
    plot = plot.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    colors = []
    for p in plot["p-value"]:
        p_str = str(p)
        is_sig = p_str.startswith("<") or (p_str not in {"", "underpowered"} and float(p_str) < 0.05)
        colors.append(NORD["green"] if is_sig else NORD["blue"])
    fig_height = max(6, 0.42 * len(plot) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.errorbar(
        plot["estimate"],
        y,
        xerr=[plot["estimate"] - plot["ci_low"], plot["ci_high"] - plot["estimate"]],
        fmt="none",
        ecolor=NORD["cyan"],
        elinewidth=1.6,
        capsize=3,
        alpha=0.95,
    )
    ax.scatter(plot["estimate"], y, s=42, c=colors, zorder=3)
    ax.axvline(1, color=NORD["axis"], linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel("Interaction estimate per +5 BMI and +5 annual MILP volume")
    ax.set_title("Sensitivity analyses: BMI x centre-volume interaction")
    xmin = max(0.45, float(plot["ci_low"].min()) * 0.9)
    xmax = min(2.2, float(plot["ci_high"].max()) * 1.1)
    ax.set_xlim(xmin, xmax)
    ticks = [tick for tick in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] if xmin <= tick <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.2g}" if tick != 1.0 else "1.0" for tick in ticks])
    apply_nord_theme(ax)
    ax.grid(axis="x", color=NORD["grid"], alpha=0.45)
    for yi, p in zip(y, plot["p-value"]):
        ax.text(xmax - 0.02 * (xmax - xmin), yi, f"p={p}", va="center", ha="right", fontsize=8, color=NORD["axis"])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sensitivity_interaction_forest.png", dpi=300, facecolor=NORD["background"])
    fig.savefig(FIGURE_DIR / "sensitivity_interaction_forest.svg", facecolor=NORD["background"])
    plt.close(fig)


def make_cci_plot(df: pd.DataFrame) -> None:
    data = [df.loc[df["bmi3"].astype(str) == group, "cci_postop"].dropna() for group in BMI3_ORDER]
    p_value = stats.kruskal(*data).pvalue
    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(
        data,
        labels=[f"{group}\nn={len(values)}" for group, values in zip(BMI3_ORDER, data)],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": NORD["title"], "linewidth": 1.6},
        whiskerprops={"color": NORD["axis"], "linewidth": 1.1},
        capprops={"color": NORD["axis"], "linewidth": 1.1},
        boxprops={"color": NORD["axis"], "linewidth": 1.1},
    )
    for patch, color in zip(box["boxes"], [NORD["blue"], NORD["orange"], NORD["green"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.set_xlabel("BMI category (kg/m2)")
    ax.set_ylabel("CCI")
    ax.set_title("Comprehensive Complication Index by BMI category")
    ax.text(
        0.98,
        0.94,
        f"Kruskal-Wallis p={fmt_p(p_value)}" if p_value >= 0.001 else "Kruskal-Wallis p<0.001",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=NORD["title"],
        fontsize=10,
    )
    apply_nord_theme(ax)
    ax.grid(axis="y", color=NORD["grid"], alpha=0.45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cci_by_bmi3.png", dpi=300, facecolor=NORD["background"])
    fig.savefig(FIGURE_DIR / "cci_by_bmi3.svg", facecolor=NORD["background"])
    plt.close(fig)


def extract_reviewer_comments() -> list[str]:
    if not REVIEW_DOCX.exists():
        return []
    doc = Document(REVIEW_DOCX)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def write_markdown_outputs(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    validation = validation_report(df)
    (REVISION_ROOT / "validation_report.md").write_text(validation, encoding="utf-8")

    methods = """# HBSN Revision Methods Addendum

Additional analyses were performed to address the HBSN reviewers' methodological concerns. Best Performer was retained as a recovery-quality endpoint, but it was no longer treated as the sole marker of surgical morbidity because it includes length of stay. Endpoint robustness was therefore assessed using the Comprehensive Complication Index (CCI), clinically relevant postoperative pancreatic fistula (CR-POPF B/C), conversion, Clavien-Dindo >=III morbidity, Ideal Outcome, and a morbidity composite excluding length of stay.

BMI was reported in three clinically interpretable categories for the main descriptive revision tables: <25, 25-29.9, and >=30 kg/m2. Continuous variables were summarized as median [IQR] and compared using Kruskal-Wallis tests. Categorical variables were summarized as n/N (%) and compared using chi-square tests, with Fisher's exact test used for 2 x 2 sparse tables.

Centre volume was defined as mean annual MILP volume per centre, calculated as the total number of left pancreatectomies recorded for the centre (`NOMBRE_PG`) divided by the number of observed study years for that centre. The primary categories were Low <5/year, Mid 5-10/year, and High >10/year. Centre-level tier definitions were summarized using the number of centres, median [IQR] annual MILP volume, range, and annual MIPD volume when available. Patient case-mix and perioperative outcomes were compared across these tiers to document decentralization, selection structure, and the unadjusted association between volume and postoperative recovery.

For endpoint robustness, the adjusted main association between BMI and each endpoint was estimated in patient-level multivariable models including BMI per +5 kg/m2, centre volume per +5 annual MILP cases, age, ASA >=III, sex, malignant indication, robotic approach, and splenectomy. Effect modification by centre volume was tested in a separate model adding the BMI x centre-volume interaction term. Binary endpoints were analysed with logistic generalized linear models and centre-clustered robust standard errors. CCI was analysed as log(CCI+1) using Gaussian generalized linear models with centre-clustered robust standard errors; coefficients were back-transformed and reported as relative percent change.

Selection bias was assessed with two propensity-score approaches for High-volume versus Low/Mid-volume centres. The propensity model included age, sex, ASA >=III, malignant indication, BMI, robotic approach, splenectomy, pancreatic texture, duct diameter, and missingness indicators for texture and duct diameter. First, inverse probability of treatment weighting (IPTW) was performed using stabilized weights truncated at the 1st and 99th percentiles. Weighted outcome models estimated the association between high-volume care and Best Performer, morbidity excluding LOS, CR-POPF B/C, Clavien-Dindo >=III morbidity, and conversion, adjusted for BMI, age, ASA >=III, sex, malignant indication, robotic approach, and splenectomy. Second, 1:1 nearest-neighbour propensity score matching (PSM) without replacement was performed using a caliper of 0.2 standard deviations of the logit of the propensity score. Covariate balance was summarized with standardized mean differences before adjustment, after IPTW, and after PSM. Matched binary outcomes were compared using matched-pair odds ratios and McNemar exact tests based on discordant pairs.

Robotic and laparoscopic approaches were compared descriptively. A three-way BMI x volume x robotic interaction was tested for Best Performer, and approach-stratified BMI x volume models were fitted when event counts were sufficient. Complete-case anatomic sensitivity models included soft pancreatic texture and pancreatic duct diameter in addition to the main adjustment set. A stricter volume-threshold sensitivity analysis classified high-volume centres using >20 mean annual MIPD cases when available.
"""
    (METHODS_DIR / "methods_addendum.md").write_text(methods, encoding="utf-8")

    text = f"""# HBSN Revised Core Language

## Proposed title
Body Mass Index, Centre Volume, and Postoperative Recovery after Minimally Invasive Left Pancreatectomy: A Nationwide Multicentre Cohort Study

## Revised conclusion
Higher BMI was associated with lower probability of optimal recovery after MILP. This association was attenuated in higher-volume centres for the Best Performer recovery endpoint, but CCI, morbidity-only, robotic/laparoscopic, anatomic-risk, IPTW, and propensity-score matched sensitivity analyses indicate that the finding should be interpreted as an association reflecting technical expertise, perioperative organization, discharge practices, and selection rather than proof that centre expertise eliminates BMI-related morbidity.

## Revised discussion lead
The present revision reframes the study around postoperative recovery rather than a purely technical mitigation of obesity-related risk. The strongest signal remains the interaction between BMI and centre volume for Best Performer; however, because this endpoint includes length of stay, the result cannot be interpreted independently of institutional discharge policies and perioperative pathways. Sensitivity analyses using CCI and morbidity-only endpoints provide a more conservative clinical interpretation.

## Key limitations to add
BMI does not measure visceral adiposity or sarcopenic obesity. Pancreatic texture and duct diameter were incompletely available and were therefore incorporated as complete-case sensitivity analyses. Robotic and laparoscopic MILP differ mechanically and were analysed separately as requested. Finally, high-volume centres may select lower-risk high-BMI patients, so IPTW and propensity-score matched analyses were added to address measured allocation bias, while unmeasured selection remains possible.
"""
    (TEXT_DIR / "revised_core_language.md").write_text(text, encoding="utf-8")

    response = """# HBSN Point-by-Point Response Draft

Do not delete reviewer comments in the submitted response. Page and line numbers must be filled after manuscript editing.

## Reviewer A

**Comment:** Some words are understandable but not very commonly used in English, such as carcinologic or pejorative. Please check them.

**Reply:** We agree. The manuscript was edited throughout to use standard surgical terminology. In particular, "carcinologic" was replaced by "oncological", "pejorative" by "adverse", and causal wording such as "offsets" was replaced by associative wording.

**Changes in the text:** Page XX, lines XX-XX.

**Comment:** Methods section: one of my severe concerns is the use of the best performer measure. The ideal outcome has been defined previously, and there is an international consensus on it. The concept of best performer is nice, but it does not allow you to compare with other series. You used 25 and 40 BMI cut-offs when dividing the results. 30 is the most common cutoff for obesity. You should also reference the BMI measurement, including the citation of the literature. The cite on the number of pancreatectomies that should be done to be a high-volume center is nice, but the Miami Consensus is more recent and more widely used. Dindo was the first author, but is usually named as Clavien-Dindo.

**Reply:** We agree with these points. Best Performer was redefined in the revised manuscript as a recovery-quality endpoint rather than a standalone morbidity endpoint, and robustness analyses were added using CCI, CR-POPF B/C, conversion, Clavien-Dindo >=III morbidity, Ideal Outcome, and a morbidity composite excluding length of stay. BMI categories were revised to <25, 25-29.9, and >=30 kg/m2 in the main tables, while continuous BMI and extreme BMI categories remain in supplementary analyses. BMI measurement and classification were clarified. The volume-threshold wording was corrected so that our thresholds are described as prespecified cohort thresholds rather than as identical to prior consensus thresholds. "Clavien-Dindo" is used consistently.

**Changes in the text:** Page XX, lines XX-XX; Supplementary Tables R1-R18.

**Comment:** Results: conversion rate is higher than in other series. In the results, you use Clavien only; please always use the same name. The ideal outcome is similar to other published series, near 60%. The best performer concept should be explained better in the Methods, or erased.

**Reply:** The high conversion rate is now explicitly discussed in the context of a nationwide multicentre cohort with substantial low-volume participation. Perioperative outcomes were additionally stratified by centre-volume tier, showing lower conversion in high-volume centres. The terminology was standardized to Clavien-Dindo. Best Performer was retained only as a recovery-quality endpoint and was supported by standard morbidity endpoints and CCI analyses.

**Changes in the text:** Page XX, lines XX-XX; Table R5; Supplementary Methods.

## Reviewer B

**Comment:** In fig1, complication rate is lower when BMI is less than 18.5.

**Reply:** We agree that low-BMI strata require cautious interpretation. The revised manuscript now states that the underweight subgroup is sparse and should not be interpreted as an "ideal" BMI range. BMI extremes are reported in a dedicated supplementary table and described as hypothesis-generating observations.

**Changes in the text:** Page XX, lines XX-XX; Table R16.

**Comment:** Regarding center volume, low-volume (<5 MILP/year), intermediate-volume (5-10 MILP/year), and high volume (>10 MILP/year) were used with references.

**Reply:** The definition was clarified. Centre volume was calculated as mean annual MILP volume per centre and categorized as Low <5/year, Mid 5-10/year, and High >10/year for the primary analysis. We removed wording implying that these thresholds were identical to the cited consensus thresholds. A stricter sensitivity analysis using >20 mean annual MIPD cases was added.

**Changes in the text:** Page XX, lines XX-XX; Tables R3 and R17.

**Comment:** There are only 29 patients (1.4%) with BMI >=40 kg/m2, which is relatively less compared to other group. Is this group necessary to show your impact of BMI.

**Reply:** We agree. The revised manuscript no longer bases the main interpretation on BMI >=40 alone. The main descriptive BMI categories are <25, 25-29.9, and >=30 kg/m2, and continuous BMI modelling is used to avoid overinterpreting small extreme strata. BMI >=40 is retained only for transparency in supplementary/descriptive material.

**Changes in the text:** Page XX, lines XX-XX.

## Reviewer C

**Comment:** The primary outcome of "Best Performer" status is mathematically engineered by layering an arbitrary length of stay (LOS) constraint onto the "Ideal Outcome" composite marker. LOS is heavily influenced by institutional discharge policies and nonclinical logistics rather than physical recovery alone. Treating an administrative metric as a primary benchmark for surgical success creates an artificial quality hierarchy. A short LOS can mask delayed complications or early readmissions occurring just after the discharge window. This inflates the perceived quality of high-volume centers solely due to their faster organizational throughput. The authors must either provide objective clinical validation for this endpoint or reanalyze their primary data using standard clinical indexes, such as the Comprehensive Complication Index (CCI).

**Reply:** We agree that Best Performer is partly dependent on LOS and therefore cannot be interpreted as a pure surgical morbidity endpoint. The revised manuscript explicitly demotes Best Performer to a recovery-quality endpoint and adds endpoint-robustness analyses using CCI, CR-POPF B/C, conversion, Clavien-Dindo >=III morbidity, Ideal Outcome, and a morbidity-only composite excluding LOS. The revised conclusions are intentionally more conservative: the BMI-volume signal is strongest for recovery quality and should not be presented as proof that centre volume eliminates BMI-related morbidity.

**Changes in the text:** Page XX, lines XX-XX; Tables R8-R9; Figure R1.

**Comment:** Concluding that center expertise mitigates high-BMI risks overlooks a significant patient-selection bias among institutions. High-volume centers typically perform rigorous screening, selecting obese patients with favorable fat-to-muscle ratios or peripheral tumors. In contrast, low-volume centers treat unselected local populations. The authors' data (Supplementary Table S4) confirm this reality, showing that low-volume centers operated on a significantly higher proportion of patients with a BMI of 30 or more (P < 0.001). High-volume centers essentially operate on a trimmer, lower-risk cohort. Since the multivariable models fail to adjust for these baseline physical selectors and the institutional capacity to manage complications, the interaction model must be completely refitted using propensity score matching to control for this allocation bias.

**Reply:** We agree that selection bias is a major limitation. We first added a case-mix table by centre-volume tier to make the imbalance in BMI, obesity, ASA status, robotic access, splenectomy, pancreatic texture, and duct diameter transparent. We then performed propensity score matching as requested. The propensity model included age, sex, ASA >=III, malignant indication, BMI, robotic approach, splenectomy, pancreatic texture, duct diameter, and missingness indicators for texture and duct diameter. Patients treated in high-volume centres were matched 1:1 to patients treated in low/mid-volume centres using nearest-neighbour matching without replacement and a caliper of 0.2 standard deviations of the logit propensity score. Balance before and after matching is now reported using standardized mean differences, and matched outcomes were compared using matched-pair odds ratios and McNemar exact tests. Because matching necessarily excludes unmatched patients and may reduce precision, we also added IPTW as a complementary whole-cohort propensity-score sensitivity analysis. The revised discussion explicitly states that these approaches address measured allocation bias but cannot remove unmeasured selection.

**Changes in the text:** Page XX, lines XX-XX; Tables R4 and R12-R15.

**Comment:** The authors combine robotic and laparoscopic approaches into a single, generalized "MILP" entity, creating a massive confounding variable that violates basic surgical mechanics. The robotic platform provides instrumentation with wrist articulation and stereoscopic vision, which specifically neutralize the technical hurdles of a thick abdominal wall and dense visceral fat. By failing to disaggregate these two modalities or perform a strict subgroup interaction analysis, the manuscript obscures whether the "expertise attenuation" reflects superior surgical skill or simply a higher baseline availability of robotic systems in tertiary centers. The authors must separate the robotic and laparoscopic cohorts and demonstrate whether this volume benefit persists independently for both modalities.

**Reply:** We agree that approach is clinically important. The revised analysis includes a robotic-versus-laparoscopic descriptive table, a three-way BMI x volume x robotic interaction test, and approach-stratified BMI x volume models. These analyses are presented as exploratory because the robotic subgroup is smaller than the laparoscopic subgroup.

**Changes in the text:** Page XX, lines XX-XX; Tables R6-R7.

**Comment:** Using total BMI as a surrogate for surgical difficulty is flawed from a pathophysiological perspective because it does not correlate linearly with intra-abdominal visceral adiposity, which is the true driver of surgical visibility loss and tissue tension. Furthermore, predictive models often omit multivariable adjustments for pancreatic parenchymal texture (soft versus firm) and main pancreatic duct diameter. These factors are universally recognized as the primary drivers of clinically relevant pancreatic fistula (CR-POPF). A patient with a BMI of 35 and a firm pancreas has a much lower risk of developing a fistula than a thin patient with a soft pancreas. To provide the anatomic resolution required to support centralization guidelines, the regression models must be adjusted for these primary parenchymal factors.

**Reply:** We agree that BMI is an imperfect surrogate for operative difficulty and does not capture visceral adiposity or sarcopenic obesity. We added complete-case sensitivity models incorporating soft pancreatic texture and pancreatic duct diameter. Missingness is explicitly reported because texture and duct diameter were not available for all patients. The revised discussion now treats residual anatomic confounding as a central limitation.

**Changes in the text:** Page XX, lines XX-XX; Tables R10 and R18.

## Editorial comments

**Comment:** Please add the STROBE reporting checklist sentence in the Introduction and footnote.

**Reply:** The requested STROBE statement should be added at the end of the Introduction and in the footnote, with the final HBSN checklist URL inserted after resubmission details are available.

**Changes in the text:** Page XX, lines XX-XX.

**Comment:** Please specify the APC-responsible author and email, provide a Data Sharing Statement, COI forms, originality confirmation, updated references, and ensure compliance with author instructions.

**Reply:** These administrative items should be completed during final submission assembly. The scientific revision package includes updated tables, methods, conservative language, and response language; the corresponding author should add the APC-responsible author/email, data-sharing statement, COI forms, originality statement, updated references, structured abstract, highlight box, abbreviation list, and final line numbers.

**Changes in the text:** Page XX, lines XX-XX.
"""
    (RESPONSE_DIR / "point_by_point_response_draft.md").write_text(response, encoding="utf-8")

    readme = """# HBSN Revision Package

Run all analyses:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplcache python hbsn_revision/run_all.py
```

Main outputs:

- `tables/`: CSV tables for HBSN revision, including IPTW and propensity-score matched analyses.
- `figures/`: endpoint interaction forest plot and CCI plot.
- `HBSN_revision_tables.docx`: Word-ready supplementary/main revision tables R1-R18.
- `methods/methods_addendum.md`: Methods text for revision.
- `manuscript_text/revised_core_language.md`: revised title/conclusion/discussion language.
- `response_to_reviewers/point_by_point_response_draft.md`: response draft with HBSN priorities.
"""
    (REVISION_ROOT / "README.md").write_text(readme, encoding="utf-8")


def validation_report(df: pd.DataFrame) -> str:
    checks = [
        ("Total cohort n", int(df.shape[0]), 2028),
        ("Centres n", int(df["CENTRE"].nunique()), 55),
        ("Robotic n", int(df["robotic"].sum()), 270),
        ("Laparoscopic n", int((~df["robotic"]).sum()), 1758),
        ("Mortality n", int(df["mortality"].sum()), 28),
        ("CR-POPF B/C n", int(df["popf_bc"].sum()), 387),
        ("Texture available n", int(df["texture_available"].sum()), 1579),
        ("Duct diameter available n", int(df["duct_available"].sum()), 1259),
    ]
    lines = ["# Validation report", ""]
    for label, observed, expected in checks:
        lines.append(f"- {'PASS' if observed == expected else 'CHECK'}: {label}: observed {observed}, expected {expected}")
    return "\n".join(lines) + "\n"


def build_docx(tables: dict[str, pd.DataFrame]) -> None:
    doc = Document()
    doc.core_properties.author = AUTHOR
    doc.core_properties.last_modified_by = AUTHOR
    doc.core_properties.comments = ""
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11.69)
    section.page_height = Inches(8.27)
    section.left_margin = Inches(0.45)
    section.right_margin = Inches(0.45)
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(8)
    doc.add_heading("HBSN Revision Tables", level=1)

    default_note = "Values are n/N (%) unless otherwise stated. P values are exploratory unless specified in the Methods."
    captions = [
        ("Table R1. Baseline characteristics by three BMI categories.", "baseline_bmi3", default_note),
        ("Table R2. Postoperative outcomes by three BMI categories.", "postoperative_outcomes_bmi3", default_note),
        ("Table R3. Definition of centre-volume tiers.", "centre_volume_definition", "Centre-volume tiers are based on mean annual MILP volume per centre."),
        ("Table R4. Patient case-mix according to centre-volume tier.", "case_mix_by_volume", default_note),
        ("Table R5. Outcomes according to centre-volume tier.", "outcomes_by_volume", default_note),
        ("Table R6. Robotic versus laparoscopic baseline and outcomes.", "approach_table", default_note),
        ("Table R7. Robotic/laparoscopic interaction and stratified models.", "approach_interactions", "P values are from multivariable interaction models described in the Methods."),
        ("Table R8. Endpoint robustness models.", "endpoint_robustness_models", "Binary endpoints are reported as adjusted odds ratios. CCI is reported as relative percent change after log(CCI+1) modelling."),
        ("Table R9. Summary of BMI x centre-volume interaction across sensitivity analyses.", "sensitivity_interaction_summary", "Estimates are ORs except for CCI, which is shown as a relative ratio from the log(CCI+1) model."),
        ("Table R10. Complete-case anatomic sensitivity models.", "anatomic_complete_case_sensitivity", "Models include pancreatic texture and duct diameter in complete cases."),
        ("Table R11. CCI according to BMI category and centre volume.", "cci_by_groups", "CCI values are median [IQR]; p values are from Kruskal-Wallis tests."),
        ("Table R12. Propensity score balance before adjustment, after IPTW, and after PSM.", "propensity_balance", "Absolute standardized mean differences below 0.10 are usually considered acceptable balance."),
        ("Table R13. Propensity-weighted outcome estimates.", "propensity_weighted_outcomes", "IPTW estimates are sensitivity analyses based on stabilized, truncated weights."),
        ("Table R14. Propensity-score matched outcome estimates.", "propensity_matched_outcomes", "Matched outcomes use matched-pair odds ratios and McNemar exact tests."),
        ("Table R15. Propensity-score matching diagnostic.", "propensity_matching_diagnostic", "PSM used 1:1 nearest-neighbour matching without replacement."),
        ("Table R16. BMI extremes and sparse high-BMI strata.", "bmi_extremes", "Extreme BMI categories are descriptive and should not be overinterpreted."),
        ("Table R17. Centre-volume threshold sensitivity.", "volume_threshold_sensitivity", "P values compare high-volume versus other centres for each threshold definition."),
        ("Table R18. Availability of anatomic-risk and sensitivity variables.", "missingness", "Availability is reported over the total analytic cohort."),
    ]
    for caption, key, note in captions:
        df = tables[key]
        p = doc.add_paragraph()
        run = p.add_run(caption)
        run.bold = True
        table = doc.add_table(rows=df.shape[0] + 1, cols=df.shape[1])
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Table Grid"
        for j, col in enumerate(df.columns):
            table.cell(0, j).text = str(col)
            for r in table.cell(0, j).paragraphs[0].runs:
                r.bold = True
                r.font.size = Pt(7)
        for i in range(df.shape[0]):
            for j, col in enumerate(df.columns):
                val = "" if pd.isna(df.iloc[i, j]) else str(df.iloc[i, j])
                table.cell(i + 1, j).text = val
                for r in table.cell(i + 1, j).paragraphs[0].runs:
                    r.font.size = Pt(7)
                    if "p-value" in str(col).lower():
                        try:
                            r.bold = val.startswith("<") or float(val) < 0.05
                        except Exception:
                            pass
        doc.add_paragraph(note)
        doc.add_page_break()
    doc.save(REVISION_ROOT / "HBSN_revision_tables.docx")


def main() -> None:
    for path in [TABLE_DIR, FIGURE_DIR, METHODS_DIR, TEXT_DIR, RESPONSE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    df = build_dataset()

    tables = {
        "baseline_bmi3": grouped_table(df, "bmi3", BMI3_ORDER, include_baseline=True, include_outcomes=False),
        "postoperative_outcomes_bmi3": grouped_table(df, "bmi3", BMI3_ORDER, include_baseline=False),
        "centre_volume_definition": centre_volume_definition(df),
        "case_mix_by_volume": case_mix_by_volume(df),
        "outcomes_by_volume": grouped_table(df, "centre_volume_tier", VOLUME_ORDER, include_baseline=False),
        "approach_table": None,
        "approach_interactions": None,
        "endpoint_robustness_models": endpoint_robustness(df),
        "anatomic_complete_case_sensitivity": anatomic_complete_case(df),
        "cci_by_groups": cci_group_tables(df),
        "volume_threshold_sensitivity": volume_threshold_sensitivity(df),
        "bmi_extremes": bmi_extremes(df),
        "missingness": missingness_table(df),
    }
    approach_table, approach_interactions = approach_analysis(df)
    tables["approach_table"] = approach_table
    tables["approach_interactions"] = approach_interactions
    ps_balance, ps_outcomes, ps_matched, ps_match = propensity_analysis(df)
    tables["propensity_balance"] = ps_balance
    tables["propensity_weighted_outcomes"] = ps_outcomes
    tables["propensity_matched_outcomes"] = ps_matched
    tables["propensity_matching_diagnostic"] = ps_match
    tables["sensitivity_interaction_summary"] = sensitivity_interaction_summary(tables)

    for name, table in tables.items():
        write_csv(table, f"{name}.csv")

    make_forest_plot(tables["endpoint_robustness_models"])
    make_sensitivity_interaction_plot(tables["sensitivity_interaction_summary"])
    make_cci_plot(df)
    write_markdown_outputs(df, tables)
    build_docx(tables)
    print(f"Wrote HBSN revision outputs to {REVISION_ROOT}")
    print(validation_report(df))


if __name__ == "__main__":
    main()
