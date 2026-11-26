"""Local dataset builder for HBSN-2026-0394 revision analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "raw" / "pg_afc_sheet1.csv"
LOS_75TH_THRESHOLD = 13
LOS_25TH_THRESHOLD = 7
VOLUME_ORDER = ["Low", "Mid", "High"]
BMI3_ORDER = ["<25", "25-29.9", ">=30"]
BMI6_ORDER = ["<18.5", "18.5-24.9", "25-29.9", "30-34.9", "35-39.9", ">=40"]
BMI_EXT_ORDER = ["<18.5", "18.5-19.9", "20-24.9", "25-29.9", "30-34.9", "35-39.9", ">=40"]
APPROACH_ORDER = ["Laparoscopic", "Robotic"]


def _empty_like(df: pd.DataFrame) -> pd.Series:
    return pd.Series(["" for _ in range(len(df))], index=df.index)


def _clean_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def _bool(series: pd.Series) -> pd.Series:
    return _clean_str(series).isin({"OUI", "YES", "1", "TRUE"})


def _numeric(df: pd.DataFrame, src: str) -> pd.Series:
    if src not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[src], errors="coerce")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def _add_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = _numeric(out, "ANNEE")
    years = out.groupby("CENTRE", observed=False)["year"].nunique().reset_index(name="n_years")
    out["NOMBRE_PG"] = _numeric(out, "NOMBRE_PG")
    pg_total = (
        out.groupby("CENTRE", observed=False)["NOMBRE_PG"]
        .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
        .reset_index(name="pg_total")
    )
    centre_volume = pg_total.merge(years, on="CENTRE", how="left")
    centre_volume["centre_volume_annual"] = centre_volume["pg_total"] / centre_volume["n_years"]
    out = out.merge(centre_volume, on="CENTRE", how="left")
    out["centre_volume_tier"] = np.select(
        [
            out["centre_volume_annual"].lt(5),
            out["centre_volume_annual"].ge(5) & out["centre_volume_annual"].le(10),
            out["centre_volume_annual"].gt(10),
        ],
        VOLUME_ORDER,
        default=np.nan,
    )
    out["centre_volume_tier"] = pd.Categorical(out["centre_volume_tier"], categories=VOLUME_ORDER, ordered=True)

    out["NOMBRE_MIPD"] = _numeric(out, "NOMBRE_MIPD")
    mipd_total = (
        out.groupby("CENTRE", observed=False)["NOMBRE_MIPD"]
        .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
        .reset_index(name="mipd_total")
    )
    mipd = mipd_total.merge(years, on="CENTRE", how="left")
    mipd["mipd_mean_annual"] = mipd["mipd_total"] / mipd["n_years"]
    out = out.merge(mipd[["CENTRE", "mipd_mean_annual"]], on="CENTRE", how="left")
    out["strict_mipd_high"] = out["mipd_mean_annual"].gt(20)
    return out


def build_dataset(path: str | Path | None = None) -> pd.DataFrame:
    df = load_raw(path).copy()
    df = _add_volume_metrics(df)

    df["bmi"] = _numeric(df, "IMC")
    df["age"] = _numeric(df, "AGE")
    df["los_days"] = _numeric(df, "LHS")
    df["or_time"] = _numeric(df, "DUREE_OP")
    df["blood_loss"] = _numeric(df, "SANG")
    df["cci_postop"] = _numeric(df, "CCI_COMPLI")
    df["duct_diameter"] = _numeric(df, "CPP_DILATE")

    sex = _clean_str(df.get("SEXE", _empty_like(df)))
    df["female"] = sex.eq("F")
    df["sex_male"] = sex.eq("M")
    df["asa_ge3"] = _numeric(df, "ASA").ge(3)
    df["malignant"] = _bool(df.get("MALIN", _empty_like(df)))
    abord = _clean_str(df.get("ABORD", _empty_like(df)))
    df["robotic"] = abord.eq("ROBOT")
    df["approach"] = np.where(df["robotic"], "Robotic", "Laparoscopic")
    df["approach"] = pd.Categorical(df["approach"], categories=APPROACH_ORDER, ordered=True)
    df["splenectomy"] = _clean_str(df.get("SPLENECTOMIE", _empty_like(df))).isin({"OUI", "URG"})

    texture = _clean_str(df.get("CONSISTANCE", _empty_like(df)))
    df["soft_pancreas"] = texture.eq("MOU")
    df["firm_pancreas"] = texture.eq("DUR")
    df["texture_available"] = texture.isin({"MOU", "DUR"})
    df["duct_available"] = df["duct_diameter"].notna()

    popf_raw = _clean_str(df.get("POPF", _empty_like(df)))
    df["popf_raw_clean"] = popf_raw
    df["popf_any"] = ~popf_raw.isin({"", "NON", "NAN"})
    df["popf_bc"] = popf_raw.isin({"B", "C"})

    df["major_clavien"] = _numeric(df, "CLAVIEN").ge(3)
    df["mortality"] = _bool(df.get("DECES", _empty_like(df)))
    df["readmission"] = _bool(df.get("REHOSPITALISATION", _empty_like(df)))
    df["reoperation"] = _bool(df.get("REOPERATION", _empty_like(df)))
    df["conversion"] = _bool(df.get("CONVERSION", _empty_like(df)))

    df["ideal_outcome"] = (
        ~df["mortality"]
        & ~df["major_clavien"]
        & ~df["popf_bc"]
        & ~df["reoperation"]
        & ~df["readmission"]
        & df["los_days"].le(LOS_75TH_THRESHOLD)
    )
    df["best_performer"] = df["ideal_outcome"] & df["los_days"].le(LOS_25TH_THRESHOLD)
    df["los_prolonged"] = df["los_days"].gt(LOS_75TH_THRESHOLD)
    df["major_morbidity_no_los"] = (
        df["mortality"]
        | df["major_clavien"]
        | df["popf_bc"]
        | df["reoperation"]
        | df["readmission"]
    )

    df["bmi3"] = pd.cut(df["bmi"], bins=[-np.inf, 25, 30, np.inf], labels=BMI3_ORDER, right=False, ordered=True)
    df["bmi6"] = pd.cut(df["bmi"], bins=[-np.inf, 18.5, 25, 30, 35, 40, np.inf], labels=BMI6_ORDER, right=False, ordered=True)
    df["bmi_ext"] = pd.cut(df["bmi"], bins=[-np.inf, 18.5, 20, 25, 30, 35, 40, np.inf], labels=BMI_EXT_ORDER, right=False, ordered=True)
    df["high_volume"] = df["centre_volume_tier"].astype(str).eq("High")
    return df


__all__ = [
    "APPROACH_ORDER",
    "BMI3_ORDER",
    "BMI6_ORDER",
    "BMI_EXT_ORDER",
    "DATA_PATH",
    "LOS_25TH_THRESHOLD",
    "LOS_75TH_THRESHOLD",
    "REPO_ROOT",
    "VOLUME_ORDER",
    "build_dataset",
]
