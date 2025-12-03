"""Shared dataset builders for BMI-outcome analyses (local copy for export).

Copied from original analysis/lib/dataset.py to ensure export is runnable
without the analysis/ tree present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .data_loader import load_excel_sheet

DEFAULT_CSV = Path("data/raw/pg_afc_sheet1.csv")
LOS_75TH_THRESHOLD = 13  # per manuscript
LOS_25TH_THRESHOLD = 7


_DEF_BOOL_TRUE = {"OUI", "YES", "1", "TRUE"}


def _bool(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.upper()
    return values.isin(_DEF_BOOL_TRUE)


def _empty_like(df: pd.DataFrame) -> pd.Series:
    return pd.Series(["" for _ in range(len(df))])


def _coerce_numeric(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    for src, dest in mapping.items():
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce")
        else:
            df[dest] = np.nan
    return df


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    path_obj = Path(path or DEFAULT_CSV)
    if path_obj.suffix.lower() == ".csv" and path_obj.exists():
        return pd.read_csv(path_obj)
    if path_obj.suffix.lower() == ".xlsx" and path_obj.exists():
        return load_excel_sheet(str(path_obj), sheet_index=1)
    legacy_csv = Path("data/pg_afc_sheet1.csv")
    if DEFAULT_CSV.exists():
        return pd.read_csv(DEFAULT_CSV)
    if legacy_csv.exists():
        return pd.read_csv(legacy_csv)
    raise FileNotFoundError(f"Unable to locate dataset at {path_obj}")


def build_dataset(path: str | Path | None = None) -> pd.DataFrame:
    df = load_raw(path).copy()
    df.rename(columns={"Date de sortie* ": "DATE_SORTIE"}, inplace=True)

    numeric_casts = {
        "IMC": "bmi",
        "LHS": "los_days",
        "AGE": "age",
        "POIDS": "weight",
        "TAILLE_SUJET": "height_cm",
        "DUREE_OP": "or_time",
        "SANG": "blood_loss",
        "CGR": "cgr_units",
        "ANNEE": "year",
    }
    df = _coerce_numeric(df, numeric_casts)

    df["centre_volume"] = df.groupby("CENTRE")["CODE"].transform("count")
    df["centre_volume_cat"] = pd.qcut(
        df["centre_volume"],
        q=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Mid", "High"],
        duplicates="drop",
    )

    df["major_clavien"] = pd.to_numeric(df.get("CLAVIEN"), errors="coerce").ge(3)
    df["popf_bc"] = df.get("POPF", _empty_like(df)).fillna("").astype(str).str.upper().isin(["B", "C"])
    df["reoperation"] = _bool(df.get("REOPERATION", _empty_like(df)))
    df["mortality"] = _bool(df.get("DECES", _empty_like(df)))
    df["readmission"] = _bool(df.get("REHOSPITALISATION", _empty_like(df)))
    df["conversion"] = _bool(df.get("CONVERSION", _empty_like(df)))
    df["splenectomy"] = df.get("SPLENECTOMIE", _empty_like(df)).str.upper().isin(["OUI", "URG"])
    df["asa_ge3"] = pd.to_numeric(df.get("ASA"), errors="coerce").ge(3)
    df["sex_male"] = df.get("SEXE", _empty_like(df)).str.upper().eq("M")
    # MALIN column may be missing; safer mapping: from anapath summary or MALIN/ADK/CM
    df["malignant"] = _bool(df.get("MALIN", _empty_like(df))) if "MALIN" in df.columns else _bool(df.get("ADK", _empty_like(df)))
    df["robotic"] = df.get("ABORD", _empty_like(df)).str.upper().eq("ROBOT")

    df["ideal_outcome"] = (
        ~df["mortality"]
        & ~df["major_clavien"]
        & ~df["popf_bc"]
        & ~df["reoperation"]
        & ~df["readmission"]
        & (df["los_days"] <= LOS_75TH_THRESHOLD)
    )
    df["best_performer"] = df["ideal_outcome"] & (df["los_days"] <= LOS_25TH_THRESHOLD)

    bmi_bins = [0, 18.5, 25, 30, 35, 40, np.inf]
    bmi_labels = ["<18.5", "18.5-24.9", "25-29.9", "30-34.9", "35-39.9", "≥40"]
    df["bmi_class"] = pd.cut(df["bmi"], bins=bmi_bins, labels=bmi_labels)

    return df


__all__ = ["build_dataset", "load_raw", "LOS_25TH_THRESHOLD", "LOS_75TH_THRESHOLD"]

