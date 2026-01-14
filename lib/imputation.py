from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .dataset import build_dataset

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
except Exception:  # pragma: no cover
    IterativeImputer = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = REPO_ROOT / "data/derived"


def mice_impute(df: pd.DataFrame, cols: Iterable[str], n_iter: int = 15, random_state: int = 42) -> pd.DataFrame:
    """Lightweight MICE‑style imputation using sklearn's IterativeImputer.

    - Works on the specified columns; leaves others unchanged.
    - Binary columns are rounded back to {0,1} after imputation.
    - If IterativeImputer is unavailable, falls back to median fill.
    """
    df_out = df.copy()
    sub = df_out[list(cols)].copy()
    # Identify binary columns (0/1/True/False)
    bin_cols = [c for c in sub.columns if sub[c].dropna().astype(str).str.match(r"^(0|1|True|False)$").all()]
    sub_float = sub.astype(float)
    if IterativeImputer is None:
        sub_imputed = sub_float.fillna(sub_float.median())
    else:
        imp = IterativeImputer(max_iter=n_iter, random_state=random_state, sample_posterior=True)
        sub_imputed = pd.DataFrame(imp.fit_transform(sub_float), columns=sub_float.columns, index=sub_float.index)
    for c in bin_cols:
        sub_imputed[c] = (sub_imputed[c] >= 0.5).astype(int)
    df_out[sub_imputed.columns] = sub_imputed
    return df_out


def build_dataset_imputed(n_iter: int = 15) -> pd.DataFrame:
    df = build_dataset().copy()
    cols = [
        "bmi", "centre_volume", "age", "asa_ge3", "sex_male", "malignant", "robotic", "splenectomy",
    ]
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    df_imp = mice_impute(df, cols, n_iter=n_iter)
    out = DERIVED_DIR / "pg_afc_sheet1_imputed.csv"
    df_imp.to_csv(out, index=False)
    return df_imp


__all__ = ["build_dataset_imputed", "mice_impute"]
