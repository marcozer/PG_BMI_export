# Pipeline for the HBSN 2026 article

This folder reproduces the figures and tables of:

> Nassar A, Chouillard MA, Pastier C, Dokmak S, Fuks D, De Ponthaud C, Gaujoux S; AFC-2023 study group. *Influence of body mass index (BMI) on minimally invasive left pancreatectomy: can center expertise mitigate BMI-related challenges?* Hepatobiliary Surg Nutr 2026. doi:10.21037/hbsn-2026-0394

It supersedes the earlier `01_bmi_risk/`, `02_interaction/` and `03_visuals/` analyses at the repository root, which use a different centre-volume definition and are kept for history.

## Data

The patient-level dataset (AFC-2023 registry) is **not** distributed: it is confidential and available on reasonable request. The scripts expect it at:

```
data/raw/pg_afc_sheet1.csv      # relative to the repository root (git-ignored)
```

## Definitions (as in the article)

- **Centre volume:** a centre's mean annual number of minimally invasive left pancreatectomies (total MILP recorded for the centre ÷ number of observed study years).
  - Tiers: Low < 5, Mid 5–10, High > 10 per year.
  - Sensitivity analysis: High > 20 minimally invasive pancreatic resections per year.
- **Ideal Outcome:** no death, no Clavien-Dindo ≥ III, no POPF grade B/C, no reoperation, no readmission, and LOS ≤ 75th percentile.
- **Best Performer:** Ideal Outcome with LOS ≤ 25th percentile.

## How to run

From the `hbsn_revision/` folder:

1. Supplementary analyses and tables (endpoint robustness, propensity-score matching, approach, anatomic, threshold sensitivity) are written to `tables/` and `figures/`:

```bash
python run_all.py
```

2. Figures 1, 2A, 2B and 2C, plus the supplementary figures, are written to `figures_to_incorporate/{main,supplementary}/`, together with the CSV data behind the curves:

```bash
python figures_to_incorporate/build_hbsn_ready_figures.py
```

3. Main Tables 1 and 2 (BMI < 25 / 25–29.9 / ≥ 30) are written to `minimal_R1_from_R0/tables_csv/`:

```bash
python minimal_R1_from_R0/scripts/build_r0_format_main_tables_bmi3.py
```

Requirements: Python ≥ 3.10, pandas, numpy, scipy, statsmodels, patsy, matplotlib, python-docx.

## Published files

- `minimal_R1_from_R0/figures/`: the exact figure files submitted with the article. Figures 2A and 2C carry a "Number of patients" label that was added to the rug panel by manual editing after script generation.
- `minimal_R1_from_R0/tables_csv/`: the exact CSV tables submitted. The supplementary tables are a subset of `tables/`.

## Reproducibility check

The data behind the figures (`figures_to_incorporate/supplementary/*_hbsn.csv`), all `tables/*.csv` and the main Tables 1–2 were regenerated from a clean clone with the original dataset. They were identical to the committed files.
