Export package for BMI × volume analyses

Overview
- This folder contains the cleaned, GitHub‑ready export of the project’s three analyses:
  - 01_bmi_risk: BMI intrinsic effect (dose–response) and diagnostics
  - 02_interaction: BMI × centre volume interaction (primary result)
  - 03_visuals: component breakdown (descriptive)

What’s included
- METHODS.md for each analysis describing the models and tests (FR/EN)
- Scripts: minimal `run.py` and `plot.py` (kept for reference)
- Outputs (CSV/PNG/SVG) ready for figures and tables

Notes
- Scripts expect the original dataset (e.g. `data/raw/pg_afc_sheet1.csv`). The data are not exported here.
- The primary inferential analysis is 02_interaction (linear BMI × volume with cluster‑robust SE). Curves are marginal means (g‑computation) with delta‑method 95% CIs.
- Alternative visualisations are provided: combined panels, facets, point+whiskers at BMI 25/30/35/40, and OR(+5) bars per tier.

Git hygiene
- `.DS_Store` files were removed; a `.gitignore` is provided to keep them out.

