Export package for BMI and center volume analyses for predicting Best Performers outcomes after Laparoscopic Distal Pancreatectomy. 

The model were built by Alexandra Nassar, Clément Pastier, Sébastien Gaujoux and Marc-Anthony Chouillard for the [AFC](https://www.association-francaise-chirurgie.fr/) (French Association of Visceral Surgery). 

## Overview
- This folder contains the cleaned, GitHub‑ready export of the project’s three analyses:
  - 01_bmi_risk: BMI intrinsic effect ("dose–response" like analysis.) and diagnostics
  - 02_interaction: BMI × centre volume interaction (primary result : stratification by BMI and Center, and statistical analysis.)
  - 03_visuals: component breakdown (descriptive on the contribution of each complication stratified by BMI)

## What’s included
- METHODS.md for each analysis describing the models and tests (FR only, English is in progress)
- Scripts: minimal `run.py` and `plot.py` (kept for reference)
- Outputs (CSV/PNG/SVG for quality preservation) ready for figures and tables

## Notes
- Scripts expect the original dataset (e.g. `data/raw/pg_afc_sheet1.csv`). The data are not exported here.
- The primary inferential analysis is 02_interaction (linear BMI × volume with cluster‑robust SE). Curves are marginal means (g‑computation) with delta‑method 95% CIs.
- Alternative visualisations are provided: combined panels, facets, point+whiskers at BMI 25/30/35/40, and OR(+5) bars per tier.


