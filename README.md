Export package for BMI and center volume analyses for predicting Best Performers outcomes after Laparoscopic Distal Pancreatectomy. 

The model were built by Alexandra Nassar, Clément Pastier, Sébastien Gaujoux and Marc-Anthony Chouillard for the [AFC](https://www.association-francaise-chirurgie.fr/) (French Association of Visceral Surgery). 

## Overview
- This folder contains the cleaned, GitHub‑ready export of the project’s three analyses:
  - 01_bmi_risk: BMI intrinsic effect ("dose–response" like analysis.) and diagnostics
  - 02_interaction: BMI × centre volume interaction (primary result : stratification by BMI and Center, and statistical analysis.)
  - 03_visuals: component breakdown (descriptive on the contribution of each complication stratified by BMI)
  - lib folder centralises ressources used by all the scripts 

## What’s included
- METHODS.md for each analysis describing the models and tests (FR only, English is in progress)
- Scripts: minimal `run.py` and `plot.py` (kept for reference)
- Outputs (CSV/PNG/SVG for quality preservation) ready for figures and tables
- Specifically edited figures for manuscript  

## Notes
- Scripts expect the original dataset (e.g. `data/raw/pg_afc_sheet1.csv`). The data are not exported here.
- The primary inferential analysis is 02_interaction (linear BMI × volume with cluster‑robust SE). Curves are marginal means (g‑computation) with delta‑method 95% CIs.
- Alternative visualisations are provided: combined panels, facets, point+whiskers at BMI 25/30/35/40, and OR(+5) bars per tier.

## CLI flags (volume definition)
- All `run.py` scripts support `--volume-tier-mode`:
  - `tertiles` (default): Low/Mid/High defined by tertiles of total cases per centre in the dataset.
  - `annual_threshold`: Low <5, Mid 5–10, High >10 pancreatectomies per year (per centre-year). This recomputes `centre_volume` and `centre_volume_cat` internally and propagates to all analyses and plots.
  - the latter is the prefered definition for volume. The tertile mode would be used for sensitivity analysis. 
- Examples:
  - `python export/02_interaction/run.py --volume-tier-mode annual_threshold`
  - `python export/01_bmi_risk/run.py --volume-tier-mode tertiles`

