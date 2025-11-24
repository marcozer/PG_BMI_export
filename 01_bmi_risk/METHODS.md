## Méthodes – Effet intrinsèque du BMI (IO/BP)

### Modèle principal (inférentiel)
- GLM logistique binomial avec spline cubique (`bs(BMI, df=4)`).
- Ajustements : âge, ASA≥III, sexe, malignité, abord robotique, splénectomie, volume annuel de centre (continu).
- Courbe dose-réponse : prédictions du GLM le long d’une grille 18–45 kg/m² avec IC 95% du modèle (SE robustes). Fichiers : `bmi_bp_curve.csv`, `bmi_bp_dose_response.svg/png`.
- Comparaisons vs BMI 25 : OR et IC 95% à 30/35/40 (`bmi_or_vs25.csv`). Diagnostics : AUC/calibration (`bmi_bp_calibration.csv`, `bmi_bp_diagnostics.csv`), VIF (`bmi_bp_vif.csv`).

### Courbes descriptives (non inférentiel)
- `bmi_bp_volume_pooled` : High vs Low+Mid.
  - Ligne : LOWESS sur les données brutes, échantillonage 2 kg/m² visuel.
  - Bande : bootstrap (300 rééchantillonnages) autour de la courbe LOWESS, pointwise 95% (descriptif).
  - Rug : un trait par patient sous l’axe X.
  - Tests séparés : High vs Low+Mid sur bins de 5 kg/m² (seuils n_high≥10, n_pool≥20), z-test de proportions, correction Holm. Sortie : `volume_pooled_bin_tests.csv`. Les astérisques indiquent les bins significatifs après Holm.
  - Test global d’interaction spline×volume (cluster centre) : `volume_pooled_global_test.csv`.
- `bmi_bp_volume_tiers` : lignes lissées (bins 1 kg/m², moyenne glissante) pour Low/Mid/High + rug par tier ; étoiles = bins où High diffère de Low+Mid (z-test naïf descriptif).

### Données
- Fichier source : `data/raw/pg_afc_sheet1.csv` (ou `PG_AFC.xlsx` feuille Sheet1 en secours) via `lib/dataset.py`.

## Methods – Intrinsic BMI effect (IO/BP)

### Primary model (inference)
- Binomial logistic GLM with cubic spline BMI (`bs(BMI, df=4)`).
- Adjusted for age, ASA≥III, sex, malignant status, robotic approach, splenectomy, centre volume (continuous).
- Dose-response curve: model predictions on BMI 18–45 with 95% model-based CIs (cluster-robust SE). Files: `bmi_bp_curve.csv`, `bmi_bp_dose_response.svg/png`.
- ORs vs BMI 25 at 30/35/40 (`bmi_or_vs25.csv`). Diagnostics: AUC/calibration (`bmi_bp_calibration.csv`, `bmi_bp_diagnostics.csv`), VIF (`bmi_bp_vif.csv`).

### Descriptive curves (non-inferential)
- `bmi_bp_volume_pooled`: High vs Low+Mid.
  - Line: LOWESS on raw points.
  - Band: bootstrap (300 resamples) pointwise 95% around the LOWESS curve (descriptive only).
  - Rug: one tick per patient below the axis.
  - Separate tests: High vs Low+Mid on 5-kg bins (n_high≥10, n_pool≥20), two-proportion z-test with Holm correction. Output: `volume_pooled_bin_tests.csv`. Stars mark bins significant after Holm.
  - Global spline×volume interaction test (clustered LRT): `volume_pooled_global_test.csv`.
- `bmi_bp_volume_tiers`: smoothed lines (1-kg bins, rolling mean) for Low/Mid/High + rug per tier; stars = bins where High vs Low+Mid differs (naïve descriptive z-test).

### Data
- Source file: `data/raw/pg_afc_sheet1.csv` (fallback to `PG_AFC.xlsx` Sheet1) via `lib/dataset.py`.
