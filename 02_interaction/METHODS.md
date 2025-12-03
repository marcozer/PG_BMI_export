## Méthodes – Interaction BMI × expertise du centre

- Modèles : GLM logistique binomial avec BMI continu + interaction BMI×(volume/10), ajusté sur âge, ASA≥III, sexe, malignité, robotique, splénectomie. GEE (structure exchangeable) pour SE robustes de cluster centre.
- Définition du volume (option `--volume-tier-mode`) :
  - `tertiles` : volume total par centre (base entière)
  - `annual_threshold` : volume = MOYENNE annuelle par centre; tiers : <5 / 5–10 / >10 cas/an
- Inférence principale : p-valeur de l’interaction (GEE) dans `interaction_p.csv`; coefficients GLM/GEE dans `glm_parameters.csv` / `gee_parameters.csv`.
- Courbe par tiers de volume : prédictions marginalisées (g‑computation) le long de 18–45 kg/m², en fixant le volume à la médiane du tier (Low/Mid/High). IC 95% par delta‑method avec covariance cluster‑robuste. Fichiers : `bmi_volume_curves.csv`, `bmi_interaction_tiers.svg/png`.
- Standardisation marginale: pour chaque valeur de BMI et chaque tier, les probabilités sont prédites pour tous les patients (avec BMI fixé et volume au tier), puis moyennées pour refléter le case‑mix réel; les IC sont obtenus par delta‑method (gradient moyen p(1−p)·x et matrice de variance cluster‑robuste du modèle), plus stable et cohérent que des bandes bootstrap.
- Comparaisons ponctuelles pré-spécifiées : High vs Low à BMI 30/35/40 avec delta method sur la différence de probabilité, IC 95%, p-valeur (Holm si >1 test). Fichier : `pairwise_bmi_comparisons.csv`; barrettes associées dans `pairwise_bmi_bars.svg/png`.
- Sensibilités (supplément) : OR par +5 BMI et ARD 40 vs 25 par tier (`or_per5_by_volume_tier.csv`, `ard_bmi25_40_by_tier.csv`), seuils de volume alternatifs (`volume_threshold_tests.csv`), interactions triples (malignité, robotique) (`threeway_interactions.csv`).

## Methods – BMI × centre expertise interaction

- Models: binomial logistic GLM with continuous BMI and BMI×(volume/10) interaction, adjusted for age, ASA≥III, sex, malignant status, robotic approach, splenectomy; plus GEE (exchangeable) for centre-clustered SE.
- Main inference: interaction p-value (GEE) in `interaction_p.csv`; coefficients in `glm_parameters.csv` / `gee_parameters.csv`.
- Volume-tier curves: marginalized predictions (g‑computation) across BMI 18–45 at median volume per tier (Low/Mid/High), with 95% CIs via delta method using the model’s cluster‑robust covariance. Files: `bmi_volume_curves.csv`, `bmi_interaction_tiers.svg/png`.
 - Volume definition (`--volume-tier-mode`):
   - `tertiles`: total cases per centre over the dataset
   - `annual_threshold`: mean annual volume per centre; tiers: <5 / 5–10 / >10 cases/year
- Marginal standardization: for each BMI value and tier, predict for all patients (BMI fixed, tier volume applied), then average predictions to reflect the empirical covariate mix; CIs via delta method (average gradient p(1−p)·x and robust covariance). This avoids artifacts from a single “median patient”.
- Prespecified pointwise comparisons: High vs Low at BMI 30/35/40 using delta method for probability differences, 95% CI, p-value (Holm if >1 test). File: `pairwise_bmi_comparisons.csv`; displayed in `pairwise_bmi_bars.svg/png`.
- Sensitivity (supplement): OR per +5 BMI and ARD 40 vs 25 by tier (`or_per5_by_volume_tier.csv`, `ard_bmi25_40_by_tier.csv`), alternative volume thresholds (`volume_threshold_tests.csv`), three-way interactions with malignant/robotic status (`threeway_interactions.csv`).
