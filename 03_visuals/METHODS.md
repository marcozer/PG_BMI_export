## Méthodes – Composantes limitantes de l’Ideal Outcome

- Mesures par classes BMI (OMS) : taux de CR‑POPF B/C, Clavien ≥III, conversion, réadmission, réintervention. Tableau : `outputs/component_rates.csv`; figure : `outputs/component_stack.svg/png`.
- Tests globaux par composante : chi‑deux d’association (table croisée BMI_class × composante) ; résultats dans `outputs/component_chi_square.csv`.
- Lien avec le volume (CR‑POPF B/C) :
  - 2×2 (BMI <30 vs ≥30) × (volume Low/Mid/High regroupé en High vs autres)
  - Chi‑deux et logit POPF ~ BMI30 × HighVolume; p‑interaction dans `outputs/popf_bmi_volume_tests.csv`.
- Définition du volume (option `--volume-tier-mode`) : tertiles (par défaut) ou moyenne annuelle par centre, tiers <5 / 5–10 / >10 cas/an.

## Methods – Outcome component breakdown

- By WHO BMI classes we computed rates of CR‑POPF B/C, Clavien ≥III, conversion, readmission, reoperation (`outputs/component_rates.csv`), displayed as stacked bars (`outputs/component_stack.svg/png`).
- Global tests per component: chi‑square on BMI_class × component (`outputs/component_chi_square.csv`).
- Volume link for CR‑POPF B/C: 2×2 table (BMI <30 vs ≥30) × (High vs others) with chi‑square and logistic regression POPF ~ BMI30 × HighVolume (`outputs/popf_bmi_volume_tests.csv`).
- Volume tiering (`--volume-tier-mode`): tertiles (default) or mean annual cases per centre with thresholds <5 / 5–10 / >10 cases/year.
