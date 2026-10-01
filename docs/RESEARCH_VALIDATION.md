# Research validation and evidence audit

This document explains how the evidence shown in the app was produced, where each claim comes from, and what the Phase 9 audit found. Detailed tables are in [evaluation.md](evaluation.md), procedures in [methodology.md](methodology.md), and intended use in [model_card.md](model_card.md). Every number below is read from a committed artifact in `artifacts/`; no external results or citations are claimed beyond the two statistical methods named in section 4.

**Scope.** This is an educational decision-support prototype. Every result describes agreement with the labels of one public dataset. None of it is clinical validation, and nothing here establishes medical usefulness.

## 1. Research objective
Estimate the probability of the dataset label `cardio = 1` from 11 self-reportable inputs. The aim is to show, with honest evidence, how well such a model ranks and calibrates on this dataset, how it explains itself, and where it is unreliable. Building a clinical tool is not an aim.

## 2. Dataset
- **Source:** Kaggle `sulianova/cardiovascular-disease-dataset`, downloaded from the public API and pinned by SHA-256 (`src/data_loader.py`).
- **Size:** 70,000 raw records; 68,573 after cleaning (34,646 absent, 33,927 present). These counts come from `metrics.json → dataset`.
- **Target:** `cardio`, where 1 = cardiovascular disease present. The source does not define the label in detail.
- **Inputs:** age, gender, height, weight, systolic (`ap_hi`), diastolic (`ap_lo`), cholesterol and glucose (3 levels each), smoking, alcohol, physical activity.
- **Coverage:** ages 29–65 only. 65% of records have gender code 1, which is inferred to be female.
- **History:** the earliest prototype used the 303-record UCI Cleveland data. It is no longer used anywhere in the pipeline.

## 3. Preprocessing
1. **Cleaning** runs ordered data-quality rules (`clean_with_exclusions`): malformed values, invalid target or codes, 24 duplicates, implausible ranges, systolic ≤ diastolic and implausible BMI. It removes 1,427 rows. These are recording-error rules, not tuned for performance.
2. **Split:** stratified 80/20 with seed 42, giving 54,858 training and 13,715 test records.
3. **Pipeline:** derive BMI (replacing height and weight), standard-scale the numeric features, and one-hot encode the documented codes. This lives inside every model pipeline, so it is fitted on training folds only.

## 4. Experimental protocol
- **Candidates:** Logistic Regression, Random Forest (200 trees, depth 10) and XGBoost (300 trees, depth 4). Hyperparameters were fixed in Phase 1, and no hyperparameter search was ever run.
- **Variants:** each model is evaluated raw, sigmoid-calibrated and isotonic-calibrated (`CalibratedClassifierCV`, cv=3, nested inside each outer training fold).
- **Resampling:** `RepeatedStratifiedKFold(5, 5, seed 42)` on the training split, giving 25 splits shared by all 9 variants.
- **Statistical test:** a corrected resampled t-test (Nadeau & Bengio, 2003) on paired per-split scores, α = 0.05.
- **Test set:** scored once per model, after selection.

## 5. Model comparison (CV, selected variants; `metrics.json → comparison`)
| Model | ROC-AUC | Brier |
|---|---|---|
| Logistic Regression (isotonic) | 0.791 ± 0.004 | 0.185 ± 0.002 |
| Random Forest (isotonic) | 0.799 ± 0.004 | 0.181 ± 0.002 |
| XGBoost (raw) | 0.801 ± 0.004 | 0.181 ± 0.002 |

## 6. Model selection (`metrics.json → selection`)
The protocol was declared in `src/train_models.py` before the repeated-CV run:
1. Keep raw probabilities unless a calibrator significantly lowers CV Brier.
2. A model needs an exact SHAP explanation to be eligible.
3. Walk LR → RF → XGB, moving on only for a significant gain in **both** ROC-AUC and Brier.

The steps recorded in the artifact were:
- RF vs LR: ROC-AUC +0.0084 (p = 2.9e-10), Brier −0.0040 (p = 7.9e-11). Switch to RF.
- XGB vs RF: ROC-AUC +0.0012 (p = 0.0087), Brier −0.0005 (p = 0.010). Switch to XGB.
- XGB calibration: neither calibrator lowered Brier, so raw probabilities were kept.

**Final model: XGBoost, raw probabilities.** The gains are small but consistent across splits. An earlier version (commit `da15801`) used margins chosen after seeing results; that history is disclosed in evaluation.md §1. The protocol is not blind, because the author had seen earlier single-CV results, but it has no tunable number.

## 7. Final evaluation (test set; `analysis.json → models, bootstrap`)
| Metric | XGBoost | 95% bootstrap interval |
|---|---|---|
| ROC-AUC | 0.804 | 0.797–0.812 |
| PR-AUC | 0.785 | 0.774–0.796 |
| Brier | 0.180 | 0.176–0.183 |
| F1 / accuracy / precision / recall at 0.50 | 0.722 / 0.737 / 0.755 / 0.692 | F1: 0.714–0.731 |

Logistic Regression's test ROC-AUC is 0.011 lower (paired interval 0.009–0.014). These intervals describe sampling variability on this test set only.

## 8. Calibration (`reliability.json → calibration`)
- **XGBoost out-of-fold (training split):** ECE 0.004, slope 0.99.
- **XGBoost on the test set:** ECE 0.008, slope 1.02, intercept −0.009.
- **Logistic Regression:** its raw miscalibration is S-shaped (ECE 0.034 with slope ≈ 1). Isotonic calibration fixes it; sigmoid calibration cannot.

"Calibrated" means agreement with this dataset's label frequencies.

## 9. Explainability (`src/explainability.py`; `analysis.json → shap_global, permutation_importance, interactions`)
- **Method:** exact TreeSHAP on XGBoost's log-odds score, with one-hot columns summed back to readable features. The displayed probability is the sigmoid of that score, so contributions keep their direction but are not percentage points.
- **Reference baseline:** for trees, TreeSHAP uses the training data's leaf coverage. Logistic Regression's background is the training split. No test rows are involved.
- **Global attribution:** systolic 49%, age 16%, cholesterol 13%, BMI 7%, diastolic 6%. Permutation importance (test ROC-AUC drop) ranks systolic, age and cholesterol first.
- **Interactions:** about 30% of XGBoost's attribution.
- **Reconciliation** (`reliability.json → reconciliation`): base + ΣSHAP equals the model score within 3e-6. The probability rebuilt from SHAP matches the pipeline within 4e-7.
- **Not causal:** the UI states "Model contribution ≠ medical causation". Smoking, alcohol and very high glucose get negative contributions, and these are shown as confounded dataset patterns.

## 10. Robustness (`reliability.json → perturbation, explanation_stability, monotonicity`)
- **Prediction stability:** a ±1% change in one numeric input moves XGBoost's estimate by a median of 0.7 pp, a 95th percentile of 11.4 pp and a maximum of 37.4 pp. The figures for LR are 0.0 / 2.7 / 6.0.
- **Explanation stability:** under ±1% changes, the top-5 features stay identical in 74% of cases for XGBoost, against 95% for LR and 92% for RF.
- **Monotonicity:** XGBoost is non-monotone in systolic, diastolic, age and weight for all 23 tested profiles. LR is non-decreasing for all of them.

These are weaknesses of the selected model. They were measured after selection and reported; they were not used to re-select.

## 11. Reliability signals (`reliability.json → novelty, disagreement`)
- **Input conformity:** a Mahalanobis distance on the four scaled continuous features, fitted on 43,886 training rows, with its threshold at the 99th percentile of 10,972 held-back training rows. It was chosen over an Isolation Forest by ROC-AUC against atypical recombinations of those reference rows (0.653 vs 0.560). The test-set flag rate, reported afterwards, is 1.04%, against 1% by construction on the reference rows. It is a caution only and never blocks an estimate.
- **Model disagreement:** across the three candidates on the test set, the spread has a median of 5.1 pp and a 95th percentile of 17.3 pp. All three give the same class at 0.50 for 92.3% of records.
- **No combined score:** the Explain page shows probability, conformity, explanation stability, disagreement and calibration status separately. No "reliability score" exists.

## 12. Subgroup observations (`analysis.json → subgroups`)
- **Gender:** test ROC-AUC 0.803 (code 1) and 0.806 (code 2).
- **Age:** test ROC-AUC 0.855, 0.830, 0.774 and 0.699 for the 29–39, 40–49, 50–59 and 60–65 bands. Performance differed across the evaluated age groups and was lowest at 60–65.
- **Agreement in the large:** mean predicted probability is within 0.02 of the observed rate in every group.
- **Limits of the analysis:** only gender and age bands are available. The data has no ethnicity, socioeconomic or clinical-history variables. No fairness, equity or safety claim is made.

## 13. Dataset-shift observations (`shift_analysis.json`, flagged `synthetic: true`)
The test set is resampled toward older, higher-systolic or higher-BMI people (β = +1). Under each shift, XGBoost's ROC-AUC falls to about 0.75, while mean predicted stays within about 0.01 of observed.

A systematic +10 mmHg recording offset raises the mean estimate from 0.496 to 0.638, against an unchanged observed rate of 0.495.

This is a synthetic sensitivity experiment, not external validation.

## 14. Limitations
- The model is validated on one dataset only, with limited provenance, self-reported lifestyle inputs and coarse lab categories.
- It has learned confounded patterns (smoking, alcohol).
- Its estimates move in steps and are non-monotone. Explanations are less stable than a linear model's.
- Ranking is weak at ages 60–65.
- It has no external or prospective validation, and no evaluation of clinical utility.
- Setup has been tested on macOS arm64 with Python 3.12 only.

## 15. Reproducibility
- **Commands:** `./scripts/setup.sh --regenerate`, or run `python -m src.train_models`, then `src.analysis`, then `src.reliability`, then `src.shift_analysis`.
- **Pins:** seed 42 everywhere, a dataset SHA-256, pinned library versions, and a `provenance` block in every artifact.
- **Repeatability:** Logistic Regression reproduces exactly. The tree models reproduce to floating-point precision (multithreaded training), and selection decisions are identical.
- **Suite:** `python -m pytest -q`.

## Traceability: claim → evidence
| Claim shown to users | Evidence source | Read by |
|---|---|---|
| XGBoost selected, raw probabilities | `metrics.json → selection, final_model` | Model → Overview |
| CV metrics and t-tests | `metrics.json → comparison, selection.tests` | Model → Overview |
| Test ROC-AUC, Brier and intervals | `analysis.json → models, bootstrap` | Model → Overview/Uncertainty, Methodology → Limitations |
| Calibration statistics | `reliability.json → calibration` | Model → Calibration, Explain → reliability panel |
| Threshold and net-benefit trade-offs | `analysis.json → thresholds`, `reliability.json → thresholds_oof` | Model → Thresholds |
| Subgroup differences | `analysis.json → subgroups` | Model → Subgroups, Methodology → Limitations |
| Per-assessment SHAP contribution | `src/explainability.py` at request time | Assess, Explain |
| Global SHAP, permutation, PDP/ICE, interactions | `analysis.json` | Model → Explainability |
| Non-monotone response | `reliability.json → monotonicity` | Explore, Model → Robustness |
| Stability, disagreement, conformity | `reliability.json`, `models/novelty_detector.joblib` | Explain, Model → Robustness |
| Dataset size, cleaning, correlations, rounded blood pressures | `analysis.json → data_quality` | Methodology → Data quality lab, Model, Explore |
| Synthetic shift | `shift_analysis.json` | Model → Robustness |

The UI computes per-assessment values live and reads every other number from these artifacts. Fixed design constants (bands at 30% and 60%, the 0.50 class threshold, plausibility ranges) are code constants and are labelled as prototype choices.

## Phase 9 audit record
| Area | Result |
|---|---|
| Test-set leakage | **One defect found and fixed.** The input-conformity detector's method choice (Mahalanobis vs Isolation Forest) was scored on test records, contradicting the "nothing chosen on the test set" rule. It now uses held-back training rows; a new test checks that the choice is independent of the test set. The same method and threshold are chosen as before (AUC 0.653 vs 0.560 on training rows; previously 0.648 vs 0.557 on test rows), so app behaviour is unchanged. The model, calibration, thresholds, preprocessing and SHAP baselines were already test-free. |
| Model selection | The recorded decision matches the protocol and is reproduced by retraining. No change. |
| Metric consistency | Doc figures were checked against the artifacts. Seven double-rounded values in evaluation.md (two repeated in the model card) were corrected (e.g. LR ±1% 95th percentile 2.8 → 2.7 pp, XGBoost ECE upper bound 0.018 → 0.017). UI prose that hard-coded numbers (calibration finding, correlation, age-band ROC-AUC, test ROC-AUC, prevalence, rounded blood pressures) now reads the artifacts. |
| Wording | "Calibrated" was used for the final model's raw probabilities in the Assess page, README and docstrings, and was replaced. No fairness, safety or causal claims were found. |
| Documentation | The analysis docstring now states that global SHAP uses train and test rows (descriptive only). Hyperparameters are documented as fixed and untuned. The model card gained ethical considerations and reproducibility sections. |
