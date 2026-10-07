# Methodology

Pipeline: dataset → data-quality audit → train/test strategy → preprocessing → repeated-CV model comparison → probability calibration → pre-declared model selection → test-set evaluation → threshold analysis → explainability → reliability and robustness → prediction service → UI.

Measured results are in [evaluation.md](evaluation.md). Intended use and limitations are in [model_card.md](model_card.md).

## 1. Dataset and why it was chosen
The project uses the Cardiovascular Disease dataset by S. Ulianova on Kaggle (`sulianova/cardiovascular-disease-dataset`). It contains 70,000 records with 11 everyday health inputs and a binary disease label.

It replaced the 303-record UCI Cleveland data used in the earliest prototype, for two reasons:
- **Size:** with about 68,500 usable records, cross-validation, calibration, bootstrap intervals and subgroup analysis become statistically meaningful.
- **Everyday inputs:** a non-specialist can actually supply age, blood pressure, height, weight and lifestyle answers. Angiography or thallium-scan results cannot be supplied that way.

**Dataset limitations:**
- Provenance is only briefly described, and the label definition is not detailed.
- Lifestyle inputs are self-reported.
- Cholesterol and glucose are three-level categories, not lab values.
- The data covers ages 29–65 only.

`src/data_loader.py` downloads the file from Kaggle's public API and checks it against a SHA-256 hash. The data is not redistributed in this repository.

## 2. Data-quality audit and cleaning
The raw file has no missing values and no duplicate ids. It does contain 24 records that are duplicates apart from `id`, plus many impossible measurements, for example blood pressures of −150 or 16,020 and heights of 55 cm.

The documented rules below are applied in order. They are data-quality rules, not medical thresholds, and they were not tuned for performance.

| Rule | Rows removed |
|---|---|
| Malformed, missing or non-numeric values | 0 |
| Target not 0 or 1 | 0 |
| Undocumented category code | 0 |
| Duplicate record (ignoring `id`) | 24 |
| Age outside 29–65 years | 0 |
| Height outside 120–220 cm | 53 |
| Weight outside 30–250 kg | 7 |
| Systolic outside 60–250 mmHg | 226 |
| Diastolic outside 30–200 mmHg | 989 |
| Systolic not above diastolic | 103 |
| BMI outside 12–60 | 25 |

That leaves 68,573 records. `clean_with_exclusions` keeps the removed rows for the Data Quality Lab. The same ranges validate user input, so the model is never asked about values its training data excluded.

**Features:**
- Age is converted from days to years.
- BMI is the only engineered feature. It replaces raw height and weight; adding those back changed CV ROC-AUC by at most 0.0002.
- Pulse pressure (systolic − diastolic) was rejected because it is an exact linear combination of two existing features.
- `id` is never used.
- Gender code 1 is inferred to be female from mean height, because the source does not label the codes.

## 3. Train/test strategy and preprocessing
- **Split:** stratified 80/20 with seed 42, giving 54,858 training and 13,715 test records. Duplicates are removed before splitting.
- **Preprocessing:** one scikit-learn `Pipeline` that derives BMI, scales the numeric features and one-hot encodes the documented category codes. It lives inside each model pipeline, so every learned step is fitted on training folds only.
- **Test set:** scored once per model, after selection. No model, calibrator, threshold or novelty detector is fitted or chosen on it, and automated tests check this (section 13).

## 4. Model comparison with repeated cross-validation
Three models are compared: Logistic Regression, Random Forest (200 trees, depth 10) and XGBoost (300 trees, depth 4). These hyperparameters are fixed in `build_models` and have not changed since Phase 1. No hyperparameter search was run, on any data. Each is evaluated raw, with sigmoid calibration and with isotonic calibration, using `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)` on the training split. That gives 25 train/validation splits, shared by all nine variants.

`oof_cross_validate` fits a fresh clone of the whole pipeline on each split's training rows, including any calibrator. It then scores eight metrics on the split's validation rows. Each metric is reported as mean, standard deviation, minimum and maximum over the 25 splits, not as the mean alone.

| Group | Metrics |
|---|---|
| Threshold metrics (at 0.50) | accuracy, precision, recall, F1 |
| Ranking | ROC-AUC, PR-AUC |
| Probability quality | log loss, Brier score |

**Out-of-fold (OOF) predictions.**
- **Coverage:** within each repeat, every training row is predicted exactly once, by a model (and calibrator) fitted without it.
- **What is stored:** `artifacts/oof_predictions.npz` stores, for every variant, a (5 repeats × 54,858 rows) probability array. It also stores each row's validation-fold number per repeat, the training-set index and the labels. The predicted class is probability ≥ 0.50.
- **Not committed:** the file is regenerated by training.

**Where OOF predictions are used, and why that is valid:**
- **Calibration analysis (curves, ECE, slope, intercept).** Every OOF probability is a genuine out-of-sample prediction. Statistics are computed per repeat (each row once) and then averaged. Pooling the repeats would count each row five times.
- **Threshold exploration.** Exploring thresholds on training-split OOF predictions does not touch the test set.

**What OOF predictions are not used for:** they are not treated as a generalisation estimate of the final refitted model. That estimate is the test set's job.

## 5. Probability calibration
Each calibrated variant is `Pipeline[preprocess → CalibratedClassifierCV(classifier, method, cv=3, ensemble=False)]`.
- **Inside the calibrator:** the classifier is refitted on all rows it receives, and one calibrator is learned from 3-fold OOF scores of those rows.
- **Inside an outer CV split:** those rows are only the split's training rows, so the calibrator never sees validation or test labels. That makes the calibration nested.

Probability quality is measured by Brier score, log loss, ECE (10 equal-width bins) and calibration slope and intercept.
- **Slope:** a logistic regression of the outcome on logit(p).
- **Intercept:** calibration-in-the-large, with the slope fixed at 1.

Slope and intercept only capture linear miscalibration on the logit scale. Raw Logistic Regression's S-shaped miscalibration (slope ≈ 1 but ECE 0.034) shows why the reliability curve is also needed.

## 6. Model-selection protocol
**History.** The previous version selected the model with margins of 0.02 ROC-AUC and 0.001 Brier. Those margins were fixed after the first comparison was known. That was a methodological weakness, and the margins have been removed from the active selection logic.

**The protocol.** The protocol below was written into `src/train_models.py` before the repeated-CV comparison was run. It contains no tunable number other than the conventional α = 0.05.
- **Paired test:** every comparison is a corrected resampled t-test on paired per-split scores (Nadeau & Bengio 2003). Repeated-CV splits share training rows, so the paired-difference variance is inflated by (1/k + n_test/n_train), with k = 25 and df = 24.
- **Calibration (per model):** keep raw probabilities unless a calibrator lowers CV Brier significantly.
- **Explanation-quality gate:** a model is eligible only if an exact additive SHAP explanation of its score exists and the score maps monotonically to the displayed probability.
- **Model:** start from the most interpretable model (Logistic Regression → Random Forest → XGBoost). Move on only if the next model is significantly better on both ROC-AUC and Brier.

**How each priority is represented:**

| Priority | How the protocol uses it |
|---|---|
| Validation performance | ROC-AUC significance test |
| Calibration | Brier significance test, plus calibrator choice |
| Stability | The test's variance term penalises split-to-split variability; per-split ranges are reported |
| Interpretability | The default order |
| Explanation quality | The gate |

**Result.** Logistic Regression became isotonic, Random Forest isotonic and XGBoost raw. Random Forest beat Logistic Regression on both metrics (p < 1e-9), and XGBoost then beat Random Forest on both (p ≈ 0.01). **The final model is XGBoost with raw probabilities.**

The protocol was not changed after this outcome was known. Reliability analyses run after selection show its costs: less smooth responses and less stable explanations (section 12). These are reported as limitations. They are not grounds for a post-hoc re-selection.

`choose_variant` and `select_final` are tested so that they reproduce the recorded decision, keep it when every test-set number is removed or scrambled, and behave correctly on synthetic inputs.

## 7. Threshold analysis
`threshold_metrics` computes TP, FP, TN and FN, plus precision, recall (sensitivity), specificity, F1, accuracy, false-positive rate and false-negative rate.
- **Grid:** 0.05 to 0.95 in steps of 0.01.
- **Sources:** the final model's OOF predictions (training split, for exploration) and the test set (descriptive).

`decision_curve` adds net benefit (TP/n − FP/n × t/(1 − t)) against "flag everyone" and "flag no one". It is labelled exploratory and dataset-specific.

**Nothing is tuned:** no threshold is selected, and the class output stays at 0.50.

Three different concepts are kept apart:
- **Model probability:** the model's estimate.
- **Display band:** the prototype presentation categories <30%, 30–60% and ≥60%.
- **Classification threshold:** the cut-off for the class output.

None of them is clinically optimal.

## 8. Uncertainty
`bootstrap_ci` runs a paired percentile bootstrap: 1,000 resamples of the test rows, seed 42, with the 2.5th–97.5th percentiles as a 95% interval. It covers ROC-AUC, PR-AUC, F1 at 0.50, Brier and ECE for all three models, plus each model's difference from the final model.

Binned ECE is biased upward under resampling, so its interval can lie above the point estimate. The intervals describe uncertainty around evaluation metrics on this sample. They are not clinical uncertainty.

## 9. Subgroup Performance Analysis
`subgroup_metrics` reports, for gender and for the age bands [29, 40), [40, 50), [50, 60) and [60, 66), on the test set:
- N, prevalence and mean predicted probability;
- ROC-AUC, PR-AUC, recall, precision, F1 and Brier;
- within-group bootstrap 95% intervals (500 resamples) for ROC-AUC, PR-AUC, recall and Brier.

Groups with only one class, or with fewer than 30 records of either class, get "Metric unavailable for this subgroup" instead of numbers. Nothing is tuned on these results, and the analysis makes no fairness claim.

## 10. Explainability
- **Local SHAP:** `ModelExplainer` supports all three candidates.
  - *Logistic Regression:* `shap.LinearExplainer` with all training rows as background. Values are exact (`coef × (x − mean)`).
  - *Random Forest and XGBoost:* TreeSHAP (`tree_path_dependent`), which is exact for the trees' own output.
  - *Score spaces:* log-odds for Logistic Regression and XGBoost; probability for Random Forest.
  - *Readable features:* one-hot columns are summed back into readable features.
  - *Displayed probability:* link(score), then the calibrator if there is one. Each step is monotone, so directions are preserved but sizes are not additive in percentage points.
- **Global SHAP:** mean |SHAP| over 5,000 random cleaned records (training and test rows). This describes the fitted model's attributions; nothing is fitted or chosen from it.
- **Permutation importance:** the drop in test ROC-AUC when an input is shuffled (10 repeats).
- **Partial dependence and ICE:** 2,000 test records, all three candidates, skipping invalid grid points.
- **Interactions:** XGBoost SHAP interaction values on 1,000 test records.
- **What-if:** `src/what_if.py` changes only adjustable inputs and rebuilds every scenario through `PatientInput`.

## 11. SHAP and model-output reconciliation
For every candidate, base value + Σ SHAP must equal the model's internal score, and the probability derived from it must equal the pipeline's probability.
- **Tolerance:** 1e-5. Logistic Regression and Random Forest reconcile to about 1e-15; XGBoost to about 3e-6, because it computes in float32.
- **XGBoost base value:** shap 0.52 reports it about 6e-4 log-odds off the model's base margin, constantly across rows. The explainer re-anchors the base value, and refuses to run if the offset is not constant.
- **Guards:** tests check this on the demo inputs and on 200 test records. `reliability.json` records the maximum errors on 1,000 records.

## 12. Reliability and Robustness
All of these analyses run offline (`python -m src.reliability` and `python -m src.shift_analysis`), and the UI reads their artifacts. Per assessment, the UI only runs lightweight steps: prediction, local SHAP, the novelty check, a 28–30-row local stability check and three candidate predictions.

- **Repeated CV (section 4):** shows how much each metric depends on the particular split.
- **Out-of-fold predictions (section 4):** the basis for calibration and threshold exploration.
- **Calibration (section 5)** and **threshold analysis (section 7).**
- **Bootstrap uncertainty (section 8)** and **subgroup analysis (section 9).**
- **Input conformity (novelty detection).**
  - *Space and data:* the model's preprocessed feature space, refitted on 80% of the training split. The threshold is the 99th percentile of scores on the remaining 20% (reference flag rate 1%).
  - *Methods compared:* a Mahalanobis distance on the four scaled continuous features, and an Isolation Forest on all 14 columns.
  - *Pre-declared rule:* the higher ROC-AUC for separating the reference rows from "atypical combinations" (reference values recombined at random, kept only if valid) wins; a tie goes to the simpler method. The test set's flag rate is reported afterwards, descriptively.
  - *Phase 9 correction:* until Phase 8 this comparison used test records, so a detector choice was made with test-set data. That contradicted section 3. It now uses training rows only, and a test checks that the choice does not depend on the test set (section 13).
  - *Result:* Mahalanobis won (0.653 vs 0.560). The earlier test-based comparison had chosen the same method (0.648 vs 0.557), and the threshold is unchanged.
  - *Why not robust MCD covariance:* it degenerates because blood pressures are heavily rounded.
  - *Novelty, not outlier detection:* following scikit-learn's distinction, the detector is fitted on reference data only and then applied to new inputs. It never refits on, or learns from, the user's input.
  - *Limitations:* it separates atypical combinations only weakly, so it mainly catches extreme values. It is a statistical caution, never a validation error or a medical judgement.
- **Prediction stability.**
  - *Inputs perturbed:* age, height, weight, systolic and diastolic are each changed by ±1%, ±2% and ±5% alone. There are also 4 random joint perturbations per size, from 153 baseline profiles.
  - *Measured:* the |Δprobability| distribution per model and per input.
  - *Not perturbed:* categorical inputs have no small change.
- **Explanation stability.** SHAP is recomputed for every perturbed version and compared with its baseline:
  - Spearman rank correlation of |SHAP|;
  - top-5 overlap (|top-5 ∩ top-5′| / 5);
  - normalised L1 distance (Σ|φ − φ′| / Σ|φ|).

  These are summarised descriptively; there is no universal "stable" threshold. The Explain page runs the one-input-at-a-time version for each assessment.
- **Monotonicity sanity analysis.** One input (systolic, diastolic, age, weight) is varied over its accepted range with everything else fixed, for 23 profiles and all three candidates. The curve shape (non-decreasing, non-increasing or non-monotone) and the largest single-step decrease are recorded. Nothing is constrained, and directions are described as model behaviour, never as causes.
- **Model disagreement.** The spread is max − min of the three candidates' probabilities. It is reported as a distribution over the test set and for representative cases, and per assessment with its test-set percentile. It is called model disagreement, not uncertainty.
- **Assessment reliability panel.** The Explain page shows five separate signals: probability, input conformity, explanation stability, model disagreement and calibration status. They are deliberately not combined into a score, because any weighting would be arbitrary.
- **Controlled distribution-shift experiment.** This is synthetic. It uses population-mix resampling of the test set (weights exp(β·z), labels kept) for age, systolic BP and BMI, and measurement offsets (blood pressure ±5/10 mmHg, weight ±5%, age +3 years) with labels unchanged. It is a separate artifact flagged `synthetic: true`, and it is not a validation.

## 13. Test-set integrity checks
- **Selection never reads the test set.** The selection functions only read CV results, and a test confirms the decision is unchanged when every test-set number is scrambled or removed.
- **Calibrated variants are never scored on the test set.** Only each model's selected variant is scored, once, after selection.
- **The final pipeline used training rows only.** Refitting the selected variant on the training split reproduces the persisted final pipeline (to 1e-6). The same holds for the Logistic Regression candidate (to 1e-12), so no fitted step, calibrator included, saw test rows.
- **The OOF file covers only training rows,** with valid fold metadata. The novelty detector's fitting rows are training rows only, and its method choice is unchanged when a different test set is passed in.
- **No threshold is tuned:** no artifact contains a selected threshold.

## 14. Prediction service and UI
The component diagram and module map are in [architecture.md](architecture.md).

- **Prediction service:** `src/prediction.py` loads `models/final_model.joblib`, checks the feature contract, validates input and calls the selected pipeline. `src/recommendations.py` adds rule-based informational guidance.
- **Precomputation:** heavy analyses are precomputed into `artifacts/*.json`. The Streamlit UI caches the models, explainer and novelty detector, and never recomputes global analyses per request.

## 15. Reproduction
```bash
python -m src.train_models     # about 5 min: 9 variants x 25 splits, selection, persistence, OOF file
python -m src.analysis         # about 1 min: test-set analyses -> artifacts/analysis.json
python -m src.reliability      # about 1.5 min: reliability analyses, novelty detector
python -m src.shift_analysis   # seconds: synthetic shift experiment
python -m pytest -q
```
