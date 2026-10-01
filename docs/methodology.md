# Methodology

Pipeline: dataset → data-quality audit → train/test strategy → preprocessing → model comparison → probability calibration → threshold analysis → final model selection → explainability → robustness and subgroup analysis → prediction service → UI.

Measured results are in [evaluation.md](evaluation.md). Intended use and limitations are in [model_card.md](model_card.md).

## 1. Dataset and why it was chosen
The project uses the Cardiovascular Disease dataset by S. Ulianova on Kaggle (`sulianova/cardiovascular-disease-dataset`). It contains 70,000 records with 11 everyday health inputs and a binary disease label.

It replaced the 303-record UCI Cleveland data for three reasons:
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
- **Test set:** used only after all choices have been made.

## 4. Model comparison
Three models are compared: Logistic Regression, Random Forest (200 trees, depth 10) and XGBoost (300 trees, depth 4). Each is evaluated raw, with sigmoid calibration and with isotonic calibration, using stratified 5-fold CV on the training split.

`oof_cross_validate` fits a fresh clone of the whole pipeline (including any calibrator) on each fold's training rows. It then scores eight metrics on the fold's validation rows and keeps the out-of-fold probabilities.

| Group | Metrics |
|---|---|
| Threshold metrics (at 0.50) | accuracy, precision, recall, F1 |
| Ranking | ROC-AUC, PR-AUC |
| Probability quality | log loss, Brier score |

Model selection never relies on accuracy alone or on the test set.

## 5. Probability calibration
Each calibrated variant is `Pipeline[preprocess → CalibratedClassifierCV(classifier, method, cv=3, ensemble=False)]`. Within any training data it is given, the classifier is refitted on all rows, and one calibrator is learned from 3-fold out-of-fold scores. Calibration therefore never sees validation or test labels.

**Rule:** a calibrated variant is adopted only if it lowers mean CV Brier by at least 0.001 *and* lowers it in every fold.

**Result:**
- Logistic Regression became isotonic. Raw Logistic Regression was systematically miscalibrated.
- Random Forest and XGBoost stayed raw; their calibration gains were not material.

The deployed pipeline is that exact calibrated object.

## 6. Final model selection
The framework walks from the most to the least interpretable model. It moves to a less interpretable model only if that model gains at least 0.02 CV ROC-AUC without worse Brier.

Both tree models gained about 0.009 to 0.010, so the final model is **Logistic Regression with isotonic calibration**. The margins are judgement calls, fixed after the first comparison was known.

`select_final` and `choose_variant` are tested so that they reproduce the recorded decision. Training stops with an error if the framework ever picks a model the explainer does not support.

## 7. Threshold analysis
`threshold_metrics` computes TP, FP, TN and FN, plus precision, recall, specificity, F1, false-positive rate and false-negative rate.
- It runs at 0.10, 0.20, …, 0.90 and on a 0.01 grid.
- It uses the selected model on the test set.

Three different concepts are kept apart:
- **Probability:** the model's calibrated estimate.
- **Display band:** the prototype presentation categories <30%, 30–60% and ≥60%.
- **Classification threshold:** the cut-off for the class output, default 0.50. None of these is clinically optimal.

## 8. Uncertainty
`bootstrap_ci` runs a paired percentile bootstrap: 1,000 resamples of the test rows with seed 42, giving 95% intervals.
- It covers ROC-AUC, PR-AUC, F1 at 0.50 and Brier for all three models.
- It also gives intervals for each model's difference from Logistic Regression.

These intervals describe uncertainty from the evaluation sample, not clinical uncertainty.

## 9. Subgroup Performance Analysis
`subgroup_metrics` reports, for gender and for the age bands [29, 40), [40, 50), [50, 60) and [60, 66):
- n and the observed rate;
- the mean predicted probability (calibration-in-the-large);
- ROC-AUC, precision, recall, F1 and Brier.

Groups with fewer than 30 records of either class are reported without metrics. The analysis identifies differences and makes no fairness claim. The notable finding is lower ROC-AUC at ages 60–65.

## 10. Explainability
- **Local SHAP:** `shap.LinearExplainer` on the logistic regression inside the calibrator. The background is all 54,858 preprocessed training records. Values are exact (`coef × (x − mean)`) and additive on the model's score. The explainer reads the one-hot column layout from the fitted encoder and sums those columns back into readable features. The displayed probability is the isotonic map of that score, which is monotone (direction-preserving) but not additive in percentage points.
- **Global SHAP:** mean |SHAP| over all cleaned records. Direction is the sign of the value-vs-contribution slope for numeric features, and the mean contribution per level for categorical features.
- **Permutation importance:** the drop in test ROC-AUC when an input is shuffled (10 repeats, scikit-learn `permutation_importance`). It answers a different question from SHAP, and works on the 11 raw inputs.
- **Partial dependence and ICE:** a manual implementation over the 2nd–98th percentile of systolic BP, age, diastolic BP and weight, on 2,000 test records. It skips grid points that would create an invalid record and records how many rows remain valid. It is shown for the selected model and for XGBoost, with the caveat about the systolic/diastolic correlation.
- **Interactions:** XGBoost SHAP interaction values on 1,000 test records, grouped to features. The selected logistic regression has no interaction terms.
- **What-if:** `src/what_if.py` changes only adjustable inputs and rebuilds every scenario through `PatientInput`, so impossible combinations are rejected. Results are reported in percentage points and labelled as model sensitivity, not as outcomes.

## 11. Prediction service and UI
- **Prediction service:** `src/prediction.py` loads `models/final_model.joblib`, checks the feature contract, validates input and calls the calibrated pipeline. `src/recommendations.py` adds rule-based informational guidance.
- **Precomputation:** heavy analyses are computed once into `artifacts/analysis.json`. The Streamlit UI reads them and never recomputes global analyses per request.

## 12. Reproduction
```bash
python -m src.train_models   # about 1.5 min: CV comparison of 9 variants, selection, persistence
python -m src.analysis       # about 30 s: test-set analyses into artifacts/analysis.json
python -m pytest -q
```
