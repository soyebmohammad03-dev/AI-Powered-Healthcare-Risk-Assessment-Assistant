## Model card: Cardiovascular disease probability model

**Status: educational research prototype. Not clinically validated. Not a medical device.**

### Model purpose
The model estimates the probability that a record carries the dataset's "cardiovascular disease present" label (`cardio = 1`), given 11 structured health inputs. The estimate is shown together with an explanation and general information in a human-centred AI prototype.

### Intended use
- Teaching and demonstrating explainable, human-centred AI on tabular health data.
- Exploring how a trained model weights inputs, how calibrated its probabilities are, and where its performance differs.
- Portfolio and academic evaluation of ML methodology.

### Not intended for
- Diagnosis, screening, triage or any clinical decision.
- Treatment, medication or lifestyle prescriptions.
- Individual risk communication to patients as if the estimate were a validated clinical risk.
- Use on people outside the data's coverage, for example ages outside 29–65, or on populations unlike the source data.

### Dataset
| | |
|---|---|
| Source | Cardiovascular Disease dataset, S. Ulianova, Kaggle (`sulianova/cardiovascular-disease-dataset`). Provenance is only briefly described ("examination results"). |
| Size | 70,000 raw records, of which 68,573 remain after documented data-quality cleaning (1,427 removed: duplicates, impossible blood pressures, heights, weights and BMI). |
| Target | `cardio`: 1 = disease present (33,927 clean records), 0 = absent (34,646). |
| Split | Stratified 80/20, seed 42: 54,858 training records and 13,715 held-out test records. |

### Features
- **Inputs:** age (years), gender, height, weight, systolic and diastolic blood pressure, cholesterol and glucose (3 levels each), smoking, alcohol intake, physical activity.
- **Model features:** age, BMI (derived from height and weight), systolic, diastolic, plus the one-hot encoded categorical inputs.
- **Excluded:** `id` is never used.

### Preprocessing
1. BMI is derived inside the saved pipeline.
2. Numeric features are standard-scaled.
3. Categorical features are one-hot encoded with a fixed list of documented codes.

All of these steps are fitted on training data only. Input validation uses the same plausibility ranges as the data cleaning, plus two cross-checks: systolic must be above diastolic, and BMI must be within 12–60.

### Models evaluated
Logistic Regression, Random Forest and XGBoost. Each was tested in three versions: raw probabilities, sigmoid-calibrated and isotonic-calibrated. All nine were compared with 5-fold stratified cross-validation (CV) repeated 5 times, giving 25 splits of the training split.

### Final model
**XGBoost, raw (uncalibrated) probabilities.**

It was chosen by a selection protocol declared before the repeated-CV comparison was run:
- **Comparisons:** each is a corrected resampled t-test on the same 25 splits at α = 0.05.
- **Calibration:** a calibrator is adopted only if it lowers CV Brier significantly.
- **Model:** starting from the most interpretable model, the next one is adopted only if it is significantly better on both ROC-AUC and Brier.

Random Forest beat Logistic Regression, and XGBoost then beat Random Forest, both consistently: about +0.01 and +0.001 CV ROC-AUC respectively, each with lower Brier.

**Correction.** An earlier version selected Logistic Regression using margins (0.02 ROC-AUC, 0.001 Brier) chosen after the results were known. Those margins are no longer used.

### Evaluation (held-out test set, final model)
| Metric | Value | 95% bootstrap interval |
|---|---|---|
| ROC-AUC | 0.804 | 0.797–0.812 |
| PR-AUC | 0.785 | 0.774–0.796 |
| F1 at 0.50 | 0.722 | 0.714–0.731 |
| Brier score | 0.180 | 0.176–0.183 |
| ECE (10 bins) | 0.008 | 0.007–0.018 |
| Log loss | 0.538 | |
| Accuracy / precision / recall at 0.50 | 0.737 / 0.755 / 0.692 | |

CV ROC-AUC was 0.801 ± 0.004 (range 0.795–0.808 over 25 splits). Logistic Regression's test ROC-AUC is lower by 0.011 (paired interval 0.009–0.014). The bootstrap intervals describe uncertainty from the evaluation sample, not clinical uncertainty.

### Calibration
- **Final model:** out of fold, XGBoost's raw probabilities agree closely with observed outcome frequencies in this dataset (ECE 0.004, slope 0.99). Neither calibrator lowered CV Brier, so the protocol kept raw probabilities. On the test set, ECE is 0.008.
- **Logistic Regression candidate:** its raw miscalibration is S-shaped (slope ≈ 1 but ECE 0.034), and isotonic calibration corrects it.

"Calibration" here means agreement with this dataset's labels, not clinical accuracy.

### Explainability
- **SHAP:** exact TreeSHAP values for XGBoost's log-odds score, grouped into readable features. The base value is re-anchored for a constant 6e-4 offset in shap's reported expected value. Base value + contributions reconciles with the model's score within 1e-5 (documented float32 tolerance), and the displayed probability is the sigmoid of that score.
- **Interactions:** XGBoost has them (about 30% of its attribution), so one input's contribution can depend on the others.
- **Other analyses:** permutation importance, partial dependence and ICE curves, interaction values, and a guarded what-if analysis.
- **Top features:** systolic blood pressure, age and cholesterol.

### Reliability and robustness
Measured after selection, and reported rather than used to re-select:
- **Responses are less smooth.** Changing one numeric input by 1% moves XGBoost's estimate by a median of 0.7 percentage points (pp), but by 11 pp at the 95th percentile and up to 37 pp. Logistic Regression's equivalents are 0.0 / 2.8 / 6.0 pp. Recorded blood pressures cluster at round values (40% of systolic readings are exactly 120 mmHg), and the trees split near them.
- **The response is not monotone.** For all 23 tested profiles, XGBoost's estimate is non-monotone in systolic BP, diastolic BP, age and weight. The largest single-step drop along systolic BP was 14 pp. Logistic Regression was non-decreasing in every case.
- **Explanations are less stable.** Under ±1% changes, XGBoost's top-5 contributions stayed identical in 74% of cases (Logistic Regression 95%, Random Forest 92%). The top feature was preserved in 96%.
- **The candidate models disagree.** Across the test set, the spread between the three models' estimates has a median of 5 pp and a 95th percentile of 17 pp. All three give the same class at 0.50 in 92% of records.
- **Input conformity.** A Mahalanobis-distance check fitted on training data flags the 1% most unusual input profiles as a caution. It never blocks a prediction.
- **Synthetic distribution shifts.** Ranking quality falls to about 0.75 ROC-AUC in older or higher-blood-pressure populations. A systematic +10 mmHg blood-pressure recording offset raises the mean estimate from 0.50 to 0.64 at an unchanged outcome rate.

### Subgroup performance (test set, final model)
| Group | N | ROC-AUC [95% CI] | Prevalence | Mean predicted |
|---|---|---|---|---|
| Female | 8,972 | 0.803 [0.794, 0.812] | 0.491 | 0.496 |
| Male | 4,743 | 0.806 [0.793, 0.819] | 0.501 | 0.497 |
| Age 29–39 | 367 | 0.855 [0.801, 0.905] | 0.210 | 0.227 |
| Age 40–49 | 3,949 | 0.830 [0.817, 0.843] | 0.374 | 0.382 |
| Age 50–59 | 6,947 | 0.774 [0.762, 0.785] | 0.516 | 0.515 |
| Age 60–65 | 2,452 | 0.699 [0.678, 0.722] | 0.670 | 0.668 |

- **Ranking by age:** the model ranks noticeably worse at ages 60–65.
- **Agreement in the large:** mean predicted tracks prevalence in every group.
- **Gender:** the two gender groups perform similarly.

This is a Subgroup Performance Analysis. It describes differences and is not a fairness certification.

### Known limitations and potential bias
- **Provenance:** the population, the collection period and the label definition are not documented in detail.
- **Self-reported inputs:** lifestyle inputs are self-reported, and cholesterol and glucose are coarse three-level categories.
- **Confounded patterns:** smokers and drinkers have slightly *lower* label rates in this data (smoking is far more common in gender code 2). The model therefore gives "Smoking: Yes" and "Alcohol: Yes" small negative contributions. "Glucose: well above normal" also receives a lower contribution than "above normal". These are learned dataset artefacts, not health effects.
- **Gender coding:** female = 1 is inferred from height, because the source does not label the codes.
- **Narrow coverage:** the data covers ages 29–65 only, and gender groups are imbalanced (65% code 1).
- **Correlated inputs:** systolic and diastolic blood pressure are correlated (r = 0.73), so explanations can split credit between them unintuitively.
- **Step-wise model:** the selected tree model reacts in steps and is not monotone in its main inputs (see Reliability and robustness). Small input changes, or rounding, can change the estimate noticeably.

### Non-clinical status
No clinical validation, regulatory review or prospective evaluation has been performed. Display bands (<30%, 30–60%, ≥60%) and the 0.50 classification threshold are prototype devices, not clinical thresholds. The guidance shown alongside the estimate comes from fixed informational rules. It is not medical advice.
