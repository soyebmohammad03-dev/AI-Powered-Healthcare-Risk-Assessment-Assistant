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
Logistic Regression, Random Forest and XGBoost. Each was tested in three versions: raw probabilities, sigmoid-calibrated and isotonic-calibrated. All were compared with 5-fold stratified cross-validation (CV) on the training split.

### Final model
**Logistic Regression with isotonic calibration** (`CalibratedClassifierCV`, 3-fold, `ensemble=False`).

It was chosen by a documented framework that uses CV results only:
- **Calibration:** a calibrated version is adopted only if it lowers mean CV Brier score by at least 0.001 and lowers it in every fold.
- **Model:** a less interpretable model is adopted only if it gains at least 0.02 CV ROC-AUC without worse calibration.

The tree models were 0.009–0.010 CV ROC-AUC better, which is below that margin. The margins are judgement calls, fixed after the first comparison was known.

### Evaluation (held-out test set, selected model)
| Metric | Value | 95% bootstrap interval |
|---|---|---|
| ROC-AUC | 0.793 | 0.786–0.801 |
| PR-AUC | 0.770 | 0.759–0.781 |
| F1 at 0.50 | 0.725 | 0.717–0.733 |
| Brier score | 0.184 | 0.181–0.188 |
| Log loss | 0.553 | |
| Accuracy / precision / recall at 0.50 | 0.732 / 0.735 / 0.715 | |

CV ROC-AUC was 0.791 ± 0.004. XGBoost's test ROC-AUC is higher by 0.011 (paired 95% interval 0.009–0.014), and its Brier score is lower by 0.005. These are real but small differences.

### Calibration
Raw logistic regression was systematically miscalibrated. It over-estimated in the 25–35% range and in the top bin, and under-estimated in the 55–75% range. Isotonic calibration lowered CV Brier from 0.1870 to 0.1852 and log loss from 0.561 to 0.555, without changing ROC-AUC. Its step function slightly lowered PR-AUC (0.771 to 0.767), and calibrated probabilities rarely exceed about 0.9. Sigmoid calibration made no difference. Calibration here means agreement with this dataset's labels, not clinical validity.

### Explainability
- **SHAP:** exact SHAP values (`LinearExplainer`) for the logistic-regression score, grouped into readable features. The displayed probability is that score passed through the monotone calibration, so directions are preserved but sizes are not additive in percentage points.
- **Other analyses:** permutation importance, partial dependence and individual conditional expectation (ICE) curves, XGBoost interaction values, and a guarded what-if analysis.
- **Top features:** systolic blood pressure, age and cholesterol.

### Subgroup performance (test set)
| Group | n | ROC-AUC | Observed rate | Mean predicted |
|---|---|---|---|---|
| Female | 8,972 | 0.791 | 0.491 | 0.495 |
| Male | 4,743 | 0.797 | 0.501 | 0.498 |
| Age 29–39 | 367 | 0.803 | 0.210 | 0.245 |
| Age 40–49 | 3,949 | 0.816 | 0.374 | 0.360 |
| Age 50–59 | 6,947 | 0.767 | 0.516 | 0.529 |
| Age 60–65 | 2,452 | 0.692 | 0.670 | 0.659 |

- **Ranking by age:** the model ranks noticeably worse at ages 60–65.
- **Recall at 0.50 by age:** ranges from 0.44 (ages 29–39) to 0.86 (ages 60–65), following the base rates.
- **Youngest group:** slightly over-estimated.
- **Gender:** the two gender groups perform similarly.

These results describe differences. They do not establish fairness.

### Known limitations and potential bias
- **Provenance:** the population, the collection period and the label definition are not documented in detail.
- **Self-reported inputs:** lifestyle inputs are self-reported, and cholesterol and glucose are coarse three-level categories.
- **Confounded patterns:** smokers and drinkers have slightly *lower* label rates in this data (smoking is far more common in gender code 2). The model therefore gives "Smoking: Yes" and "Alcohol: Yes" small negative contributions. "Glucose: well above normal" also receives a lower contribution than "above normal". These are learned dataset artefacts, not health effects.
- **Gender coding:** female = 1 is inferred from height, because the source does not label the codes.
- **Narrow coverage:** the data covers ages 29–65 only, and gender groups are imbalanced (65% code 1).
- **Correlated inputs:** systolic and diastolic blood pressure are correlated (r = 0.73), so explanations can split credit between them unintuitively.

### Non-clinical status
No clinical validation, regulatory review or prospective evaluation has been performed. Display bands (<30%, 30–60%, ≥60%) and the 0.50 classification threshold are prototype devices, not clinical thresholds. The guidance shown alongside the estimate comes from fixed informational rules. It is not medical advice.
