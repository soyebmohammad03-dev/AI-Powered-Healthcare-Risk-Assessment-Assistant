# Evaluation report

Every number here is produced by `python -m src.train_models` (written to `artifacts/metrics.json`) and `python -m src.analysis` (written to `artifacts/analysis.json`). Settings: seed 42; 54,858 training records and 13,715 held-out test records.

## 1. Model comparison (5-fold stratified CV on the training split)
Each model was evaluated raw, with sigmoid (Platt) calibration and with isotonic calibration. Calibrators are `CalibratedClassifierCV(cv=3, ensemble=False)` placed *inside* the pipeline, so in every outer fold they are fitted on that fold's training rows only. A leakage test checks this: a memorising 1-nearest-neighbour model scores at chance on random labels.

The table shows each model in its selected variant (mean ± standard deviation across the 5 folds):

| Model (variant) | ROC-AUC | PR-AUC | Brier | Log loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (isotonic) | 0.791 ± 0.004 | 0.767 ± 0.004 | 0.185 ± 0.002 | 0.555 ± 0.005 | 0.730 | 0.739 | 0.702 | 0.720 |
| Random Forest (raw) | 0.800 ± 0.004 | 0.782 ± 0.003 | 0.181 ± 0.002 | 0.544 ± 0.004 | 0.733 | 0.764 | 0.667 | 0.712 |
| XGBoost (raw) | 0.801 ± 0.004 | 0.782 ± 0.004 | 0.181 ± 0.002 | 0.542 ± 0.004 | 0.735 | 0.753 | 0.691 | 0.721 |

Accuracy, precision, recall and F1 use a 0.50 threshold. The tree models are ahead of Logistic Regression in all five folds.

## 2. Selection framework
1. **Calibration per model:** the raw model is kept unless a calibrator lowers mean CV Brier by at least 0.001 *and* lowers it in every fold.
   - Logistic Regression → isotonic (−0.0018, every fold).
   - Random Forest → raw (best calibrator changed Brier by only −0.0003).
   - XGBoost → raw (best calibrator +0.0001).
2. **Model:** the search walks from the most to the least interpretable model (Logistic Regression → Random Forest → XGBoost). It switches to a less interpretable model only for a CV ROC-AUC gain of at least 0.02 without worse calibration.
   - Random Forest: +0.0087, so Logistic Regression is kept.
   - XGBoost: +0.0096, so Logistic Regression is kept.
3. The test set is never used to choose a model.

Both margins are judgement calls about practical relevance. They were fixed in the upgrade phase, after the Phase 1 comparison was already known. They are stated so they can be challenged. They are not statistically derived.

## 3. Calibration
Reliability on out-of-fold predictions (training split, 10 equal-width bins):

| Bin | Raw LR: mean predicted → observed | Isotonic LR: mean predicted → observed |
|---|---|---|
| 0.2–0.3 | 0.25 → 0.22 | 0.25 → 0.25 |
| 0.3–0.4 | 0.35 → 0.31 | 0.34 → 0.34 |
| 0.6–0.7 | 0.65 → 0.72 | 0.65 → 0.64 |
| 0.9–1.0 | 0.94 → 0.86 (n = 3,406) | 0.93 → 0.73 (n = 30, too few to interpret) |

Findings:
- Isotonic calibration lowered CV Brier from 0.1870 to 0.1852 and log loss from 0.561 to 0.555. ROC-AUC was unchanged (0.791).
- Its step function (75 distinct levels) lowered PR-AUC slightly, from 0.771 to 0.767.
- Calibrated probabilities rarely exceed about 0.9. Example Patient C moved from 97.9% (raw) to 86.1% (calibrated).
- Sigmoid calibration had no effect on Logistic Regression, which already uses a sigmoid.
- Calibration means agreement with this dataset's labels. It does not mean clinical validity.

## 4. Held-out test set and bootstrap intervals
Percentile bootstrap: 1,000 paired resamples of the 13,715 test rows (seed 42), 95% intervals.

| Model | ROC-AUC | PR-AUC | F1 (0.50) | Brier |
|---|---|---|---|---|
| Logistic Regression (isotonic) | 0.793 [0.786, 0.801] | 0.770 [0.759, 0.781] | 0.725 [0.717, 0.733] | 0.184 [0.181, 0.188] |
| Random Forest | 0.803 [0.795, 0.810] | 0.783 [0.773, 0.794] | 0.717 [0.708, 0.725] | 0.181 [0.177, 0.184] |
| XGBoost | 0.804 [0.797, 0.812] | 0.785 [0.774, 0.796] | 0.722 [0.714, 0.731] | 0.180 [0.176, 0.183] |

Paired differences from Logistic Regression:

| Model | ROC-AUC | PR-AUC | F1 (0.50) | Brier |
|---|---|---|---|---|
| XGBoost | +0.011 [+0.009, +0.014] | +0.015 [+0.009, +0.021] | −0.003 [−0.007, +0.002] | −0.005 [−0.006, −0.004] |
| Random Forest | +0.010 [+0.007, +0.012] | +0.014 [+0.009, +0.020] | −0.008 [−0.014, −0.004] | −0.004 [−0.005, −0.003] |

The tree models are reliably, but only slightly, better at ranking and probability quality. These intervals describe uncertainty from the evaluation sample. They do not describe clinical uncertainty.

## 5. Threshold analysis (selected model, test set)
| Threshold | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|
| 0.30 | 0.619 | 0.878 | 0.470 | 0.726 |
| 0.40 | 0.689 | 0.790 | 0.651 | 0.736 |
| 0.50 | 0.735 | 0.715 | 0.748 | 0.725 |
| 0.60 | 0.787 | 0.605 | 0.840 | 0.684 |
| 0.70 | 0.817 | 0.509 | 0.888 | 0.627 |
| 0.80 | 0.841 | 0.362 | 0.933 | 0.506 |

The full table (0.10–0.90) and a 0.01-step grid are in `analysis.json`.
- Raising the threshold trades recall for specificity.
- At 0.90 almost no record is flagged, because calibrated probabilities rarely exceed 0.9.
- No threshold is clinically optimal.
- The prototype display bands (<30%, 30–60%, ≥60%) are a separate presentation device.

## 6. Subgroup performance (selected model, test set)
| Group | n | ROC-AUC | Recall (0.50) | Brier | Observed rate | Mean predicted |
|---|---|---|---|---|---|---|
| Female (code 1) | 8,972 | 0.791 | 0.709 | 0.186 | 0.491 | 0.495 |
| Male (code 2) | 4,743 | 0.797 | 0.725 | 0.182 | 0.501 | 0.498 |
| Age 29–39 | 367 | 0.803 | 0.442 | 0.119 | 0.210 | 0.245 |
| Age 40–49 | 3,949 | 0.816 | 0.573 | 0.161 | 0.374 | 0.360 |
| Age 50–59 | 6,947 | 0.767 | 0.712 | 0.195 | 0.516 | 0.529 |
| Age 60–65 | 2,452 | 0.692 | 0.862 | 0.201 | 0.670 | 0.659 |

Age bins are [29, 40), [40, 50), [50, 60) and [60, 66). Groups with fewer than 30 records of either class would be reported without metrics; no group here was that small.

Findings:
- **Ranking by age:** ROC-AUC drops markedly at ages 60–65. This is partly because age, a strong signal, varies little within a narrow band, but it is a real difference.
- **Recall by age:** recall at 0.50 follows each group's base rate.
- **Youngest group:** slightly over-estimated (0.245 predicted vs 0.210 observed).
- **Gender:** the two gender groups are similar.

This analysis identifies differences. It does not establish fairness.

## 7. Explainability analyses
**Global SHAP** (selected model, share of mean |SHAP| across all cleaned records):

| Feature | Share | Direction |
|---|---|---|
| Systolic BP | 45.0% | higher → higher estimate |
| Age | 17.2% | higher → higher estimate |
| Cholesterol | 15.6% | well above normal +0.89 |
| BMI | 7.1% | |
| Diastolic BP | 4.5% | |
| Physical activity | 4.4% | |
| Glucose | 2.5% | well above normal −0.27, counter-intuitive |
| Smoking | 1.5% | Yes −0.14, confounded |
| Alcohol | 1.4% | Yes −0.21, confounded |
| Gender | 0.8% | |

**Permutation importance** (test set, ROC-AUC drop, 10 repeats):

| Input | ROC-AUC drop |
|---|---|
| Systolic BP | 0.173 |
| Age | 0.029 |
| Cholesterol | 0.026 |
| Weight | 0.005 |
| Others | ≤ 0.002 |
| Gender | ≈ 0 |

The two methods agree on the ranking at the top. Permutation importance works on the 11 raw inputs, while SHAP works on the 10 model features.

**Partial dependence and ICE** (2,000 test records; grid points that would create an impossible record are skipped):
- **Logistic Regression:** the estimate rises smoothly with systolic BP (about 0.19 at 100 mmHg, 0.58 at 130, 0.85 at 170).
- **XGBoost:** shows a plateau and then a step between 125 and 140 mmHg.
- **Caveat:** systolic and diastolic BP are correlated (r = 0.73), so varying one alone partly creates unusual combinations.

**Interactions** (XGBoost SHAP interaction values, 1,000 test records):
- Interactions account for about 30% of XGBoost's attribution.
- Strongest pairs: age × systolic BP (0.162), systolic BP × cholesterol (0.092), BMI × systolic BP (0.072).
- The selected logistic regression has no interaction terms.
- These describe the model, not biology.

## 8. What-if analysis
`src/what_if.py` rebuilds every scenario through `PatientInput`, so the same ranges and cross-field rules apply (systolic must be above diastolic; BMI must be plausible). Age, gender and height cannot be changed.

For Example Patient B, raising systolic BP from 130 to 150 mmHg moves the estimate from 46.0% to 77.8% (+31.9 percentage points), with everything else held fixed.

Single-change effects need not add up to the combined change, because the calibrated probability is a step function. For example, "stop smoking" can *raise* the estimate (about +6 percentage points for Patient B), which is the confounded dataset pattern described in section 7.

These results describe model sensitivity. They are not predicted medical outcomes.
