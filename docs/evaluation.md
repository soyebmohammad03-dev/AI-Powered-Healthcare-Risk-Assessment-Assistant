# Evaluation report

Every number here is produced by four commands:

| Command | Writes |
|---|---|
| `python -m src.train_models` | `artifacts/metrics.json`, plus `artifacts/oof_predictions.npz` (not committed) |
| `python -m src.analysis` | `artifacts/analysis.json` |
| `python -m src.reliability` | `artifacts/reliability.json` |
| `python -m src.shift_analysis` | `artifacts/shift_analysis.json` |

Settings: seed 42; 54,858 training records and 13,715 held-out test records.

## 1. Methodological correction
The previous version (commit `da15801`) selected Logistic Regression with two margins: a less interpretable model had to gain at least 0.02 CV ROC-AUC, and a calibrator had to lower CV Brier by at least 0.001. **Those margins were set after the first comparison had been seen.** They are no longer used.

This phase declares a selection protocol (section 3) that contains no tunable numbers. It was written into `src/train_models.py` before the repeated-CV comparison was run. It is not blind: the author had seen the earlier single 5-fold results. It does, however, remove every quantity that could be adjusted to favour a particular model. Under this protocol, the evidence selects **XGBoost with raw probabilities**.

## 2. Repeated cross-validation (training split)
`RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)` gives 25 train/validation splits. All nine variants (3 models × raw / sigmoid / isotonic) are scored on the same splits.

Every learned step is fitted inside each split's training rows only. That includes the scaler, the encoder, the classifier, and the calibrator, which uses a further internal 3-fold split.

Each model in its selected variant, mean ± std [min, max] over the 25 splits:

| Model (variant) | ROC-AUC | PR-AUC | Brier | Log loss | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (isotonic) | 0.791 ± 0.004 [0.785, 0.801] | 0.766 ± 0.004 [0.760, 0.777] | 0.185 ± 0.002 [0.181, 0.188] | 0.554 ± 0.005 | 0.720 ± 0.005 | 0.730 ± 0.004 | 0.738 ± 0.005 | 0.704 ± 0.009 |
| Random Forest (isotonic) | 0.799 ± 0.004 [0.793, 0.807] | 0.776 ± 0.005 [0.768, 0.788] | 0.181 ± 0.002 [0.178, 0.184] | 0.544 ± 0.004 | 0.719 ± 0.008 | 0.733 ± 0.004 | 0.750 ± 0.011 | 0.691 ± 0.021 |
| **XGBoost (raw)** | **0.801 ± 0.004 [0.795, 0.808]** | **0.782 ± 0.005 [0.776, 0.795]** | **0.181 ± 0.002 [0.177, 0.183]** | **0.542 ± 0.004** | 0.720 ± 0.004 | 0.735 ± 0.004 | 0.753 ± 0.004 | 0.690 ± 0.007 |

F1, accuracy, precision and recall use the 0.50 threshold. The split-to-split spread (std ≈ 0.004 ROC-AUC) is about half the gap between Logistic Regression and the tree models. Random Forest has the least stable recall and precision across splits (std 0.021 and 0.011). The Model page shows the 25 per-split values as box plots.

## 3. Selection protocol and result
Every comparison uses the **corrected resampled t-test** (Nadeau & Bengio 2003; Bouckaert & Frank 2004) on paired per-split scores. Repeated-CV splits overlap, so a naive paired t-test would be over-confident. The variance is therefore inflated by (1/k + n_test/n_train), with k = 25, df = 24 and α = 0.05.

1. **Calibration (per model):** keep raw probabilities unless a calibrator lowers CV Brier significantly. If both calibrators do, take the lower mean.
2. **Explanation-quality gate:** a model is eligible only if an exact, additive SHAP explanation of its internal score exists and that score maps monotonically to the displayed probability. Logistic Regression uses exact linear SHAP; the tree models use exact TreeSHAP. All three pass.
3. **Model:** start from the most interpretable model (Logistic Regression → Random Forest → XGBoost). Move on only if the next model is significantly better on **both** ROC-AUC (higher) and Brier (lower).

Results:

| Step | Mean difference | p |
|---|---|---|
| LR: isotonic vs raw Brier | −0.0018 | 2.5e-10 → isotonic |
| LR: sigmoid vs raw Brier | +0.00002 | (worse) |
| RF: isotonic vs raw Brier | −0.0003 | 0.011 → isotonic |
| XGBoost: sigmoid / isotonic vs raw Brier | +0.0002 / +0.0001 | (worse) → raw |
| RF vs LR: ROC-AUC / Brier | +0.0084 / −0.0040 | 2.9e-10 / 7.9e-11 → switch to RF |
| XGBoost vs RF: ROC-AUC / Brier | +0.0012 / −0.0005 | 0.0087 / 0.010 → switch to XGBoost |

**Final model: XGBoost, raw probabilities.**

Reading the result:
- **Small but consistent.** The advantages are small in absolute terms (about 0.01 ROC-AUC over Logistic Regression, and 0.001 over Random Forest), but consistent across splits.
- **Significance is not importance.** With 55,000 training rows, small differences reach significance. The protocol asks for consistent evidence, not for practical importance.
- **Known costs, not used to choose.** The costs of this choice are measured in section 10 (less smooth responses, less stable explanations). They were measured after selection and were not used to override it.

## 4. Held-out test set and bootstrap intervals
The test set is scored once per model, after selection. Nothing (model, calibrator, threshold, detector) is fitted or chosen on it. Until Phase 8 the novelty detector's method choice was scored on test records; Phase 9 moved it to training rows (section 10).

Percentile bootstrap: 1,000 paired resamples of the 13,715 test rows (seed 42); 95% interval = 2.5th–97.5th percentile.

| Model | ROC-AUC | PR-AUC | F1 (0.50) | Brier | ECE |
|---|---|---|---|---|---|
| **XGBoost (raw, final)** | 0.804 [0.797, 0.812] | 0.785 [0.774, 0.796] | 0.722 [0.714, 0.731] | 0.180 [0.176, 0.183] | 0.008 [0.007, 0.017] |
| Random Forest (isotonic) | 0.802 [0.795, 0.810] | 0.776 [0.766, 0.786] | 0.725 [0.717, 0.733] | 0.180 [0.177, 0.184] | 0.009 [0.007, 0.017] |
| Logistic Regression (isotonic) | 0.793 [0.786, 0.801] | 0.770 [0.759, 0.781] | 0.725 [0.717, 0.733] | 0.184 [0.181, 0.188] | 0.005 [0.006, 0.016] |

Paired differences from the final model (other − XGBoost):

| Model | ROC-AUC | PR-AUC | F1 (0.50) | Brier |
|---|---|---|---|---|
| Logistic Regression | −0.011 [−0.014, −0.009] | −0.015 [−0.021, −0.009] | +0.003 [−0.002, +0.007] | +0.005 [+0.003, +0.006] |
| Random Forest | −0.002 [−0.004, −0.001] | −0.009 [−0.013, −0.005] | +0.003 [−0.001, +0.006] | +0.001 [+0.000, +0.001] |

- **ECE is biased under resampling.** ECE is a binned statistic, and resampling noise adds to every bin's gap. Its bootstrap interval can therefore lie *above* the point estimate, as it does for Logistic Regression. Read it as variability, not as a bracket.
- **What the intervals mean.** They describe uncertainty around evaluation metrics on the available sample. They are not clinical uncertainty, and not confidence that the model is correct.

## 5. Calibration deep dive (leakage-safe)
The source is the repeated-CV out-of-fold (OOF) probabilities of the training split. Each calibrator is fitted inside the outer training fold, on 3-fold internal OOF scores, so neither the model nor the calibrator ever saw the row it predicts.

Statistics are computed per repeat, with each repeat predicting every row once, and summarised as mean ± std over the 5 repeats.
- **ECE:** count-weighted mean |observed − predicted| over 10 equal-width bins.
- **Calibration slope:** logistic regression of the outcome on logit(p). A value of 1 is ideal; below 1, predictions are too extreme.
- **Calibration intercept:** calibration-in-the-large, with the slope fixed at 1. A value of 0 is ideal.
- **Clipping:** probabilities are clipped to [1e-6, 1 − 1e-6] before the logit, because isotonic calibration can output exactly 0 or 1.

| Variant (OOF, training split) | Brier | Log loss | ECE | Slope | Intercept |
|---|---|---|---|---|---|
| LR raw | 0.1870 | 0.5610 | 0.0338 | 0.998 | −0.000 |
| LR sigmoid | 0.1870 | 0.5610 | 0.0338 | 1.001 | −0.000 |
| LR isotonic | 0.1852 | 0.5534 | 0.0019 | 0.991 | −0.000 |
| XGBoost raw (final) | 0.1807 | 0.5416 | 0.0040 | 0.993 | 0.000 |
| XGBoost sigmoid | 0.1808 | 0.5421 | 0.0113 | 1.007 | 0.000 |
| XGBoost isotonic | 0.1808 | 0.5422 | 0.0024 | 0.999 | −0.000 |

Standard deviations across repeats are ≤ 0.001 for every entry.

Findings:
- **Raw Logistic Regression's miscalibration is S-shaped.** Its slope is ≈ 1 and intercept ≈ 0, yet ECE is 0.034. It over-estimates around 25–35% and in the top bin, and under-estimates around 55–75%.
- **Why sigmoid calibration cannot help.** A logistic recalibration (sigmoid / Platt) cannot represent an S-shape, so sigmoid calibration changes nothing. Slope and intercept alone would have hidden this; the reliability curve and ECE show it.
- **Isotonic calibration fixes it.** It lowers Logistic Regression's ECE to 0.002 and Brier in every repeat.
- **XGBoost needs no calibrator.** Its raw probabilities already agree closely with observed frequencies (ECE 0.004). Isotonic calibration lowers ECE slightly but not Brier, so the protocol keeps raw probabilities.

Held-out test set (final evaluation, reported only):

| | Brier | Log loss | ECE | Slope | Intercept |
|---|---|---|---|---|---|
| LR raw / isotonic | 0.186 / 0.184 | 0.558 / 0.553 | 0.034 / 0.005 | 1.02 / 1.00 | −0.003 / −0.007 |
| XGBoost raw (final) | 0.180 | 0.538 | 0.008 | 1.02 | −0.009 |

**Reliability diagram binning:** 10 equal-width bins on OOF repeat 1, so bin counts are real row counts. Bins with fewer than 100 rows are drawn hollow and are not interpreted; the highest LR-isotonic bin is one of them. A lower panel shows rows per bin.

"Calibration" here means agreement between predicted probabilities and observed outcome frequencies in this dataset. It is not a claim about clinical accuracy.

## 6. Threshold analysis 2.0 and decision trade-off
The thresholds run from 0.05 to 0.95 in steps of 0.01. Each threshold reports TP, FP, TN and FN, sensitivity (recall), specificity, precision, F1, FPR, FNR and accuracy.

They are computed on two sources:
- the final model's **OOF predictions of the training split**, where exploration is legitimate;
- the **test set**, which is descriptive only.

No threshold is selected from either source.

| Threshold (test) | Precision | Recall | Specificity | F1 | Accuracy | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.30 | 0.627 | 0.886 | 0.483 | 0.734 | 0.682 | 3,584 | 774 |
| 0.40 | 0.700 | 0.787 | 0.671 | 0.741 | 0.728 | 2,283 | 1,448 |
| 0.50 | 0.755 | 0.692 | 0.781 | 0.722 | 0.737 | 1,519 | 2,093 |
| 0.60 | 0.803 | 0.590 | 0.858 | 0.680 | 0.726 | 981 | 2,781 |
| 0.70 | 0.825 | 0.521 | 0.892 | 0.639 | 0.709 | 748 | 3,248 |
| 0.80 | 0.847 | 0.408 | 0.928 | 0.551 | 0.671 | 499 | 4,017 |

The OOF values agree closely (at 0.50: precision 0.753, recall 0.691, specificity 0.778).

**Decision trade-off.**
- **Error counts:** the Model page plots false positives and false negatives against the threshold, with confusion matrices at 0.30, 0.50 and 0.70.
- **Net-benefit curve:** net benefit = TP/n − FP/n × t/(1 − t), compared with "flag everyone" and "flag no one". On the test set, the model's net benefit exceeds "flag everyone" at every threshold in the grid. The margin is negligible below about t = 0.1 and grows with t (0.326 vs 0.278 at t = 0.3).
- **What the curve can and cannot tell you:** it is exploratory and specific to this dataset's roughly 50% prevalence. It does not establish clinical utility, and no threshold is called optimal.

**Three separate concepts:** the *model probability*, the *display band* (prototype presentation categories at 30% and 60%) and the *classification threshold* (0.50 for the class output).

## 7. Subgroup Performance Analysis (final model, test set)
95% intervals come from a within-group percentile bootstrap (500 resamples, seed 42). Recall, precision and F1 use the 0.50 threshold.

| Group | N | Prevalence | Mean predicted | ROC-AUC | PR-AUC | Recall | Precision | F1 | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Female (code 1) | 8,972 | 0.491 | 0.496 | 0.803 [0.794, 0.812] | 0.780 [0.767, 0.794] | 0.692 [0.679, 0.706] | 0.747 | 0.719 | 0.180 [0.176, 0.184] |
| Male (code 2) | 4,743 | 0.501 | 0.497 | 0.806 [0.793, 0.819] | 0.794 [0.777, 0.813] | 0.690 [0.670, 0.708] | 0.771 | 0.728 | 0.178 [0.172, 0.184] |
| Age 29–39 | 367 | 0.210 | 0.227 | 0.855 [0.801, 0.905] | 0.655 [0.553, 0.760] | 0.519 [0.410, 0.646] | 0.678 | 0.588 | 0.112 [0.089, 0.135] |
| Age 40–49 | 3,949 | 0.374 | 0.382 | 0.830 [0.817, 0.843] | 0.759 [0.736, 0.783] | 0.610 [0.586, 0.635] | 0.779 | 0.684 | 0.154 [0.147, 0.160] |
| Age 50–59 | 6,947 | 0.516 | 0.515 | 0.774 [0.762, 0.785] | 0.783 [0.768, 0.798] | 0.654 [0.637, 0.668] | 0.759 | 0.703 | 0.192 [0.187, 0.197] |
| Age 60–65 | 2,452 | 0.670 | 0.668 | 0.699 [0.678, 0.722] | 0.805 [0.783, 0.827] | 0.855 [0.837, 0.872] | 0.738 | 0.792 | 0.197 [0.189, 0.205] |

Any group with fewer than 30 records of either class (or with only one class) would show "Metric unavailable for this subgroup" instead of numbers. No group here is that small.

Findings:
- **Ranking falls with age.** ROC-AUC drops from 0.83 (ages 40–49) to 0.70 (ages 60–65), and the intervals do not overlap. This is partly because age, a strong signal, varies little within a band, but it is a real difference.
- **PR-AUC and recall follow prevalence.** They rise with each group's prevalence.
- **Mean predicted tracks prevalence everywhere.** It is within 0.02 of prevalence in every group.
- **Gender groups are similar.**

These are descriptive differences. No model was tuned on them, and they are not a fairness certification.

## 8. Explainability analyses (final model)
- **Global SHAP:** TreeSHAP, mean |SHAP| in log-odds, on 5,000 random cleaned records. Systolic BP has 49% of the attribution, age 16%, cholesterol 13%, BMI 7%, diastolic 6% and physical activity 4%.
- **Directions:** smoking "Yes" (−0.15) and alcohol "Yes" (−0.22) get negative mean contributions, and so does glucose "well above normal" (−0.17). These are confounded dataset patterns, not health effects.
- **Permutation importance:** the drop in test ROC-AUC is 0.182 for systolic BP, 0.039 for age, 0.031 for cholesterol and 0.008 for weight. Everything else is ≤ 0.003.
- **Partial dependence and ICE:** 2,000 test records, all three candidate models. Logistic Regression rises smoothly with systolic BP; the tree models step.
- **Interactions:** XGBoost SHAP interaction values account for about 30% of XGBoost's attribution, led by age × systolic BP. Unlike Logistic Regression, XGBoost's per-input contribution depends on the other inputs.

## 9. What-if analysis
`src/what_if.py` rebuilds every scenario through `PatientInput`, so the same ranges and cross-field rules apply. With XGBoost, response curves are step-shaped and can dip. They describe model sensitivity, not predicted medical outcomes.

## 10. Reliability and robustness
### Input conformity (novelty detection)
**What is evaluated:**
- **Space:** the model's preprocessed feature space (BMI derived, numerics standard-scaled, categoricals one-hot), with a preprocessor refitted on the detector's own fitting rows.
- **Fitting:** 80% of the training split (43,886 rows).
- **Threshold:** the 99th percentile of scores on the other 20% of the training split (10,972 reference rows). The flag rate on reference data is therefore 1% by construction.
- **The user's input is never used for fitting.**

**Two methods were compared under a pre-declared rule:**
- **Evaluation set:** the 10,972 reference rows versus 10,862 "atypical combinations", which are reference values recombined at random and kept only if valid. Only training rows are used.
- **Rule:** the higher ROC-AUC wins; a tie goes to the simpler method.
- **Phase 9 correction:** this comparison used test records until Phase 8. The choice and the threshold are unchanged by the correction (the earlier AUCs were 0.648 vs 0.557).

| Method | Flag rate, reference | Flag rate, atypical | AUC reference vs atypical | Flag rate, test (reported afterwards) |
|---|---|---|---|---|
| **Mahalanobis distance** on the 4 scaled continuous features (chosen) | 1.0% | 3.6% | 0.653 | 1.04% |
| Isolation Forest on all 14 columns | 1.0% | 0.7% | 0.560 | 0.91% |

- **Why classical covariance.** The robust (MCD) covariance estimate was tried first and discarded: it degenerates on this data, where 40% of systolic readings are exactly 120 mmHg and 51% of diastolic readings exactly 80.
- **Generalisation.** The test-set flag rate (1.04%) matches the reference rate, so the threshold generalises.
- **What it catches.** Neither method separates atypical combinations strongly. The check mainly catches extreme values and is a statistical caution, not a medical abnormality detector.
- **In the app.** An unusual input still receives an estimate, with a caution and the values outside the central 99% of the reference data.

### Prediction stability under small perturbations
**Profiles:** 153 baseline profiles, made up of the 3 demo inputs and 150 random test records.

**Perturbations:** age, height, weight, systolic and diastolic, each changed by ±1%, ±2% or ±5% alone, plus 4 random joint perturbations per size. That gives 6,348 valid perturbed versions; invalid ones are dropped, never repaired. Categorical inputs have no small change and are excluded.

|Δprobability| in percentage points (median / 95th percentile / maximum):

| Model | ±1% | ±2% | ±5% | Unchanged at ±1% |
|---|---|---|---|---|
| Logistic Regression (isotonic) | 0.0 / 2.7 / 6.0 | 0.0 / 5.9 / 9.0 | 1.7 / 11.3 / 19.6 | 66% |
| Random Forest (isotonic) | 0.0 / 4.7 / 15.9 | 0.0 / 6.6 / 24.7 | 1.0 / 11.2 / 31.5 | 69% |
| **XGBoost (raw, final)** | **0.7 / 11.4 / 37.4** | **1.1 / 11.4 / 43.9** | **2.2 / 14.4 / 43.9** | **14%** |

**Finding (a weakness of the selected model).** XGBoost responds less smoothly: a 1% change can move its estimate by tens of percentage points.
- **Why.** Its trees split near the round values at which blood pressure is recorded (120/80), so a small change crosses a split.
- **Logistic Regression's zeros.** Its "unchanged" share comes from the flat steps of isotonic calibration.

### Explanation stability
Method:
- For each perturbed version, SHAP is recomputed and compared with its baseline's SHAP.
- **Spearman ρ:** rank correlation of |SHAP| over the 10 model features.
- **Top-5 overlap:** |top-5 ∩ top-5′| / 5.
- **L1 distance:** Σ|φ − φ′| / Σ|φ|.

| Model | Change | Spearman ρ (mean) | Top-5 overlap (mean) | Top-5 unchanged | Same top feature | L1 (median) |
|---|---|---|---|---|---|---|
| Logistic Regression | ±1% / ±5% | 0.988 / 0.959 | 0.990 / 0.951 | 95% / 76% | 97% / 88% | 0.02 / 0.08 |
| Random Forest | ±1% / ±5% | 0.987 / 0.961 | 0.984 / 0.958 | 92% / 79% | 98% / 92% | 0.02 / 0.08 |
| **XGBoost (final)** | ±1% / ±5% | 0.955 / 0.912 | 0.948 / 0.911 | **74% / 57%** | 96% / 89% | 0.05 / 0.15 |

**Finding.** The top feature is usually preserved by every model. XGBoost's top-5 set, however, changed in about a quarter of ±1% perturbations and 43% of ±5% perturbations. Its explanation rankings are noticeably more sensitive to small input changes than those of Logistic Regression or Random Forest.

No universal "stable enough" threshold exists. The Explain page reports this descriptively for each assessment.

### SHAP and model-output reconciliation
On 1,000 test records:
- **Logistic Regression:** |base + ΣSHAP − score| ≤ 9e-16.
- **Random Forest:** ≤ 8e-15.
- **XGBoost:** ≤ 3e-6, because XGBoost computes in float32. The documented tolerance is 1e-5.
- **Probability:** the displayed probability reproduced from SHAP matches the pipeline within 3e-7.

shap 0.52 reports XGBoost's expected value about 6e-4 log-odds away from the model's base margin. The offset is constant across rows, so the explainer re-anchors the base value. If the offset were ever not constant, the explainer would refuse to run.

Regression tests guard all of this.

### Controlled sensitivity (monotonicity)
One input was varied over its accepted range with every other input fixed. Invalid grid points were dropped. This was done for 23 profiles: the 3 demo inputs and 20 test records.

| Input | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Systolic BP | 23 non-decreasing | 7 non-decreasing, 16 non-monotone | 23 non-monotone (largest single-step drop 14.4 pp) |
| Diastolic BP | 23 non-decreasing | 8 / 15 | 23 non-monotone (18.5 pp) |
| Age | 23 non-decreasing | 11 / 12 | 23 non-monotone (8.2 pp) |
| Weight | 23 non-decreasing | 5 / 18 | 23 non-monotone (5.1 pp) |

These describe how the model's estimated probability changed under controlled input variation. Nothing is imposed, and no causal claim is made. XGBoost's overall trend rises with systolic BP, but it contains local reversals.

### Model disagreement (prediction variation across candidate models)
On the 13,715 test records, the spread is max − min of the three candidates' probabilities:
- **Spread:** median 5.1 pp, 75th percentile 8.4 pp, 95th percentile 17.3 pp, maximum 47.5 pp.
- **Large spreads:** 18% of records have a spread of at least 10 pp.
- **Class agreement:** all three models give the same class at 0.50 for 92.3% of records.
- **Correlation:** the probabilities correlate at 0.96 (LR vs trees) and 0.99 (RF vs XGBoost).
- **Demo inputs:** A 14.4% / 7.1% / 5.3% (spread 9.2 pp); B 46.0% / 43.0% / 40.7%; C 86.1% / 86.5% / 86.8%.

Models with nearly identical overall metrics can therefore differ substantially for one person. This is model disagreement, a consequence of model assumptions, and not a calibrated measure of uncertainty.

### Controlled distribution-shift experiment (synthetic)
`src/shift_analysis.py` writes a separate artifact flagged `"synthetic": true`. It is not a validation on a real population.

**Population-mix shift.** Test rows are resampled with weights exp(β·z) for one feature, with β ∈ {±0.5, ±1}. Each record keeps its own label.

| Shift (XGBoost) | Feature mean | ROC-AUC | Brier | Predicted − observed |
|---|---|---|---|---|
| None | age 53.3, systolic 126.6, BMI 27.5 | 0.804 | 0.180 | +0.002 |
| Older (β = +1) | age 58.5 | 0.747 | 0.197 | −0.007 |
| Younger (β = −1) | age 46.6 | 0.842 | 0.151 | +0.009 |
| Higher systolic (β = +1) | 161.2 mmHg | 0.749 | 0.140 | −0.012 |
| Higher BMI (β = +1) | 39.0 | 0.747 | 0.193 | +0.002 |

Ranking quality depends strongly on the population mix: it is lower in older or higher-BP populations, consistent with section 7. Agreement in the large (mean predicted vs observed) stays within about 0.01 for XGBoost. Logistic Regression drifts to +0.031 under the high-BMI shift.

**Measurement offset.** Inputs are recorded systematically differently; labels are unchanged.
- **Blood pressure +10 mmHg:** XGBoost's mean prediction rises from 0.496 to 0.638 against an observed 0.495, while ROC-AUC barely moves (0.785).
- **Blood pressure −10 mmHg:** the mean prediction falls to 0.402.
- **Blood pressure +5 mmHg:** XGBoost hardly reacts (0.499), because most readings stay on the same side of its split points. Logistic Regression responds linearly (0.564).
- **Weight ±5%:** shifts the estimate by about 1 pp.
- **Age +3 years:** pushes 1,314 records past the 65-year limit. They are dropped, which itself changes the population; read that row with care.

A systematic measurement bias therefore barely affects ranking but directly biases every estimate.
