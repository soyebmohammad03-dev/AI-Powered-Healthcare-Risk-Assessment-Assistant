# AI-Powered Healthcare Risk Assessment Assistant

**An Explainable AI-Based Human-Centered Healthcare Decision Support System.** This is an educational research prototype for the B.Tech course *Design of Artificial Intelligence Products*.

> **Medical disclaimer.** This prototype does **not** diagnose cardiovascular disease, prescribe treatment, or replace a healthcare professional, and it is not medical advice. Its outputs are model-based estimates from one public research dataset and are **not clinically validated**. Calibrated probabilities agree with that dataset's labels; they are not clinical risks.

The system estimates a probability, shows how trustworthy and how explainable that estimate is, and lets you inspect the model critically. Its components:
- **An evidence-selected model.** A pre-declared protocol chooses among three candidate models on 25 repeated CV splits, and the result is accepted even when it is not the most interpretable model.
- **Exact explanations.** Each assessment is broken down into per-input contributions using SHAP.
- **What-if analysis.** You can change inputs and see how the model estimate responds.
- **Model analytics.** Calibration, threshold, decision trade-off, subgroup and bootstrap-uncertainty analyses of the models.
- **Reliability signals.** For each assessment: an input-conformity check, explanation stability, and disagreement between candidate models. Offline: prediction and explanation stability, monotonicity checks and synthetic distribution-shift experiments.
- **Data Quality Lab and model card.** What was removed from the data and why, plus a formal statement of intended use and limitations.

## Application

```bash
streamlit run app.py
```

The app has five areas in a top navigation bar. Each area is a page script in `ui/`; all model logic lives in `src/`.

| Area | What it is for |
|---|---|
| **Assess** | Grouped input form (Person, Vitals, Labs, Lifestyle) with three synthetic examples. **Run Assessment** shows the model estimate as the headline, the prototype estimate band, and secondary information (model, calibration status, input conformity, reference baseline, class at 0.50). It also shows a short "How the model arrived here" preview, general guidance and an input summary. Unusual inputs get a caution; they are never blocked. |
| **Explain** | The full SHAP explanation of the current assessment. It shows the reference baseline, this estimate and the difference between them; a diverging contribution chart with ▲/▼ markers, so direction does not rely on colour; and a note that model contribution ≠ medical causation. The **Assessment reliability** panel shows probability, input conformity, explanation stability, model disagreement and calibration status as separate signals, with no combined score. |
| **Explore** | What-if (model sensitivity) analysis. Change systolic or diastolic BP, weight, cholesterol, glucose, smoking, activity or alcohol, and compare baseline, scenario and change in percentage points. It also shows each change on its own and the model estimate across one input. Impossible combinations are rejected, and the stored assessment is never modified. |
| **Model** | Tabs for Overview (repeated-CV comparison, stability box plots, selection protocol), Calibration (reliability diagram with bin counts, ECE, slope and intercept), ROC & PR, Thresholds (0.05–0.95, decision trade-off, net benefit), Explainability, Subgroups (with intervals), Uncertainty (bootstrap) and Robustness (novelty, stability, monotonicity, disagreement, synthetic shift). |
| **Methodology** | Pipeline diagram, Data Quality Lab (exclusions with examples, ranges, distributions, correlations), model card, validation strategy and limitations. |

Design notes:
- Theming uses Streamlit's native light and dark themes (`.streamlit/config.toml`), the Inter font, and semantic colours: teal = moved the estimate lower or lower band, amber = moderate, red/orange = higher. Every coloured item also has a text label or ▲/▼ marker.
- The model and explainer are cached once per server (`st.cache_resource`).
- All global analyses are precomputed into `artifacts/*.json`. An assessment costs one prediction, one exact SHAP call, a novelty check, about 30 SHAP calls for local stability (cached) and three candidate predictions.

## Dataset

[Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset), S. Ulianova, Kaggle. The raw file is semicolon-separated with 70,000 rows and 13 columns. `src/data_loader.py` downloads it from Kaggle's public API (no account needed) and checks a SHA-256 hash. The data is not committed to this repository.

**Why this dataset:**
- It has 70,000 records, enough for credible cross-validation, calibration, bootstrap and subgroup analysis.
- Its 11 inputs are ones a non-specialist can supply.

**Cleaning:** documented data-quality rules remove 1,427 rows (24 duplicates and implausible measurements such as a blood pressure of 16,020), leaving **68,573** records (34,646 absent, 33,927 present). Details are in `docs/methodology.md` and in the app's Data Quality Lab.

**Features:**

| Field | Description |
|---|---|
| Age | in years; converted from days |
| Gender | 1 = female, inferred from height |
| Height, weight | combined into **BMI**, the only engineered feature |
| Blood pressure | systolic and diastolic |
| Cholesterol, glucose | normal / above normal / well above normal |
| Smoking, alcohol, physical activity | self-reported |
| `id` | never used |

## Machine-learning methodology

Full details are in [docs/methodology.md](docs/methodology.md), and every measured number is in [docs/evaluation.md](docs/evaluation.md).

- **Strategy:** a stratified 80/20 split (54,858 train / 13,715 test). All comparisons use **5-fold CV repeated 5 times** (25 splits) on the training split. Preprocessing and calibration are fitted inside each split, and each metric is reported as mean, std, min and max. Every training row receives out-of-fold predictions, with fold and repeat recorded.
- **Models:** Logistic Regression, Random Forest and XGBoost, each tested **raw, sigmoid-calibrated and isotonic-calibrated**, which makes 9 variants.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, Brier score, ECE, and calibration slope and intercept.
- **Selection protocol** (declared before the repeated-CV run; no tunable margins):
  - every comparison is a corrected resampled t-test on the same 25 splits at α = 0.05;
  - keep raw probabilities unless a calibrator lowers CV Brier significantly;
  - a model needs an exact SHAP explanation to be eligible;
  - starting from the most interpretable model, move on only for a significant gain in **both** ROC-AUC and Brier.

  The previous version's margins (0.02 ROC-AUC, 0.001 Brier) were set after seeing results and are no longer used.
- **Selected model: XGBoost, raw probabilities.** The test set was used only after selection.

| Model (selected variant) | CV ROC-AUC (25 splits) | Test ROC-AUC [95% CI] | Test PR-AUC | Test Brier [95% CI] | Test F1 at 0.50 |
|---|---|---|---|---|---|
| Logistic Regression (isotonic) | 0.791 ± 0.004 | 0.793 [0.786, 0.801] | 0.770 | 0.184 [0.181, 0.188] | 0.725 |
| Random Forest (isotonic) | 0.799 ± 0.004 | 0.802 [0.795, 0.810] | 0.776 | 0.180 [0.177, 0.184] | 0.725 |
| **XGBoost (raw)** | **0.801 ± 0.004** | **0.804 [0.797, 0.812]** | **0.785** | **0.180 [0.176, 0.183]** | 0.722 |

**Honest reading.** XGBoost's advantage is small (about +0.011 ROC-AUC over Logistic Regression) but consistent across splits. The reliability analyses then expose its costs, which are reported, not hidden:
- **Step-wise responses:** a 1% input change can move its estimate by up to 37 percentage points (pp); the 95th percentile is 11 pp.
- **Non-monotone curves:** for every tested profile, its curves in blood pressure, age and weight are non-monotone.
- **Less stable explanations:** the top-5 explanation features stayed identical in 74% of ±1% perturbations, against 95% for Logistic Regression.

**Calibration.** XGBoost's raw probabilities already agree closely with observed outcome frequencies in this dataset (out-of-fold ECE 0.004, slope 0.99), so no calibrator was adopted. Raw Logistic Regression has an S-shaped miscalibration (slope ≈ 1, yet ECE 0.034) that isotonic calibration corrects and sigmoid calibration cannot.

**Thresholds.** At 0.30, 0.50 and 0.70, recall/specificity on the test set is 0.89/0.48, 0.69/0.78 and 0.52/0.89. Threshold choice trades false positives against false negatives; the net-benefit curve is exploratory, and no threshold is called optimal. *Model probability*, *display band* (<30%, 30–60%, ≥60%) and *classification threshold* are three separate concepts.

**Subgroup Performance Analysis:**
- The gender groups are similar (ROC-AUC 0.803 / 0.806).
- ROC-AUC falls with age: 0.830 at 40–49 but **0.699 [0.678, 0.722] at 60–65**.
- Mean predicted tracks prevalence in every group.

No fairness claim is made.

**Reliability and robustness:**
- **Input conformity:** a Mahalanobis distance, fitted on training data only, flags about 1% of inputs as unusual.
- **Model disagreement:** across the three candidates, the median spread is 5 pp and the 95th percentile 17 pp.
- **Synthetic shifts:** ranking drops to about 0.75 ROC-AUC in older or higher-blood-pressure populations, and a +10 mmHg recording offset inflates the mean estimate by 14 pp.

**Explainability:** exact TreeSHAP on XGBoost's log-odds score reconciles with the model output within 1e-5. Systolic BP has 49% of the attribution, age 16% and cholesterol 13%. SHAP and the other analyses describe the model, never medical causation. In this dataset, smoking and alcohol show slightly *lower* label rates (a confounded pattern), and the model reproduces it.

## Architecture

```
src/data_loader.py      download + checksum, cleaning rules (with exclusions), train/test split
src/preprocessing.py    schema, plausibility ranges, labels, BMI, preprocessing pipeline
src/evaluate_models.py  evaluation layer: metrics, repeated OOF CV, corrected t-test, calibration stats,
                        thresholds, decision curve, bootstrap, subgroups
src/train_models.py     9-variant repeated-CV comparison, selection protocol, persistence -> metrics.json, models/
src/analysis.py         test-set analyses + data-quality report -> artifacts/analysis.json
src/reliability.py      calibration deep dive, OOF thresholds, novelty detector, stability, monotonicity,
                        disagreement -> artifacts/reliability.json; per-assessment reliability checks
src/shift_analysis.py   synthetic distribution-shift experiment -> artifacts/shift_analysis.json
src/prediction.py       input contract (PatientInput), artifact checks, calibrated prediction, demo inputs
src/explainability.py   exact SHAP (linear or TreeSHAP) on the model score, probability mapping, global importance
src/what_if.py          guarded what-if scenarios and response curves
src/recommendations.py  rule-based informational guidance, prototype bands, disclaimer
app.py                  Streamlit entry point (top navigation)
ui/core.py              design tokens, cached resources, shared components
ui/{assess,explain,explore,model,methodology}.py   the five pages
docs/                   methodology.md, evaluation.md, model_card.md
tests/                  145 tests (data, models, evaluation, selection, reliability, prediction, SHAP, what-if, guidance, UI)
```

The rule-based guidance engine (`src/recommendations.py`) uses fixed, transparent rules on the user's inputs and the probability band, with no LLM. It never states a condition or names a treatment. A lower estimate is described as "does not rule out any health condition".

## Setup

Requires **Python 3.12**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost needs OpenMP:

```bash
brew install libomp
```

Generate the models and the analysis artifacts. The first run also downloads the dataset. The four steps take about 5 minutes, 1 minute, 1.5 minutes and a few seconds.

The seed is fixed, so:
- results reproduce exactly for Logistic Regression;
- Random Forest and XGBoost reproduce to floating-point precision (about 1e-16, from multithreaded tree training);
- every selection decision is identical.

```bash
python -m src.train_models
```

```bash
python -m src.analysis
```

```bash
python -m src.reliability
```

```bash
python -m src.shift_analysis
```

`artifacts/metrics.json`, `analysis.json`, `reliability.json` and `shift_analysis.json` are committed. Model files (including `models/novelty_detector.joblib`), `artifacts/oof_predictions.npz` and the dataset are not.

Command-line checks of the backend:

```bash
python -m src.prediction
```

```bash
python -m src.explainability
```

```bash
python -m src.recommendations
```

## Tests

```bash
python -m pytest -q
```

The suite contains 145 tests. It covers:
- metric definitions and invariants (threshold metrics are consistent, recall never rises with the threshold, net-benefit formula);
- repeated CV (fixed seed, 25 disjoint splits, each row validated once per repeat) and out-of-fold leakage (a memorising model on random labels scores at chance);
- test-set integrity: the final pipeline and calibrators are reproduced from training rows alone, the selection ignores scrambled test numbers, no threshold is tuned, and the novelty detector is fitted on training rows only;
- the selection protocol reproducing the recorded decision and behaving correctly on synthetic inputs;
- calibration statistics, reproducible calibration artifacts, bootstrap reproducibility and interval ordering, and subgroup "metric unavailable" handling;
- input conformity (unusual inputs are flagged without blocking), deterministic perturbation and explanation-stability statistics, candidate-model disagreement, and a reproducible synthetic shift kept separate from validation;
- SHAP reconciling with the model score and probability for every candidate model, within 1e-5;
- what-if guardrails, the baseline being reproduced, and response curves staying in the valid domain;
- guidance language;
- every UI page driven through Streamlit's `AppTest`.

## Intended use and limitations

See the [model card](docs/model_card.md).

- **Intended for:** teaching, demonstration and critical inspection of explainable, human-centred AI on tabular health data.
- **Not intended for:** diagnosis, screening, triage, treatment or any clinical decision, or for people outside the data's coverage (ages 29–65).

**Known limitations:**
- The selected XGBoost model responds in steps, is non-monotone in its main inputs, and has less stable explanations than Logistic Regression.
- The dataset has limited provenance.
- Lifestyle inputs are self-reported, and the lab inputs are coarse three-level categories.
- The model has learned confounded patterns (smoking, alcohol, glucose "well above normal").
- Ranking performance is weaker for ages 60–65.
- Correlated blood-pressure features share explanation credit.
- Prototype bands and thresholds are not clinical cut-offs.
- The saved models are tied to the pinned scikit-learn version.
