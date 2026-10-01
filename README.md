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
| **Assess** | Product header and a prominent educational disclaimer above a grouped input form (Demographics, Body measurements, Blood pressure, Laboratory indicators, Lifestyle) with three synthetic examples. **Run Assessment** shows the model-estimated probability as the headline (with a note when it lies within 3 pp of a band boundary or the 0.50 threshold), a "What the model does not know" panel, the prototype estimate band, and secondary information (model, calibration status, input conformity, reference baseline, class at 0.50). It also shows a short "How the model arrived here" preview, general guidance and an input summary. Unusual inputs get a caution; they are never blocked. |
| **Explain** | The full SHAP explanation of the current assessment. It opens with a plain-language summary, then shows the reference baseline, this estimate and the difference between them; a diverging contribution chart with ▲/▼ markers, so direction does not rely on colour; and a note that model contribution ≠ medical causation. The **Assessment reliability** panel shows probability, input conformity, explanation stability, model disagreement and calibration status as separate signals, with no combined score. |
| **Explore** | What-if (model sensitivity) analysis. Change systolic or diastolic BP, weight, cholesterol, glucose, smoking, activity or alcohol, and compare baseline, scenario and change in percentage points, with the direction stated. A note, read from the robustness analysis, warns when the model is non-monotone in the plotted input. It also shows each change on its own and the model estimate across one input. Impossible combinations are rejected, and the stored assessment is never modified. |
| **Model** | A scope note (dataset and protocol, not clinical performance), then tabs for Overview (repeated-CV comparison, stability box plots, selection protocol), Calibration (reliability diagram with bin counts, ECE, slope and intercept), ROC & PR, Thresholds (0.05–0.95, decision trade-off, net benefit), Explainability, Subgroups (with intervals), Uncertainty (bootstrap) and Robustness (novelty, stability, monotonicity, disagreement, synthetic shift). |
| **Methodology** | Pipeline diagram, Data Quality Lab (exclusions with examples, ranges, distributions, correlations), model card, validation strategy, limitations, and a **Scope & human-centred design** tab mapping each user question (what it knows, predicts, does not know, where it is uncertain) to where the app answers it. |

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
scripts/setup.sh        fresh-clone setup and full regeneration
tests/                  155 tests (data, models, evaluation, selection, reliability, prediction, SHAP, what-if, guidance, UI)
```

The rule-based guidance engine (`src/recommendations.py`) uses fixed, transparent rules on the user's inputs and the probability band, with no LLM. It never states a condition or names a treatment. A lower estimate is described as "does not rule out any health condition".

## Setup

Tested on macOS (Apple Silicon) with **Python 3.12**. From a fresh clone:

```bash
./scripts/setup.sh
```

The script is safe to rerun and only creates what is missing:
1. checks for Python 3.12 (`PYTHON=/path/to/python3.12 ./scripts/setup.sh` to choose one);
2. creates `.venv` and installs the pinned dependencies (`requirements.txt` for runtime, `requirements-dev.txt` adds pytest);
3. checks that XGBoost can load OpenMP (on macOS: `brew install libomp`, the one system dependency);
4. downloads and validates the dataset;
5. trains the models if `models/` is missing them or they were saved by other library versions, then builds the novelty detector;
6. loads the saved model and scores the three synthetic demo inputs.

The first run takes about 10 minutes on an Apple Silicon laptop (measured: 9 min 16 s for a full `--regenerate` including the dependency install), mostly the 9-variant repeated-CV comparison.

| Task | Command |
|---|---|
| Set up | `./scripts/setup.sh` |
| Run the tests | `.venv/bin/python -m pytest -q` |
| Run the app | `.venv/bin/streamlit run app.py` |
| Retrain the models only | `.venv/bin/python -m src.train_models` |
| Regenerate every model and artifact | `./scripts/setup.sh --regenerate` |
| Check the saved model | `.venv/bin/python -m src.prediction` |

Other command-line checks: `python -m src.explainability` and `python -m src.recommendations`.

### Dataset setup

The dataset is not committed. `src/data_loader.py` downloads it from Kaggle's public API on first use, with **no account or credentials**: `https://www.kaggle.com/api/v1/datasets/download/sulianova/cardiovascular-disease-dataset`. It then validates the file:
- **Checksum:** `cardio_train.csv` must have SHA-256 `21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c`, which pins all 70,000 rows exactly.
- **Format:** semicolon-separated, columns `id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio` in that order.
- **Cleaning:** the cleaned row count must be 68,573, checked by `setup.sh`.

**If the download fails** (offline, or Kaggle changes its API), download the dataset manually from the [Kaggle page](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset), unzip it, and place `cardio_train.csv` at `data/cardio_train.csv`. The checksum is still verified.

**Raw schema:**
- `age` is in days and is converted to years.
- `gender`: 1 = female, 2 = male.
- `height` is in cm and `weight` in kg.
- `ap_hi` and `ap_lo` are systolic and diastolic blood pressure, in mm Hg.
- `cholesterol` and `gluc`: 1 = normal, 2 = above normal, 3 = well above normal.
- `smoke`, `alco` and `active` are 0/1.
- **Target** `cardio`: 1 = cardiovascular disease present, 0 = absent.

**Split:** stratified 80/20 with seed 42, giving 54,858 train and 13,715 test rows.

### Reproducibility

- **Seeds:** every random step uses seed 42: the split, the CV folds, the models, the bootstraps and the perturbations.
- **Repeatability:** regenerating gives identical selection decisions. Logistic Regression reproduces exactly; the tree models reproduce to floating-point precision (multithreaded training).
- **Committed outputs:** `artifacts/metrics.json`, `analysis.json`, `reliability.json` and `shift_analysis.json` are committed, so the Model and Methodology pages can be read without retraining.
- **Provenance:** each committed artifact carries a `provenance` block with the generation time, the Python and library versions, the dataset SHA-256 and the seed.
- **Not committed:** model files (`models/*.joblib`), `artifacts/oof_predictions.npz` and the dataset. They are rebuilt by `setup.sh`.
- **Loader checks:** the model loader refuses a model file that was built with different scikit-learn/XGBoost versions, or whose feature contract does not match the code. It tells you the command that regenerates it.
- **Dependencies:** all are pinned in `requirements.txt`. Pickled scikit-learn/XGBoost models are only reliable on the versions that wrote them, so pins should be changed together with a full regeneration.

## Tests

```bash
python -m pytest -q
```

The suite contains 155 tests. It covers:
- metric definitions and invariants (threshold metrics are consistent, recall never rises with the threshold, net-benefit formula);
- repeated CV (fixed seed, 25 disjoint splits, each row validated once per repeat) and out-of-fold leakage (a memorising model on random labels scores at chance);
- test-set integrity: the final pipeline and calibrators are reproduced from training rows alone, the selection ignores scrambled test numbers, no threshold is tuned, and the novelty detector is fitted on training rows only;
- the selection protocol reproducing the recorded decision and behaving correctly on synthetic inputs;
- calibration statistics, reproducible calibration artifacts, bootstrap reproducibility and interval ordering, and subgroup "metric unavailable" handling;
- input conformity (unusual inputs are flagged without blocking), deterministic perturbation and explanation-stability statistics, candidate-model disagreement, and a reproducible synthetic shift kept separate from validation;
- SHAP reconciling with the model score and probability for every candidate model, within 1e-5;
- what-if guardrails, the baseline being reproduced, and response curves staying in the valid domain;
- guidance language;
- every UI page driven through Streamlit's `AppTest`;
- portability: no machine-specific paths in tracked files, pinned requirements, provenance in committed artifacts, and stale model files rejected.

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
- The saved models are tied to the pinned scikit-learn and XGBoost versions; the loader rejects a mismatch.
- Setup is tested on macOS (Apple Silicon) with Python 3.12. Linux should work; Windows is untested.

## Future work

- External validation on a dataset with documented clinical provenance.
- A monotone-constrained XGBoost model, evaluated under the same protocol, to address the step-wise and non-monotone responses.
- Continuous integration that runs the test suite on every push.

## License

[MIT](LICENSE) © 2026 Soyeb Mohammad. The dataset is not covered by this license; it is subject to its own terms on Kaggle.
