# AI-Powered Healthcare Risk Assessment Assistant

**An Explainable AI-Based Human-Centered Healthcare Decision Support System.** This is an educational research prototype for the B.Tech course *Design of Artificial Intelligence Products*.

> **Medical disclaimer.** This prototype does **not** diagnose cardiovascular disease, prescribe treatment, or replace a healthcare professional, and it is not medical advice. Its outputs are model-based estimates from one public research dataset and are **not clinically validated**. Calibrated probabilities agree with that dataset's labels; they are not clinical risks.

The system estimates a probability, shows how trustworthy and how explainable that estimate is, and lets you inspect the model critically. Its components:
- **A calibrated model.** The selected model's probabilities are corrected to match observed rates in the data.
- **Exact explanations.** Each assessment is broken down into per-input contributions using SHAP.
- **What-if analysis.** You can change inputs and see how the model estimate responds.
- **Model analytics.** Calibration, threshold, subgroup and bootstrap-uncertainty analyses of the models.
- **Data Quality Lab and model card.** What was removed from the data and why, plus a formal statement of intended use and limitations.

## Application

```bash
streamlit run app.py
```

The app has five areas in a top navigation bar. Each area is a page script in `ui/`; all model logic lives in `src/`.

| Area | What it is for |
|---|---|
| **Assess** | Grouped input form (Person, Vitals, Labs, Lifestyle) with three synthetic examples. **Run Assessment** shows the model estimate as the headline, the prototype estimate band, secondary information (model, calibration status, reference baseline, class at 0.50), a short "How the model arrived here" preview, general guidance and an input summary. |
| **Explain** | The full SHAP explanation of the current assessment. It shows the reference baseline, this estimate and the difference between them; a diverging contribution chart with ▲/▼ markers, so direction does not rely on colour; and a note that model contribution ≠ medical causation. |
| **Explore** | What-if (model sensitivity) analysis. Change systolic or diastolic BP, weight, cholesterol, glucose, smoking, activity or alcohol, and compare baseline, scenario and change in percentage points. It also shows each change on its own and the model estimate across one input. Impossible combinations are rejected, and the stored assessment is never modified. |
| **Model** | Tabs for Overview (comparison and selection framework), Calibration, ROC & PR, Thresholds (interactive), Explainability (SHAP vs permutation importance, partial dependence and ICE, interactions), Subgroups and Uncertainty (bootstrap). |
| **Methodology** | Pipeline diagram, Data Quality Lab (exclusions with examples, ranges, distributions, correlations), model card, validation strategy and limitations. |

Design notes:
- Theming uses Streamlit's native light and dark themes (`.streamlit/config.toml`), the Inter font, and semantic colours: teal = moved the estimate lower or lower band, amber = moderate, red/orange = higher. Every coloured item also has a text label or ▲/▼ marker.
- The model and explainer are cached once per server (`st.cache_resource`).
- All global analyses are precomputed into `artifacts/analysis.json`, so an assessment only costs one prediction and one exact SHAP call.

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

- **Strategy:** stratified 80/20 split (54,858 train / 13,715 test). All comparisons use stratified 5-fold CV on the training split. Preprocessing and calibration are fitted inside each fold, and a test confirms there is no leakage.
- **Models:** Logistic Regression, Random Forest and XGBoost, each tested **raw, sigmoid-calibrated and isotonic-calibrated**, which makes 9 variants.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss and Brier score.
- **Selection framework** (CV only):
  - keep raw probabilities unless a calibrator lowers CV Brier by at least 0.001 *and* in every fold;
  - start from the most interpretable model and move to a less transparent one only for a CV ROC-AUC gain of at least 0.02 without worse calibration.

  The margins are stated judgement calls.
- **Selected model: Logistic Regression with isotonic calibration.**

| Model (selected variant) | CV ROC-AUC | Test ROC-AUC [95% CI] | Test PR-AUC | Test Brier [95% CI] | Test F1 at 0.50 |
|---|---|---|---|---|---|
| **Logistic Regression (isotonic)** | 0.791 ± 0.004 | 0.793 [0.786, 0.801] | 0.770 | 0.184 [0.181, 0.188] | 0.725 |
| Random Forest (raw) | 0.800 ± 0.004 | 0.803 [0.795, 0.810] | 0.783 | 0.181 [0.177, 0.184] | 0.717 |
| XGBoost (raw) | 0.801 ± 0.004 | 0.804 [0.797, 0.812] | 0.785 | 0.180 [0.176, 0.183] | 0.722 |

**Honest reading.** XGBoost is reliably but only slightly better: paired bootstrap ROC-AUC difference +0.011 [0.009, 0.014], Brier −0.005. Logistic Regression is kept because its effects are monotone and its explanations are exact, and the gain falls under the stated margin.

**Calibration.** Raw Logistic Regression was systematically miscalibrated: predicted 0.35 → observed 0.31, predicted 0.65 → observed 0.72, predicted 0.94 → observed 0.86. Isotonic calibration lowered CV Brier from 0.1870 to 0.1852 and log loss from 0.561 to 0.555 in every fold, without changing ROC-AUC. Its step function slightly lowers PR-AUC, and calibrated probabilities rarely exceed about 0.9. For example, demo C moves from 97.9% raw to 86.1% calibrated.

**Thresholds.** At 0.30, 0.50 and 0.70, recall/specificity is 0.88/0.47, 0.72/0.75 and 0.51/0.89. The threshold is a trade-off between false positives and false negatives; none is clinically optimal. *Probability*, *display band* (<30%, 30–60%, ≥60%) and *classification threshold* are three separate concepts.

**Subgroup Performance Analysis:**
- The gender groups are similar (ROC-AUC 0.791 / 0.797).
- ROC-AUC falls with age: 0.816 at 40–49 but **0.692 at 60–65**.
- At 0.50, recall follows each age group's base rate.
- The youngest group (n = 367) is slightly over-estimated.

No fairness claim is made.

**Explainability:**
- Exact SHAP on the model score: systolic BP 45%, age 17%, cholesterol 16% of mean |SHAP|.
- Permutation importance agrees at the top: systolic BP's ROC-AUC drop is 0.17.
- Partial dependence and ICE: Logistic Regression is smooth, XGBoost has a step at 125–140 mmHg.
- XGBoost interactions account for about 30% of its attribution, led by age × systolic BP.

SHAP and the other analyses describe the model, never medical causation. In this dataset, smoking and alcohol show slightly *lower* label rates (a confounded pattern), so the model gives them small negative contributions. The analyses surface this artefact rather than hide it.

## Architecture

```
src/data_loader.py      download + checksum, cleaning rules (with exclusions), train/test split
src/preprocessing.py    schema, plausibility ranges, labels, BMI, preprocessing pipeline
src/evaluate_models.py  evaluation layer: metrics, OOF CV, calibration bins, thresholds, bootstrap, subgroups
src/train_models.py     9-variant CV comparison, selection framework, persistence -> metrics.json, models/
src/analysis.py         test-set analyses + data-quality report -> artifacts/analysis.json
src/prediction.py       input contract (PatientInput), artifact checks, calibrated prediction, demo inputs
src/explainability.py   exact SHAP on the LR score, calibrated-probability mapping, global importance
src/what_if.py          guarded what-if scenarios and response curves
src/recommendations.py  rule-based informational guidance, prototype bands, disclaimer
app.py                  Streamlit entry point (top navigation)
ui/core.py              design tokens, cached resources, shared components
ui/{assess,explain,explore,model,methodology}.py   the five pages
docs/                   methodology.md, evaluation.md, model_card.md
tests/                  121 tests (data, models, evaluation, analysis, prediction, SHAP, what-if, guidance, UI)
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

Generate the model and the analysis artifacts. The first run also downloads the dataset. The two steps take about 1.5 minutes and 30 seconds; the seed is fixed, so results reproduce exactly for Logistic Regression and to floating-point precision (about 1e-16, from multithreaded tree training) for Random Forest and XGBoost; every selection decision is identical.

```bash
python -m src.train_models
```

```bash
python -m src.analysis
```

`artifacts/metrics.json` and `artifacts/analysis.json` are committed. Model files and the dataset are not.

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

The suite contains 121 tests. It covers:
- metric definitions and invariants (threshold metrics are consistent, recall never rises with the threshold);
- out-of-fold leakage (a memorising model on random labels scores at chance);
- bootstrap reproducibility, subgroup flagging and the selection framework reproducing the recorded decision;
- calibration monotonicity, and SHAP reconciling exactly with the model score and the calibrated probability;
- what-if guardrails, the baseline being reproduced, and response curves staying in the valid domain;
- guidance language;
- every UI page driven through Streamlit's `AppTest`.

## Intended use and limitations

See the [model card](docs/model_card.md).

- **Intended for:** teaching, demonstration and critical inspection of explainable, human-centred AI on tabular health data.
- **Not intended for:** diagnosis, screening, triage, treatment or any clinical decision, or for people outside the data's coverage (ages 29–65).

**Known limitations:**
- The dataset has limited provenance.
- Lifestyle inputs are self-reported, and the lab inputs are coarse three-level categories.
- The model has learned confounded patterns (smoking, alcohol, glucose "well above normal").
- Ranking performance is weaker for ages 60–65.
- Correlated blood-pressure features share explanation credit.
- Prototype bands and thresholds are not clinical cut-offs.
- The saved models are tied to the pinned scikit-learn version.
