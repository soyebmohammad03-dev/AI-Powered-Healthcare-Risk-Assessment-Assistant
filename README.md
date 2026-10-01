# AI-Powered Healthcare Risk Assessment Assistant

Academic project for **Design of Artificial Intelligence Products** (B.Tech).
Theme: Healthcare + Human-Centered AI + Explainable AI.

> **Medical disclaimer:** This is an educational AI decision-support prototype. It does **not** diagnose cardiovascular disease, prescribe treatment, or replace a doctor, and it is not medical advice. Its outputs come from a model trained on a public research dataset and are **not clinically validated**. A larger dataset does not make it clinically validated either.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Dataset, preprocessing, three-model comparison | Done (migrated to the Cardiovascular Disease dataset) |
| 2 | Model selection, persistence, prediction pipeline | Done |
| 3 | SHAP explainability | Done |
| 4 | Rule-based informational guidance | Done |
| 5 | Streamlit application | Planned |
| 6 | Integration, testing, demo preparation | Planned |

```
PatientInput → validation → ML prediction ─┬─ probability → prototype risk band ─┐
                                           └─ SHAP explanation ──────────────────┴→ rule engine → Guidance (+ disclaimer)
```

| Layer | Module | Question it answers |
|---|---|---|
| ML model | `src/prediction.py` | What probability does the model estimate? |
| SHAP | `src/explainability.py` | How did the model weight these inputs? |
| Recommendation engine | `src/recommendations.py` | What general information should be shown alongside it? |

## Dataset

**Cardiovascular Disease dataset** (Svetlana Ulianova, Kaggle): https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

`src/data_loader.py` downloads `cardio_train.csv` from Kaggle's public API endpoint, which needs no account. It then checks the file's SHA-256 hash. The file is not committed to this repository; the first run of training or the tests downloads it.

**Raw file as inspected**

| Property | Value |
|---|---|
| Delimiter | semicolon |
| Shape | 70,000 rows × 13 columns |
| Columns | `id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio` |
| Missing values | none |
| Duplicate ids | none |
| Records identical apart from `id` | 24 |
| Target `cardio` | 0 (absent): 35,021 · 1 (present): 34,979 |

The raw file also contains measurement errors:
- `ap_hi` ranges from −150 to 16,020 and `ap_lo` from −70 to 11,000.
- 1,234 rows have systolic pressure below diastolic.
- Heights range from 55 to 250 cm, and weights go as low as 10 kg.

**Cleaning.** These are data-quality rules for removing recording errors. They are not medical thresholds and were not tuned for performance. They are applied in this order:

| Rule | Rows removed |
|---|---|
| Malformed, missing or non-numeric values | 0 |
| Target not 0/1 | 0 |
| Category code not documented | 0 |
| Duplicate record (all columns except `id`) | 24 |
| Age outside 29–65 years | 0 |
| Height outside 120–220 cm | 53 |
| Weight outside 30–250 kg | 7 |
| Systolic `ap_hi` outside 60–250 mm Hg | 226 |
| Diastolic `ap_lo` outside 30–200 mm Hg | 989 |
| Systolic not above diastolic | 103 |
| BMI outside 12–60 (height and weight each plausible, but not together) | 25 |
| **Clean dataset** | **68,573** records: 0 = 34,646, 1 = 33,927 |

**Features**

| Input | Meaning | Model use |
|---|---|---|
| `age` | Age in years (raw file stores days; converted as days / 365.25) | numeric |
| `gender` | 1 = female, 2 = male (see note below) | categorical |
| `height`, `weight` | cm, kg | combined into **BMI** = weight / (height in m)² |
| `ap_hi` | Systolic blood pressure, mm Hg | numeric |
| `ap_lo` | Diastolic blood pressure, mm Hg | numeric |
| `cholesterol` | 1 normal, 2 above normal, 3 well above normal | categorical |
| `gluc` | Glucose: 1 normal, 2 above normal, 3 well above normal | categorical |
| `smoke`, `alco`, `active` | Self-reported smoking, alcohol intake, physical activity (0/1) | categorical |

- **Gender coding:** the source does not label the two codes. 1 = female is inferred from mean height (161 cm vs 170 cm), which is how this dataset is usually read.
- **`id`:** dropped during cleaning. It is never a feature.

**Derived features.** BMI is the only one. It is computed inside the saved model pipeline, so training and inference compute it identically. BMI replaces raw height and weight because adding them back changed cross-validation ROC-AUC by at most 0.0002. Pulse pressure (`ap_hi − ap_lo`) is deliberately not added: it is an exact linear combination of two existing features, so it carries no new information, and for a linear model it would split SHAP credit arbitrarily.

## Method

- **Split:** stratified 80/20 train/test split, seed 42: 54,858 train and 13,715 test records. The test set is held out.
- **Validation:** stratified 5-fold cross-validation on the training set.
- **Preprocessing** (`src/preprocessing.py`): one `Pipeline` that is fitted on training data only:
  1. derive BMI;
  2. standard-scale age, BMI, `ap_hi` and `ap_lo`;
  3. one-hot encode the categorical codes, where any undocumented code raises an error.
- **Models:** Logistic Regression, Random Forest and XGBoost, each combined with the preprocessor in one pipeline.

## Results (measured, seed 42)

Produced by `python -m src.train_models`. The full report is in `artifacts/metrics.json`.

| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 5-fold CV | 0.728 ± 0.004 | 0.756 | 0.666 | 0.708 | 0.791 ± 0.004 |
| | Test | 0.728 | 0.755 | 0.665 | 0.707 | 0.793 |
| Random Forest | 5-fold CV | 0.733 ± 0.004 | 0.764 | 0.667 | 0.712 | 0.800 ± 0.004 |
| | Test | 0.736 | 0.765 | 0.674 | 0.717 | 0.803 |
| XGBoost | 5-fold CV | 0.735 ± 0.003 | 0.753 | 0.691 | 0.721 | 0.801 ± 0.004 |
| | Test | 0.737 | 0.756 | 0.692 | 0.722 | 0.804 |

These scores are typical for this dataset: published work on it generally reports around 0.73 accuracy. The available features are coarse, and some labels and readings are noisy.

## Final model: Logistic Regression

The final model is chosen on cross-validation first. The test set is used only as a consistency check.

- **Performance:** XGBoost and Random Forest lead by about 0.01 ROC-AUC (CV 0.801 and 0.800 versus 0.791) and by under 1 point of accuracy.
- **Stability:** all three models are equally stable (CV standard deviation ≤ 0.01), and each one's test scores match its CV scores.
- **Interpretability:** Logistic Regression's effects are monotone and the same for every patient. For example, a higher systolic blood pressure always moves its estimate up. Its SHAP values are exact (`coef × (x − mean)`) and easy to explain. For an explainable decision-support prototype, this outweighs a 1-point AUC gap.

The other models are still saved to `models/` and reported in `artifacts/metrics.json`. The reasoning above is also recorded in `metrics.json` (`final_model_reason`).

## Prediction pipeline

- **Saved artifact:** `models/final_model.joblib` holds the entire fitted pipeline (BMI derivation → scaler and one-hot encoder → classifier). It also stores the input features, the model features, the category codes, the test metrics and the library versions. Training (`src/train_models.py`) writes it; inference (`src/prediction.py`) only loads it and never retrains.
- **Artifact checks:** `load_model()` raises `ModelArtifactError` with a fix hint if the file is missing, corrupted, has no preprocessing step, or uses a different feature contract.
- **Input contract:** `PatientInput` requires all 11 inputs.
  - Values must be numbers. Strings, booleans and NaN are rejected.
  - Category fields must use one of the documented codes.
  - Measurements must fall within the cleaning ranges above. Age must also be within the dataset's coverage of 29–65 years, so the model never extrapolates.
  - Cross-field checks: systolic must be above diastolic, and the resulting BMI must be between 12 and 60.
  - All problems are reported together as `InvalidInputError.errors`, with readable messages.
- **Output:** `PredictionResult` contains `predicted_class` (threshold 0.5), `probability_positive`, `probability_negative`, `model_name`, the validated `inputs` and the derived `bmi`. This is raw model output with no wording added.

```python
from src.prediction import PatientInput, predict
from src.explainability import explain
from src.recommendations import generate

patient = PatientInput(age=50, gender=2, height=175, weight=85, ap_hi=130, ap_lo=85,
                       cholesterol=1, gluc=1, smoke=1, alco=0, active=1)
result = predict(patient)
explanation = explain(patient)
guidance = generate(patient, result, explanation)
print(result.probability_positive, guidance.risk_category.value)
```

**Demo inputs.** `DEMO_INPUTS` contains three **synthetic** examples. They are written by hand and are not real patients or rows from the dataset. Every value in the tables below is computed by `python -m src.prediction`, `python -m src.explainability` and `python -m src.recommendations`.

| Demo | Inputs | P(pos) | Band | Largest SHAP contributions (log-odds) |
|---|---|---|---|---|
| A | 38 y, female, 165 cm / 60 kg (BMI 22.0), 115/75, all normal, non-smoker, active | 0.143 | Lower | Age −0.76, Systolic BP −0.66, Cholesterol normal −0.17 |
| B | 50 y, male, 175 cm / 85 kg (BMI 27.8), 130/85, normal levels, smoker, active | 0.451 | Moderate | Systolic BP +0.19, Cholesterol normal −0.17, Age −0.16, Smoking −0.14 |
| C | 61 y, female, 160 cm / 88 kg (BMI 34.4), 160/100, cholesterol well above, glucose above, inactive | 0.979 | Higher | Systolic BP +1.88, Cholesterol well above +0.89, Age +0.38 |

## Explainable AI (SHAP)

A **prediction** answers "what does the model estimate?". An **explanation** answers "how did the model weight these inputs?". SHAP (SHapley Additive exPlanations), which is based on Shapley values from game theory, splits one prediction into one additive contribution per feature:

```
base value + sum of all feature contributions = model output
```

- **Explainer:** `shap.LinearExplainer`, SHAP's exact explainer for linear models. It works on the classifier's real inputs, which are the 14 preprocessed columns. The reference data is all 54,858 preprocessed training records, with no subsampling.
- **Units:** the values are in **log-odds**, not percentages. The base value is +0.032 log-odds: the model's output at the average training record, which converts to a probability of 0.508. Converting `base + Σ contributions` to a probability gives exactly the probability returned by `predict()`.
- **Readable feature names:** the explainer reads from the fitted one-hot encoder which columns belong to which feature, and checks this against the column names. It then adds each feature's column contributions together, which is exact because SHAP is additive. So "Cholesterol: Well above normal" appears instead of `cat__cholesterol_3`. Height and weight enter the model only through BMI, so the explanation shows "BMI (from height and weight)".
- **Local explanation:** `explain(patient)` returns the 10 model features ranked by contribution. Each contribution has its readable label, the patient's value, the SHAP value, a direction, its share of the total and a careful sentence.
- **Global explanation:** `default_explainer().global_importance()` ranks the features by mean |SHAP| over all 68,573 cleaned records.

| Rank | Feature | Mean \|SHAP\| | Share |
|---|---|---|---|
| 1 | Systolic blood pressure | 0.729 | 45.0% |
| 2 | Age | 0.279 | 17.2% |
| 3 | Cholesterol | 0.253 | 15.6% |
| 4 | BMI | 0.115 | 7.1% |
| 5 | Diastolic blood pressure | 0.073 | 4.5% |
| 6 | Physical activity | 0.072 | 4.4% |
| 7 | Glucose | 0.040 | 2.5% |
| 8 | Smoking | 0.025 | 1.5% |
| 9 | Alcohol intake | 0.023 | 1.4% |
| 10 | Gender | 0.014 | 0.8% |

**SHAP explains the model, not medicine.** These values describe how the model weighted the input features. They do not show what causes cardiovascular disease. Generated sentences read "Systolic blood pressure (160) contributed toward a higher model-estimated probability for this input", never "caused".

## Informational guidance (rule-based)

The guidance comes from explicit, deterministic `if` rules on the user's own inputs and on the model's probability band. No model and no LLM is involved. An LLM could produce unsupported medical claims and different text on each run; plain rules can be read, tested exhaustively and explained in a viva. SHAP only adds one clearly labelled "what the model weighed most" note. It never decides what guidance is shown.

**Prototype risk bands.** These are applied to the model's probability for display only. They are **not clinically validated**, and the model is unchanged (it still uses 0.5 for `predicted_class`).

| Probability | Band |
|---|---|
| < 0.30 | Lower predicted risk |
| 0.30 – < 0.60 | Moderate predicted risk |
| ≥ 0.60 | Higher predicted risk |

**Rules.** The thresholds are prototype prompts aligned with commonly cited reference points. They are not diagnostic cut-offs.

| Trigger | Item | Priority |
|---|---|---|
| Higher band | Consider a professional evaluation | high |
| Moderate band | Consider discussing these results | moderate |
| Lower band | General preventive care. Says a lower estimate "does not rule out any health condition" and never says "healthy". | low |
| `ap_hi` ≥ 130 or `ap_lo` ≥ 80 (the ACC/AHA 2017 elevated/stage-1 range begins at 130/80) | Review your blood pressure readings | moderate |
| `cholesterol` ≥ 2 (dataset category) | Discuss your cholesterol level | moderate |
| `gluc` ≥ 2 (dataset category) | Discuss your glucose level | moderate |
| `smoke` = 1 | Consider support to stop smoking | moderate |
| BMI < 18.5 or ≥ 25 (WHO adult BMI categories) | Discuss your weight | low |
| `active` = 0 / 1 | Consider regular physical activity / Keep up your physical activity | low |
| `alco` = 1 | Consider your alcohol intake | low |
| SHAP explanation given | "What the model weighed most", ending with "…not a medical cause" | info |

How the items are arranged:
- The follow-up item always comes first.
- After it come up to 3 input items, ranked by priority. Ties keep the rule order in the table.
- The model-context note always comes last, so a result shows 3–5 items.
- Any further triggered items are kept in `Guidance.additional`; none are dropped.
- Every item has a `reason` field naming the input that triggered it.
- `DISCLAIMER` is a reusable string for the UI.

Guidance for the demo inputs (`python -m src.recommendations`):

| Demo | Items shown | Also in `additional` |
|---|---|---|
| A (Lower) | General preventive care · Keep up activity · Model weighed *Age* most, moving the estimate lower | none |
| B (Moderate) | Discuss these results · Blood pressure 130/85 · Support to stop smoking · Weight (BMI 27.8) · Model weighed *Systolic BP* most, moving the estimate higher | Keep up activity |
| C (Higher) | Professional evaluation · Blood pressure 160/100 · Cholesterol well above normal · Glucose above normal · Model weighed *Systolic BP* most, moving the estimate higher | Weight (BMI 34.4) · Consider activity |

## Setup

Requires **Python 3.12**. The pinned versions were tested on 3.12; Python 3.14 is not recommended for SHAP/XGBoost wheels.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost also needs OpenMP:

```bash
brew install libomp
```

## Train

```bash
python -m src.train_models
```

This downloads the dataset if needed and trains all three models (about 15 seconds). It writes `models/final_model.joblib`, the per-model pipelines `models/{logistic_regression,random_forest,xgboost}.joblib`, and `artifacts/metrics.json`. Neither the model files nor the dataset are committed; run this once after cloning. The seed is fixed, so the results are identical each time.

## Run the backend from the command line

```bash
python -m src.prediction
```

```bash
python -m src.explainability
```

```bash
python -m src.recommendations
```

## Test

```bash
python -m pytest -q
```

The tests use the real dataset and the saved model; they download and train once if either is missing.

## Project structure

```
app.py                  Streamlit app (Phase 5)
data/                   cardio_train.csv (downloaded, not committed)
models/                 Trained pipelines, incl. final_model.joblib (generated)
artifacts/metrics.json  Measured results, cleaning report, model-selection reasoning
src/data_loader.py      Download + checksum, cleaning rules, train/test split
src/preprocessing.py    Feature schema, plausibility ranges, labels, BMI, preprocessing pipeline
src/evaluate_models.py  Metrics
src/train_models.py     Training, cross-validation, test evaluation, saving
src/prediction.py       Input contract, artifact loading/checks, inference, demo inputs
src/explainability.py   SHAP local explanations and global importance
src/recommendations.py  Prototype risk bands, rule-based guidance, disclaimer
tests/                  Automated tests
docs/methodology.md     Methodology notes
```

## Limitations

- **Not clinical validation.** A model reaching about 0.79 ROC-AUC on one public dataset is not a validated clinical tool. The larger dataset improves statistical stability, not clinical validity.
- **Provenance:** Kaggle gives limited information about where the data came from. It describes the data as examination results. The population, the time period and how the label was defined are not documented in detail.
- **Self-reported inputs:** `smoke`, `alco` and `active` are self-reported. Cholesterol and glucose are only three-level categories, not lab values.
- **Learned patterns that contradict medical knowledge:** in this dataset, smokers and drinkers have *slightly lower* disease rates (smoking 46.9% vs 49.7%). As a result, the model gives Smoking = Yes a small *negative* contribution. Smoking is also strongly linked to gender here (22% of code-2 records versus 2% of code-1 records). This is a confounded pattern in the data, not medical evidence. It is a clear example of why SHAP explains the model and not medicine. The guidance engine still suggests support to stop smoking, because guidance is based on the inputs, not on the model. Similarly, "Glucose: well above normal" gets a slightly smaller learned effect than "above normal".
- **Measurement errors:** the cleaning removes clear recording errors but cannot catch plausible-looking wrong values.
- **Correlated features:** SHAP treats features as independent of each other. Correlated features (such as systolic and diastolic blood pressure) can therefore share credit in ways that do not match their clinical meaning.
- **Age coverage:** inputs are limited to ages 29–65, the range the dataset covers.
- **Library versions:** the model files are pickles tied to the pinned scikit-learn version; retrain after upgrading it.
- **Fixed rules:** the guidance comes from fixed rules and is not personalised medical advice. Its thresholds are prototype prompts, and its risk bands are presentation categories, not clinical standards.
