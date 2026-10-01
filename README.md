# AI-Powered Healthcare Risk Assessment Assistant

Academic project for **Design of Artificial Intelligence Products** (B.Tech).
Theme: Healthcare + Human-Centered AI + Explainable AI.

> **Medical disclaimer:** This is an educational AI decision-support prototype. It does **not** diagnose heart disease, prescribe treatment, or replace a doctor, and it is not medical advice. Its outputs come from a model trained on a small historical research dataset and are not clinically validated.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Dataset acquisition, inspection, preprocessing, baseline training & evaluation | Done |
| 2 | Model selection, persistence, prediction pipeline | Done |
| 3 | SHAP explainability | Done |
| 4 | Rule-based informational guidance | Planned |
| 5 | Streamlit application | Planned |
| 6 | Integration, testing, demo preparation | Planned |

## Dataset

**UCI Heart Disease, Cleveland subset** (`processed.cleveland.data`)
Source: https://archive.ics.uci.edu/dataset/45/heart+disease (Janosi, Steinbrunn, Pfisterer, Detrano, 1988), licensed CC BY 4.0.

The file is included in `data/` (18 KB). `src/data_loader.py` re-downloads it if it is missing and checks its SHA-256 hash.

- **Records:** 303
- **Target:** the original `num` column (0–4) is binarized as `target = 1` when `num > 0` (disease present) and `0` otherwise. This is the presence/absence task described in the dataset documentation. Class counts: 164 absent, 139 present.
- **Missing values:** 6 values are recorded as `?`: 4 in `ca` and 2 in `thal`. They are imputed inside the model pipeline, fitted on training data only.

| Feature | Type | Meaning |
|---|---|---|
| age | numeric | Age in years |
| trestbps | numeric | Resting blood pressure (mm Hg) |
| chol | numeric | Serum cholesterol (mg/dl) |
| thalach | numeric | Maximum heart rate achieved |
| oldpeak | numeric | ST depression induced by exercise relative to rest |
| ca | numeric (ordinal 0–3) | Number of major vessels coloured by fluoroscopy |
| sex | categorical | 0 = female, 1 = male |
| cp | categorical | Chest pain: 1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic |
| fbs | categorical | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | categorical | Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| exang | categorical | Exercise-induced angina (1 = yes) |
| slope | categorical | Slope of peak exercise ST segment: 1 up, 2 flat, 3 down |
| thal | categorical | Thallium test: 3 normal, 6 fixed defect, 7 reversible defect |

## Method (Phase 1)

- **Split:** stratified 80/20 train/test split (242 / 61 records), seed 42. The test set is held out.
- **Validation:** stratified 5-fold cross-validation on the training set.
- **Preprocessing** (`src/preprocessing.py`): numeric features use median imputation and standard scaling. Categorical features use most-frequent imputation and one-hot encoding over the documented category codes. Any undocumented code raises an error.
- **Models:** Logistic Regression, Random Forest and XGBoost, each in a single `Pipeline` with the preprocessor.

## Results (measured, seed 42)

Produced by `python -m src.train_models`. The full output, including CV standard deviations and confusion matrices, is in `artifacts/metrics.json`.

| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 5-fold CV | 0.839 | 0.877 | 0.766 | 0.814 | 0.907 |
| | Test | 0.869 | 0.812 | 0.929 | 0.867 | 0.958 |
| Random Forest | 5-fold CV | 0.826 | 0.831 | 0.783 | 0.802 | 0.895 |
| | Test | 0.885 | 0.839 | 0.929 | 0.881 | 0.948 |
| XGBoost | 5-fold CV | 0.806 | 0.805 | 0.765 | 0.782 | 0.867 |
| | Test | 0.918 | 0.871 | 0.964 | 0.915 | 0.947 |

The test set has only 61 records, so one prediction changes accuracy by about 1.6 points. Model selection therefore relies on the cross-validation scores, not on the test scores.

## Final model: Logistic Regression

Logistic Regression is the prototype model because:

- It has the best 5-fold cross-validation ROC-AUC (0.907), accuracy, precision and F1 of the three models.
- Its predictions are a weighted sum of the inputs, so it is the easiest to explain, which matters most for this explainable-AI product.

XGBoost has the highest test accuracy. That score comes from a single 61-record split, and XGBoost has the lowest cross-validation scores, so it was not chosen on that basis.

The other two models are still saved to `models/` and reported in `artifacts/metrics.json` as evaluation artifacts.

## Prediction pipeline

```
PatientInput (validated dataclass)
      │  to_frame(): one row, columns named explicitly
      ▼
models/final_model.joblib ── Pipeline[ preprocessor → LogisticRegression ]
      │  predict_proba
      ▼
PredictionResult(predicted_class, probability_positive, probability_negative, model_name, inputs)
```

- **Persistence:** `models/final_model.joblib` is a dictionary holding:
  - the whole fitted scikit-learn `Pipeline` (imputers, scaler, one-hot encoder and classifier);
  - the feature list and category codes it was trained with;
  - its test metrics and the library versions.

  The preprocessing and the model are saved as one object, so a different preprocessor cannot be paired with the model by mistake. This is the same pipeline object that was scored on the test set, and a test checks that.
- **Training and inference:** `src/train_models.py` fits the pipeline on the training split and saves it. `src/prediction.py` only loads and applies it, and never retrains.
- **Artifact checks:** `load_model()` raises `ModelArtifactError` and tells you how to fix it if the file is missing or corrupted, has no preprocessing step, or was built for a different feature list.
- **Output:** `predicted_class` uses the default 0.5 threshold and refers to the dataset target: 1 = disease present, according to the 1988 angiography label. This is raw model output. Any user-facing risk wording added later will be prototype bands, not clinical thresholds.

### Input contract (`PatientInput`)

- All 13 features are required. The model never receives a missing value; if a test result is unknown, the input is rejected.
- Values must be numbers. Strings, booleans and NaN are rejected.
- Categorical fields and `ca` must be whole numbers, and categorical fields must be one of the documented codes listed in the dataset table above.
- Numeric fields must fall within sanity bounds:

  | Field | Bounds |
  |---|---|
  | age | 18–100 |
  | trestbps | 80–220 |
  | chol | 100–600 |
  | thalach | 60–220 |
  | oldpeak | 0–7 |
  | ca | 0–3 |

  These bounds catch typos and impossible values. They are **not** clinical limits.
- Every problem found is reported together in an `InvalidInputError`, which has one readable message per field in `.errors`.

### Predict from Python

```python
from src.prediction import PatientInput, predict, DEMO_INPUTS

patient = PatientInput(age=56, sex=1, cp=2, trestbps=134, chol=245, fbs=0, restecg=2,
                       thalach=148, exang=0, oldpeak=1.2, slope=2, ca=1, thal=3)
result = predict(patient)            # or PatientInput.from_dict({...})
print(result.probability_positive)
```

`DEMO_INPUTS` contains three **synthetic** example inputs (A, B, C). They are written by hand and are not real patients or rows from the dataset; their predictions are always computed by the model.

To check the saved artifact and score the demo inputs:

```bash
python -m src.prediction
```

To explain the demo inputs and print global importance:

```bash
python -m src.explainability
```

## Explainable AI (SHAP)

A **prediction** answers "what does the model estimate for this input?". An **explanation** answers "which inputs moved that estimate, and by how much?". `src/prediction.py` produces the prediction, and `src/explainability.py` explains it. The two modules are separate, but they use the same saved model and the same `PatientInput` validation.

**SHAP** (SHapley Additive exPlanations) splits one prediction into one contribution per feature. It is based on Shapley values from game theory. SHAP values are additive:

```
base value + sum of all feature contributions = model output
```

- **Explainer:** `shap.LinearExplainer` is SHAP's explainer for linear models. It is exact and fast. For each preprocessed column it computes `coefficient × (patient value − average training value)`.
- **Background:** the explainer's reference data is the 242 preprocessed training records, all of them, with no subsampling.
- **Units:** the values are in **log-odds**, which is what logistic regression computes before converting to a probability. They are not percentages. The base value is −0.071 log-odds: the model's output at the average training record, which converts to a probability of 0.482. Converting `base + Σ contributions` to a probability gives exactly the probability returned by `predict()`.
- **Local explanation:** `explain(patient)` returns a `LocalExplanation`. It contains one `FeatureContribution` per original feature, sorted from largest effect to smallest. Each contribution has:
  - the readable label and the patient's input value;
  - the SHAP value;
  - its direction (`toward_positive`, `toward_negative` or `neutral`);
  - its share of the total effect;
  - a careful sentence describing it.
- **Global explanation:** `default_explainer().global_importance()` ranks the features by mean |SHAP| across all 303 records.
- **Readable feature names:** preprocessing turns the 13 features into 22 columns, because one-hot encoding splits each categorical feature into several columns. For each feature, the explainer reads from the fitted encoder how many columns it produced, and checks this against the column names. It then **adds up** each feature's column contributions. Adding is valid because SHAP values are additive, so "Chest pain type" is shown as one contribution, not four encoded columns. Readable labels come from `FEATURE_LABELS` and `CATEGORY_LABELS` in `src/preprocessing.py`.

Global importance of the final model (mean |SHAP|, log-odds, all 303 records, from `python -m src.explainability`):

| Rank | Feature | Mean \|SHAP\| | Share |
|---|---|---|---|
| 1 | Major vessels coloured by fluoroscopy (`ca`) | 0.985 | 20.7% |
| 2 | Thallium stress test (`thal`) | 0.651 | 13.7% |
| 3 | Chest pain type (`cp`) | 0.637 | 13.4% |
| 4 | Sex | 0.560 | 11.8% |
| 5 | Slope of peak exercise ST segment | 0.508 | 10.7% |
| 6 | Maximum heart rate | 0.289 | 6.1% |
| 7 | Resting blood pressure | 0.255 | 5.4% |
| 8 | Exercise-induced angina | 0.239 | 5.0% |
| 9 | Resting ECG | 0.198 | 4.2% |
| 10 | ST depression | 0.170 | 3.6% |
| 11 | Cholesterol | 0.133 | 2.8% |
| 12 | Age | 0.076 | 1.6% |
| 13 | Fasting blood sugar | 0.050 | 1.1% |

```python
from src.explainability import explain, default_explainer
from src.prediction import DEMO_INPUTS

e = explain(DEMO_INPUTS["Example Patient C"])
for c in e.contributions[:3]:
    print(c.label, c.display_value, round(c.shap_value, 3), c.text)
ranking = default_explainer().global_importance()
```

**SHAP explains the model, not medicine.** A contribution means the feature moved *this model's* output for this input. It is not evidence that the feature causes disease. Wording used: "Cholesterol (245) contributed toward a higher model-estimated probability for this input". It never says "cholesterol caused…".

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

This regenerates every model artifact. It writes `models/final_model.joblib`, the per-model pipelines `models/{logistic_regression,random_forest,xgboost}.joblib`, and the metrics in `artifacts/metrics.json`. Model files are not committed; run this once after cloning. The seed is fixed, so the results are identical each time.

## Test

```bash
python -m pytest -q
```

## Project structure

```
app.py                  Streamlit app (Phase 5)
data/                   UCI Cleveland data file
models/                 Trained pipelines, incl. final_model.joblib (generated)
artifacts/metrics.json  Measured evaluation results
src/data_loader.py      Download, checksum and load the data; binarize the target
src/preprocessing.py    Feature lists, category codes, preprocessing pipeline
src/evaluate_models.py  Metrics
src/train_models.py     Training, cross-validation, test evaluation, saving
src/prediction.py       Input contract, artifact loading/checks, inference, demo inputs
src/explainability.py   SHAP local explanations and global importance
tests/                  Automated tests
docs/methodology.md     Methodology notes
```

## Limitations

- The dataset is small (303 records), comes from a single centre, dates from 1988, and is 68% male. Results may not generalize to other populations.
- The input bounds only check that values are plausible. Values outside the training data's observed range (see `src/prediction.py`) are accepted but are extrapolation.
- The model files are pickles tied to the pinned scikit-learn version; retrain after upgrading it.
- SHAP explanations describe how this model behaves on this dataset. They do not show causation. With correlated features (for example `thal` and `ca`), credit can be split between them in ways that do not match their clinical meaning. Global importance reflects this 303-record population only.
- The target is a historical angiographic label. It is not a forecast of future cardiac events.
- Predictions use a 0.5 threshold on the dataset target. Any risk categories added later will be prototype bands derived from model probability. They are not clinically validated thresholds.
