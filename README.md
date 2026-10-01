# AI-Powered Healthcare Risk Assessment Assistant

Academic project for **Design of Artificial Intelligence Products** (B.Tech).
Theme: Healthcare + Human-Centered AI + Explainable AI.

> **Medical disclaimer:** This is an educational AI decision-support prototype. It does **not** diagnose heart disease, prescribe treatment, or replace a doctor, and it is not medical advice. Its outputs come from a model trained on a small historical research dataset and are not clinically validated.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Dataset acquisition, inspection, preprocessing, baseline training & evaluation | Done |
| 2 | Model selection, persistence, prediction pipeline | Planned |
| 3 | SHAP explainability | Planned |
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

The test set has only 61 records, so one prediction changes accuracy by about 1.6 points. Model selection (Phase 2) therefore relies on the cross-validation scores, not on the test scores.

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

This writes the fitted pipelines to `models/*.joblib` (not committed) and the metrics to `artifacts/metrics.json`.

## Test

```bash
python -m pytest -q
```

## Project structure

```
app.py                  Streamlit app (Phase 5)
data/                   UCI Cleveland data file
models/                 Trained pipelines (generated)
artifacts/metrics.json  Measured evaluation results
src/data_loader.py      Download, checksum and load the data; binarize the target
src/preprocessing.py    Feature lists, category codes, preprocessing pipeline
src/evaluate_models.py  Metrics
src/train_models.py     Training, cross-validation, test evaluation, saving
tests/                  Automated tests
docs/methodology.md     Methodology notes
```

## Limitations

- The dataset is small (303 records), comes from a single centre, dates from 1988, and is 68% male. Results may not generalize to other populations.
- The target is a historical angiographic label. It is not a forecast of future cardiac events.
- Any risk categories added later will be prototype bands derived from model probability. They are not clinically validated thresholds.
