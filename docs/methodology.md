# Methodology

## Data
The data is the UCI Heart Disease Cleveland subset: 303 records, each with 13 clinical features. The dataset's documentation lists 14 attributes; the 13 features are those attributes minus the original target `num`. That target takes values 0 (no disease) and 1–4 (disease present; the levels relate to angiographic narrowing). Following the dataset's own description of the standard task, it is binarized as presence (1–4) versus absence (0).

The data was inspected before any preprocessing:
- All categorical codes in the file match the documentation.
- `restecg = 1` is rare (4 records).
- `ca` has 4 missing values and `thal` has 2, both recorded as `?`.
- There are no other missing values.

## Preprocessing
All preprocessing runs inside one scikit-learn `Pipeline` per model, so imputation and scaling are fitted only on training folds. This prevents information from the test data leaking into training.
- Numeric features (age, trestbps, chol, thalach, oldpeak, ca) use median imputation followed by standard scaling. `ca` counts vessels from 0 to 3, so it is treated as ordinal.
- Categorical features (sex, cp, fbs, restecg, exang, slope, thal) use most-frequent imputation followed by one-hot encoding. The encoding uses a fixed list of documented codes. Binary features are encoded as a single column, and an unknown code raises an error.

## Evaluation
1. The data is split 80/20 into train and test sets, stratified by class, with seed 42.
2. Stratified 5-fold cross-validation on the training set is used to compare models.
3. Each model is then refitted on the full training set and scored once on the held-out test set.

Metrics reported: accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix. Measured values are in `artifacts/metrics.json` and the README.

## Model selection
Logistic Regression was chosen as the final prototype model. It has the best 5-fold cross-validation ROC-AUC (0.907 ± 0.018), accuracy, precision and F1 of the three models. It is also directly interpretable: each feature contributes through one learned coefficient, which suits an explainable healthcare prototype.

The model was not chosen on the held-out test set. That set has only 61 records, so a single prediction moves accuracy by about 1.6 points. Random Forest and XGBoost remain saved and reported as comparison artifacts.

The final model is the pipeline fitted on the 242 training records. It is not refitted on all 303 records, so the test metrics reported for it describe exactly the saved model.

## Persistence and inference
Training (`src/train_models.py`) saves `models/final_model.joblib`, a dictionary containing:
- the fitted `Pipeline` (preprocessor and classifier),
- the feature list,
- the category codes,
- the seed,
- the test metrics,
- the library versions.

Inference (`src/prediction.py`) is a separate module. It never trains anything. It loads the dictionary, checks it against the current feature contract, and calls `predict_proba` on the same pipeline object. This means the preprocessing at inference time is the training preprocessing, not a re-implementation of it. Tests confirm two things:
- The persisted pipeline reproduces the stored test metrics exactly.
- Its preprocessor produces the same output as a freshly fitted preprocessor on the same training split.

Training data and inference input are handled differently in two ways:
- **Missing values:** the training data had 6 missing values (`ca`, `thal`), which the pipeline imputes. At inference, every field is required, so imputation is never relied on for a user's own value.
- **Input validation:** inference input is checked by the `PatientInput` contract. Values must be numeric, categorical codes must be documented ones, and numeric values must fall within sanity bounds before they reach the model.

The `predicted_class` uses a 0.5 probability threshold on the dataset target. No clinical risk thresholds are defined in this layer.
