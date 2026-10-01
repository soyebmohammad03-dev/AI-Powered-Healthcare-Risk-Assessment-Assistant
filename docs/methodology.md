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

## Explainability (SHAP)
**Why.** A probability on its own is not enough for a human-centred healthcare prototype. A user also needs to see which inputs moved the model's estimate. SHAP provides this. It is grounded in Shapley values from game theory, and its contributions are additive: the base value plus the sum of all contributions equals the model output exactly.

**Explainer.** The final model is a logistic regression, so `shap.LinearExplainer` is used. It is SHAP's model-specific explainer for linear models.
- It explains the classifier on its real inputs, which are the 22 preprocessed columns.
- It uses an `Independent` masker whose background is all 242 preprocessed training records. The default would subsample 100 of them; this is disabled.
- With this setup, each column's SHAP value is `coef_j × (x_j − mean_j)`. A test recomputes this formula without SHAP and checks that the values match.

**Output space.** The values are log-odds, which is the scale of `decision_function`, not probabilities.
- The base value (−0.071 log-odds) is the model's output at the mean of the preprocessed training data. Converted to a probability it is 0.482.
- For any input, `sigmoid(base + Σ SHAP)` equals the probability returned by the prediction service. Tests check this.
- Contributions should therefore be read as "pushed the estimate up" or "pushed the estimate down", not as percentage points.

**Mapping columns back to features.** The preprocessing turns each numeric feature into one scaled column. Each categorical feature becomes one column per category, except binary features, which become a single column.
- The explainer reads each feature's column count from the fitted `OneHotEncoder` (`categories_`, `drop_idx_`).
- It checks the result against `get_feature_names_out()`.
- It sums the column SHAP values belonging to each original feature. Summing is exact because SHAP is additive.
- Each feature's sum is reported once, with its readable label and the patient's original value. For example, "Chest pain type: Asymptomatic" replaces `cat__cp_4.0`.

**Local and global explanations.**
- **Local:** one patient's per-feature contributions, ranked by size. Each contribution also gets a direction and a share of the total absolute contribution.
- **Global:** the mean absolute grouped SHAP value per feature across all 303 records, ranked. Rows are grouped by feature before taking absolute values, so a feature's one-hot columns cannot cancel each other or be double counted.

**Limitations.**
- SHAP explains the trained model's behaviour, not human physiology. A large contribution is not evidence of medical causation.
- With an independent background, correlated features (for example `ca`, `thal`, `exang`, `oldpeak`) are treated as if they varied separately. Credit can therefore be shared among them in clinically unintuitive ways.
- Explanations inherit every limitation of the model and the dataset.
- User-facing wording therefore always says that a feature "contributed toward the model's estimate".
