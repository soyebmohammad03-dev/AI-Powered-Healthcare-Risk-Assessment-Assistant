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
