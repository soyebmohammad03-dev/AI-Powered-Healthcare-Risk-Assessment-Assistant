# Methodology

## Data
**Source.** The data is the Cardiovascular Disease dataset by Svetlana Ulianova, published on Kaggle as `sulianova/cardiovascular-disease-dataset`. The file is `cardio_train.csv`. It is downloaded from Kaggle's public API endpoint and checked against a SHA-256 hash, and it is not redistributed in this repository.

**What the raw file contains.** It was inspected before any processing:
- 70,000 rows and 13 columns, separated by semicolons.
- No missing values and no duplicate ids.
- 24 records that are identical apart from `id`.
- A balanced binary target, `cardio`: 35,021 absent and 34,979 present.
- Clear measurement errors. Blood pressure ranges from −150 to 16,020 mm Hg, height from 55 to 250 cm, and weight goes down to 10 kg. 1,234 rows have systolic pressure below diastolic.

**Cleaning.** The cleaning rules are data-quality rules. They remove values that cannot be real measurements; they are not medical thresholds and were not tuned to improve model scores. They are applied in this order, and the number of rows removed by each is recorded in `artifacts/metrics.json`:
1. Malformed, missing or non-numeric values (0 rows).
2. Target not 0 or 1 (0 rows).
3. Undocumented category codes (0 rows).
4. Duplicate records, ignoring `id` (24 rows). These are removed before the split so identical records cannot appear in both the training and test sets.
5. Age outside 29–65 years (0 rows).
6. Height outside 120–220 cm (53 rows).
7. Weight outside 30–250 kg (7 rows).
8. Systolic outside 60–250 mm Hg (226 rows).
9. Diastolic outside 30–200 mm Hg (989 rows; most are values like 1000 or 1100).
10. Systolic not above diastolic (103 rows).
11. BMI outside 12–60 (25 rows).

The result is 68,573 records: 34,646 absent and 33,927 present. The same ranges are used to validate user input, so the model is never asked about values its training data excluded.

**Features.**
- `id` is dropped; it is an identifier, never a feature.
- `age` is converted from days to years (days / 365.25).
- `gender` uses codes 1 and 2, which the source does not label. 1 = female is inferred from mean height (161 cm vs 170 cm).
- `cholesterol` and `gluc` are three-level categories: normal, above normal and well above normal.
- `smoke`, `alco` and `active` are self-reported 0/1 values.

**Derived feature.** BMI = weight / (height in m)² is the only derived feature, for two reasons:
- It is a standard, interpretable way to combine height and weight.
- In a check on the training split, adding raw height and weight back alongside BMI changed 5-fold ROC-AUC by at most 0.0002.

Pulse pressure (`ap_hi − ap_lo`) was considered and rejected. It is an exact linear combination of two existing features, so it adds no information and would make the linear model's SHAP credit arbitrary. The same check showed no gain.

## Preprocessing
Preprocessing is a single scikit-learn `Pipeline` stored inside each model pipeline. It derives BMI, standard-scales the numeric features (age, BMI, `ap_hi`, `ap_lo`), and one-hot encodes the categorical features. The encoder uses the fixed list of documented codes, and an unknown code raises an error.

Three things prevent information leaking from test data into training:
- Duplicates are removed before the split.
- The split is stratified 80/20 with seed 42 (54,858 train and 13,715 test records).
- Every learned step (scaler, encoder) is fitted on training data only. During cross-validation it is fitted inside each training fold. A test checks that the scaler's means equal the training-split means.

## Evaluation
1. Stratified 5-fold cross-validation on the training set compares the three models.
2. Each model is then refitted on the full training set and scored once on the held-out test set.

Metrics: accuracy, precision, recall, F1, ROC-AUC and the confusion matrix. All three models have a CV standard deviation of 0.01 or less on every metric, and each model's test scores agree with its CV scores.

## Model selection
| Model | CV ROC-AUC | CV accuracy | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.791 ± 0.004 | 0.728 | 0.793 |
| Random Forest | 0.800 ± 0.004 | 0.733 | 0.803 |
| XGBoost | 0.801 ± 0.004 | 0.735 | 0.804 |

Logistic Regression is the final prototype model. The tree models are better by about 0.01 ROC-AUC and less than 1 point of accuracy. The models are equally stable, so this gap is the only trade-off.

For an explainable decision-support product, interpretability is weighted above this gap:
- Logistic Regression's effects are monotone and global. A higher systolic pressure always raises its estimate.
- Its SHAP values are exact and have a closed form, `coef × (x − mean)`, which can be checked by hand.

The decision was made on cross-validation, not on the test set. The reasoning is also saved in `artifacts/metrics.json`. Switching to XGBoost would need SHAP's `TreeExplainer`, which is not implemented.

The final model is the pipeline fitted on the training split. It is not refitted on all of the data, so its reported test metrics describe exactly the saved model.

## Persistence and inference
`models/final_model.joblib` stores:
- the fitted pipeline: BMI derivation → scaler and encoder → classifier;
- the input and model feature lists;
- the category codes;
- the seed, the test metrics and the library versions.

`src/prediction.py` loads this file, checks it against the current schema, validates the input through `PatientInput`, and calls `predict_proba`. Because the preprocessing used at inference is the training preprocessing object itself, the two cannot disagree. Tests confirm two things:
- the saved model reproduces its stored test metrics exactly;
- its preprocessing matches a freshly fitted preprocessor on the same training split.

## Explainability (SHAP)
**Explainer.** `shap.LinearExplainer` is applied to the classifier's 14 preprocessed columns. It uses an `Independent` masker whose background is all 54,858 preprocessed training records; the default subsample of 100 is disabled. Each column's SHAP value is `coef_j × (x_j − mean_j)`, and a test recomputes this formula without SHAP.

**Units.** The values are log-odds. The base value is +0.032, the model's output at the mean training record, which corresponds to a probability of 0.508. For every input, `sigmoid(base + Σ SHAP)` equals the probability returned by `predict()`.

**Mapping columns back to features.**
- Numeric features map one-to-one.
- For each categorical feature, the explainer reads its number of one-hot columns from the fitted encoder (`categories_`, `drop_idx_`) and checks this against `get_feature_names_out()`.
- The SHAP values of those columns are summed, which is exact because SHAP is additive.
- Height and weight reach the model only through BMI, so the explanation reports "BMI (from height and weight)".

**Local and global explanations.**
- **Local:** the per-feature contributions for one input, ranked by size, each with a direction and its share of the total.
- **Global:** the mean absolute contribution per feature across all 68,573 cleaned records. Columns are grouped into features before absolute values are taken.

**Limitations.**
- These values describe how the model weighted the input features. They do not describe medical causation.
- Correlated features can split credit unintuitively.
- The model has learned dataset-specific patterns. Smokers and drinkers show slightly lower disease rates in this data, so Smoking = Yes receives a small negative contribution. This is confounded (for example, smoking is far more common in gender code 2) and must not be read as a health effect.

## Informational guidance
The system has three roles:
- **ML model = prediction**
- **SHAP = explanation**
- **Recommendation engine = general informational guidance**

**Why rules, not an LLM.** The guidance uses explicit, deterministic rules. Safety-relevant text must be predictable, inspectable and testable. An LLM can generate unsupported medical statements and varies between runs.

**What the rules use.** The rules read only the current schema: blood pressure, the cholesterol and glucose categories, smoking, BMI, physical activity and alcohol, plus the model's probability band. They do not use SHAP values to decide what to show. This is deliberate: when the model has learned a confounded pattern, such as for smoking, the guidance still follows the input itself.

**Prototype bands.** Probability < 0.30 is lower, 0.30 to < 0.60 is moderate, and ≥ 0.60 is higher. These are presentation categories, not clinically validated thresholds, and the model's 0.5 classification threshold is unchanged.

**Rule triggers.** These are prompts to discuss a value with a professional, not diagnostic cut-offs:
- blood pressure of 130/80 or more (where the ACC/AHA 2017 elevated/stage-1 range begins);
- cholesterol or glucose reported as above normal (the dataset's own category);
- BMI below 18.5 or 25 and above (WHO adult categories);
- smoking, alcohol intake, and physical activity as reported.

**How items are shown.** The follow-up item comes first, then at most 3 input items by priority, then the model-context note. The remaining triggered items are kept in `Guidance.additional`.

**Safety.**
- No item states a condition, says the user is healthy, or names medication or treatment.
- A lower estimate is described as "does not rule out any health condition".
- A shared `DISCLAIMER` accompanies every result.
- Tests sweep every band and trigger combination for prohibited wording.
