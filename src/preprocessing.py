"""Feature definitions and the preprocessing pipeline shared by all models."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ca (number of major vessels, 0-3) is ordinal, so it is treated as numeric.
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Valid codes per the dataset documentation (heart-disease.names); all occur in the Cleveland data.
CATEGORIES = {
    "sex": [0.0, 1.0],               # 0 female, 1 male
    "cp": [1.0, 2.0, 3.0, 4.0],      # typical angina, atypical angina, non-anginal pain, asymptomatic
    "fbs": [0.0, 1.0],               # fasting blood sugar > 120 mg/dl
    "restecg": [0.0, 1.0, 2.0],      # normal, ST-T abnormality, left ventricular hypertrophy
    "exang": [0.0, 1.0],             # exercise-induced angina
    "slope": [1.0, 2.0, 3.0],        # upsloping, flat, downsloping
    "thal": [3.0, 6.0, 7.0],         # normal, fixed defect, reversible defect
}
CATEGORICAL = list(CATEGORIES)
FEATURES = NUMERIC + CATEGORICAL

# Human-readable names and code meanings (same source as above), for explanations and the UI.
FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Maximum heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thallium stress test",
}
CATEGORY_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}


def build_preprocessor() -> ColumnTransformer:
    # Imputers only matter for the 6 missing ca/thal values; they are fit on training data only.
    # Explicit categories: rare codes (restecg=1 has 4 rows) stay encoded even when a CV fold lacks them,
    # and an undocumented code raises instead of being silently zeroed.
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(categories=list(CATEGORIES.values()), drop="if_binary",
                                            sparse_output=False)),
         CATEGORICAL),
    ])
