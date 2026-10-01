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
