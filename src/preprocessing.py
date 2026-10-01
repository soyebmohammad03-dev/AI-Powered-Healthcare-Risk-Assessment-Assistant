"""Feature schema, plausibility ranges, derived features and the preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# What the user/dataset provides (age already converted from days to years by the data loader).
FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active"]

# What the classifier sees. BMI replaces height and weight (adding them back changed CV ROC-AUC by
# <= 0.0002); pulse pressure (ap_hi - ap_lo) is not added because it is an exact linear combination.
NUMERIC = ["age", "bmi", "ap_hi", "ap_lo"]
# Codes as distributed with the dataset. gender: 1 vs 2 is not labelled in the source; 1 = female is
# inferred from mean height (161 cm vs 170 cm), the usual reading of this dataset.
CATEGORIES = {
    "gender": [1, 2],
    "cholesterol": [1, 2, 3],   # normal, above normal, well above normal
    "gluc": [1, 2, 3],          # normal, above normal, well above normal
    "smoke": [0, 1],
    "alco": [0, 1],
    "active": [0, 1],
}
CATEGORICAL = list(CATEGORIES)
MODEL_FEATURES = NUMERIC + CATEGORICAL

# Data-quality plausibility ranges (inclusive), shared by dataset cleaning and input validation.
# They remove recording errors (e.g. ap_hi = 16020, height = 55 cm); they are NOT clinical thresholds.
PLAUSIBLE = {
    "age": (29, 65),        # years; the range the dataset covers (29.6-64.9), so no extrapolation
    "height": (120, 220),   # cm
    "weight": (30, 250),    # kg
    "ap_hi": (60, 250),     # mm Hg
    "ap_lo": (30, 200),     # mm Hg; additionally ap_hi must exceed ap_lo
    "bmi": (12, 60),        # catches height/weight combinations that are each plausible but not together
}

FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "height": "Height",
    "weight": "Weight",
    "bmi": "BMI (from height and weight)",
    "ap_hi": "Systolic blood pressure",
    "ap_lo": "Diastolic blood pressure",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol intake",
    "active": "Physical activity",
}
LEVELS = {1: "Normal", 2: "Above normal", 3: "Well above normal"}
CATEGORY_LABELS = {
    "gender": {1: "Female", 2: "Male"},
    "cholesterol": LEVELS,
    "gluc": LEVELS,
    "smoke": {0: "No", 1: "Yes"},
    "alco": {0: "No", 1: "Yes"},
    "active": {0: "No", 1: "Yes"},
}


def bmi(height_cm, weight_kg):
    return weight_kg / (height_cm / 100) ** 2


def add_bmi(X: pd.DataFrame) -> pd.DataFrame:
    return X.assign(bmi=bmi(X["height"], X["weight"]))


def plausible_mask(X: pd.DataFrame) -> pd.Series:
    """Vectorised version of the input-domain rules (ranges, systolic > diastolic, plausible BMI)."""
    ok = X["ap_hi"] > X["ap_lo"]
    for col, (low, high) in PLAUSIBLE.items():
        values = bmi(X["height"], X["weight"]) if col == "bmi" else X[col]
        ok &= values.between(low, high)
    return ok


def build_preprocessor() -> Pipeline:
    """Raw FEATURES -> add BMI -> scale numerics, one-hot categoricals (height/weight dropped).

    Lives inside the persisted model pipeline, so training and inference derive features identically,
    and every learned step (scaler, encoder) is fit on training data only.
    """
    encode = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(categories=list(CATEGORIES.values()), drop="if_binary", sparse_output=False),
         CATEGORICAL),
    ])
    return Pipeline([("derive", FunctionTransformer(add_bmi)), ("encode", encode)])
