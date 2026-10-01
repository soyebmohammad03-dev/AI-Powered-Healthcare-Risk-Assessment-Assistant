"""Inference: validated patient input -> persisted model -> structured prediction.

Run `python -m src.prediction` to validate the saved artifact and score the demo inputs.
No training happens here; a missing or broken artifact is an error, never a silent retrain.
"""
import math
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost

from src.data_loader import ROOT
from src.preprocessing import CATEGORIES, FEATURE_LABELS, FEATURES, MODEL_FEATURES, PLAUSIBLE, bmi

FINAL_MODEL_PATH = ROOT / "models" / "final_model.joblib"


class InvalidInputError(ValueError):
    """Raised with every validation problem found, one per line, suitable for showing to a user."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


class ModelArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class PatientInput:
    """All 11 inputs are required. Numeric ranges are the same data-quality ranges used to clean the
    training data (preprocessing.PLAUSIBLE), so the model never sees values it was never trained on.
    Categorical fields use the dataset's integer codes (see preprocessing.CATEGORIES)."""
    age: float          # years
    gender: int         # 1 female, 2 male
    height: float       # cm
    weight: float       # kg
    ap_hi: float        # systolic blood pressure, mm Hg
    ap_lo: float        # diastolic blood pressure, mm Hg
    cholesterol: int    # 1 normal, 2 above normal, 3 well above normal
    gluc: int           # 1 normal, 2 above normal, 3 well above normal
    smoke: int          # 0/1
    alco: int           # 0/1
    active: int         # 0/1

    def __post_init__(self):
        errors = []
        for name in FEATURES:
            value, label = getattr(self, name), FEATURE_LABELS[name]
            if value is None:
                errors.append(f"{label}: value is required.")
            elif isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value):
                errors.append(f"{label}: expected a number, got {value!r}.")
            elif name in CATEGORIES and value not in CATEGORIES[name]:
                allowed = ", ".join(str(c) for c in CATEGORIES[name])
                errors.append(f"{label}: {value!r} is not a valid code (allowed: {allowed}).")
            elif name in PLAUSIBLE and not PLAUSIBLE[name][0] <= value <= PLAUSIBLE[name][1]:
                low, high = PLAUSIBLE[name]
                errors.append(f"{label}: {value!r} is outside the accepted range {low}-{high}.")
        if not errors:  # cross-field checks only once each field is individually valid
            if self.ap_hi <= self.ap_lo:
                errors.append("Systolic blood pressure must be higher than diastolic blood pressure.")
            low, high = PLAUSIBLE["bmi"]
            if not low <= self.bmi <= high:
                errors.append(f"Height and weight give a BMI of {self.bmi:.1f}, outside the accepted "
                              f"range {low}-{high}. Please check both values.")
        if errors:
            raise InvalidInputError(errors)

    @property
    def bmi(self) -> float:
        return float(bmi(self.height, self.weight))

    @classmethod
    def from_dict(cls, data: dict) -> "PatientInput":
        expected = {f.name for f in fields(cls)}
        missing, unknown = expected - data.keys(), data.keys() - expected
        errors = [f"{FEATURE_LABELS[n]}: value is required." for n in sorted(missing)]
        errors += [f"{n}: unknown field." for n in sorted(unknown)]
        if errors:
            raise InvalidInputError(errors)
        return cls(**data)

    def to_frame(self) -> pd.DataFrame:
        # Columns are named explicitly, so feature order in the caller can never shift meaning.
        row = {n: int(getattr(self, n)) if n in CATEGORIES else float(getattr(self, n)) for n in FEATURES}
        return pd.DataFrame([row], columns=FEATURES)


@dataclass(frozen=True)
class PredictionResult:
    """Raw model output. Translating probability into user-facing wording is not this module's job."""
    predicted_class: int          # 1 = cardiovascular disease present (dataset label), 0 = absent; threshold 0.5
    probability_positive: float   # P(cardio = 1)
    probability_negative: float   # P(cardio = 0)
    model_name: str
    inputs: dict                  # the validated input, as given
    bmi: float                    # derived from height and weight, as the model computes it


def load_model(path: Path = FINAL_MODEL_PATH) -> dict:
    """Load and check the persisted bundle. Raises ModelArtifactError with a fix hint on any problem."""
    hint = "Regenerate it with: python -m src.train_models"
    if not Path(path).exists():
        raise ModelArtifactError(f"Model artifact not found at {path}. {hint}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelArtifactError(f"Model artifact at {path} could not be loaded ({exc}). {hint}") from exc
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ModelArtifactError(f"Model artifact at {path} has no pipeline. {hint}")
    pipeline = bundle["pipeline"]
    if "pre" not in getattr(pipeline, "named_steps", {}):
        raise ModelArtifactError(f"Model artifact at {path} is missing its preprocessing step. {hint}")
    if (bundle.get("features") != FEATURES or bundle.get("model_features") != MODEL_FEATURES
            or bundle.get("categories") != CATEGORIES
            or list(getattr(pipeline, "feature_names_in_", [])) != FEATURES):
        raise ModelArtifactError(f"Model artifact at {path} does not match the current feature contract. {hint}")
    installed = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
    if bundle.get("versions") != installed:  # pickles are only reliable on the library versions that wrote them
        raise ModelArtifactError(f"Model artifact at {path} was built with {bundle.get('versions')}, but "
                                 f"{installed} is installed. {hint}")
    return bundle


@lru_cache(maxsize=1)
def default_model() -> dict:
    return load_model()


def predict(patient: PatientInput, model: dict | None = None) -> PredictionResult:
    model = model or default_model()
    proba = model["pipeline"].predict_proba(patient.to_frame())[0]
    classes = list(model["pipeline"].classes_)
    p_pos = float(proba[classes.index(1)])
    return PredictionResult(
        predicted_class=int(p_pos >= 0.5),
        probability_positive=p_pos,
        probability_negative=1.0 - p_pos,
        model_name=model["model_name"],
        inputs=asdict(patient),
        bmi=patient.bmi,
    )


# DEMO INPUTS / SYNTHETIC EXAMPLES: hand-written values for demonstrating the app.
# They are not real patients and not rows from the dataset. Their predictions are always computed.
DEMO_INPUTS = {
    "Example Patient A": PatientInput(age=38, gender=1, height=165, weight=60, ap_hi=115, ap_lo=75,
                                      cholesterol=1, gluc=1, smoke=0, alco=0, active=1),
    "Example Patient B": PatientInput(age=50, gender=2, height=175, weight=85, ap_hi=130, ap_lo=85,
                                      cholesterol=1, gluc=1, smoke=1, alco=0, active=1),
    "Example Patient C": PatientInput(age=61, gender=1, height=160, weight=88, ap_hi=160, ap_lo=100,
                                      cholesterol=3, gluc=2, smoke=0, alco=0, active=0),
}


if __name__ == "__main__":
    bundle = load_model()
    print(f"Artifact OK: {FINAL_MODEL_PATH} ({bundle['model_name']}, {bundle['versions']})")
    for label, patient in DEMO_INPUTS.items():
        r = predict(patient, bundle)
        print(f"{label}: class={r.predicted_class} P(pos)={r.probability_positive:.4f} "
              f"P(neg)={r.probability_negative:.4f} BMI={r.bmi:.1f}")
