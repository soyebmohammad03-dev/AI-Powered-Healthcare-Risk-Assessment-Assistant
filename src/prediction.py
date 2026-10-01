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

from src.data_loader import ROOT
from src.preprocessing import CATEGORIES, FEATURES

FINAL_MODEL_PATH = ROOT / "models" / "final_model.joblib"

# Input sanity bounds, not clinical limits. They reject typos and impossible values while staying
# a little wider than the training data (observed: age 29-77, trestbps 94-200, chol 126-564,
# thalach 71-202, oldpeak 0-6.2, ca 0-3).
NUMERIC_BOUNDS = {
    "age": (18, 100),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 7.0),
    "ca": (0, 3),
}
INTEGER_FEATURES = {"ca"} | set(CATEGORIES)


class InvalidInputError(ValueError):
    """Raised with every validation problem found, one per line, suitable for showing to a user."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


class ModelArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class PatientInput:
    """All 13 features are required, so the model never receives an unset value.
    Categorical fields use the dataset's own integer codes (see preprocessing.CATEGORIES)."""
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    def __post_init__(self):
        errors = []
        for name in FEATURES:
            value = getattr(self, name)
            if value is None:
                errors.append(f"{name}: value is required.")
            elif isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value):
                errors.append(f"{name}: expected a number, got {value!r}.")
            elif name in INTEGER_FEATURES and value != int(value):
                errors.append(f"{name}: expected a whole number, got {value!r}.")
            elif name in CATEGORIES and float(value) not in CATEGORIES[name]:
                allowed = ", ".join(str(int(c)) for c in CATEGORIES[name])
                errors.append(f"{name}: {value!r} is not a valid code (allowed: {allowed}).")
            elif name in NUMERIC_BOUNDS and not NUMERIC_BOUNDS[name][0] <= value <= NUMERIC_BOUNDS[name][1]:
                low, high = NUMERIC_BOUNDS[name]
                errors.append(f"{name}: {value!r} is outside the accepted range {low}-{high}.")
        if errors:
            raise InvalidInputError(errors)

    @classmethod
    def from_dict(cls, data: dict) -> "PatientInput":
        expected = {f.name for f in fields(cls)}
        missing, unknown = expected - data.keys(), data.keys() - expected
        errors = [f"{n}: value is required." for n in sorted(missing)]
        errors += [f"{n}: unknown field." for n in sorted(unknown)]
        if errors:
            raise InvalidInputError(errors)
        return cls(**data)

    def to_frame(self) -> pd.DataFrame:
        # Columns are named explicitly, so feature order in the caller can never shift meaning.
        return pd.DataFrame([{name: float(getattr(self, name)) for name in FEATURES}], columns=FEATURES)


@dataclass(frozen=True)
class PredictionResult:
    """Raw model output. Translating probability into user-facing wording is the UI's job."""
    predicted_class: int          # 1 = dataset target "disease present", 0 = absent (threshold 0.5)
    probability_positive: float
    probability_negative: float
    model_name: str
    inputs: dict


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
    if (bundle.get("features") != FEATURES or bundle.get("categories") != CATEGORIES
            or list(getattr(pipeline, "feature_names_in_", [])) != FEATURES):
        raise ModelArtifactError(f"Model artifact at {path} does not match the current feature contract. {hint}")
    return bundle


@lru_cache(maxsize=1)
def _default_model() -> dict:
    return load_model()


def predict(patient: PatientInput, model: dict | None = None) -> PredictionResult:
    model = model or _default_model()
    proba = model["pipeline"].predict_proba(patient.to_frame())[0]
    classes = list(model["pipeline"].classes_)
    p_pos = float(proba[classes.index(1)])
    return PredictionResult(
        predicted_class=int(p_pos >= 0.5),
        probability_positive=p_pos,
        probability_negative=1.0 - p_pos,
        model_name=model["model_name"],
        inputs=asdict(patient),
    )


# DEMO INPUTS / SYNTHETIC EXAMPLES: hand-written values for demonstrating the app.
# They are not real patients and not rows from the dataset. Their predictions are always computed.
DEMO_INPUTS = {
    "Example Patient A": PatientInput(age=42, sex=0, cp=3, trestbps=118, chol=210, fbs=0, restecg=0,
                                      thalach=172, exang=0, oldpeak=0.0, slope=1, ca=0, thal=3),
    "Example Patient B": PatientInput(age=56, sex=1, cp=2, trestbps=134, chol=245, fbs=0, restecg=2,
                                      thalach=148, exang=0, oldpeak=1.2, slope=2, ca=1, thal=3),
    "Example Patient C": PatientInput(age=63, sex=1, cp=4, trestbps=150, chol=290, fbs=1, restecg=2,
                                      thalach=118, exang=1, oldpeak=2.6, slope=2, ca=2, thal=7),
}


if __name__ == "__main__":
    bundle = load_model()
    print(f"Artifact OK: {FINAL_MODEL_PATH} ({bundle['model_name']}, {bundle['versions']})")
    for label, patient in DEMO_INPUTS.items():
        r = predict(patient, bundle)
        print(f"{label}: class={r.predicted_class} P(pos)={r.probability_positive:.4f} "
              f"P(neg)={r.probability_negative:.4f}")
