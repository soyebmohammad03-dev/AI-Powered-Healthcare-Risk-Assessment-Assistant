"""What-if analysis: how the MODEL ESTIMATE responds when selected inputs change, others held fixed.

This is model sensitivity analysis, not a forecast of medical outcomes. Guardrails:
  * only ADJUSTABLE inputs may change (age, gender and height stay as assessed);
  * every scenario is rebuilt through PatientInput, so the same ranges and cross-field rules apply
    (systolic > diastolic, plausible BMI); an impossible scenario raises InvalidInputError.
The tree model's probability is piecewise constant, so small input changes can leave it unchanged.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from src.prediction import InvalidInputError, PatientInput, PredictionResult, default_model, predict
from src.preprocessing import FEATURE_LABELS, PLAUSIBLE

ADJUSTABLE = ("ap_hi", "ap_lo", "weight", "cholesterol", "gluc", "smoke", "alco", "active")
CURVE_STEPS = {"ap_hi": 2.0, "ap_lo": 2.0, "weight": 1.0}  # grid step in the input's own unit


@dataclass(frozen=True)
class WhatIfResult:
    baseline: PredictionResult
    scenario: PredictionResult
    changes: dict          # feature -> (baseline value, scenario value), only inputs that actually changed
    delta_pp: float        # scenario minus baseline probability, in percentage points
    single_effects: dict   # feature -> change in pp when ONLY that input is changed (they need not add up)


def build_scenario(patient: PatientInput, changes: dict) -> PatientInput:
    blocked = sorted(set(changes) - set(ADJUSTABLE))
    if blocked:
        raise InvalidInputError([f"{FEATURE_LABELS.get(n, n)}: cannot be changed in what-if analysis."
                                 for n in blocked])
    return replace(patient, **changes)  # re-runs every PatientInput check


def compare(patient: PatientInput, changes: dict, model: dict | None = None) -> WhatIfResult:
    scenario = build_scenario(patient, changes)
    base, new = predict(patient, model), predict(scenario, model)
    changed = {k: (getattr(patient, k), v) for k, v in changes.items() if v != getattr(patient, k)}
    single = {}
    for k, (_, v) in changed.items():
        try:
            single[k] = (predict(build_scenario(patient, {k: v}), model).probability_positive
                         - base.probability_positive) * 100
        except InvalidInputError:  # e.g. diastolic alone would exceed the baseline systolic
            single[k] = None
    return WhatIfResult(base, new, changed, (new.probability_positive - base.probability_positive) * 100, single)


def response_curve(patient: PatientInput, feature: str, model: dict | None = None) -> pd.DataFrame:
    """Model estimate across the accepted range of one numeric input, all other inputs fixed.
    Values that would make the input invalid (e.g. systolic <= diastolic) are left out."""
    if feature not in CURVE_STEPS:
        raise ValueError(f"No response curve for {feature!r}; choose from {tuple(CURVE_STEPS)}.")
    low, high = PLAUSIBLE[feature]
    valid = []
    for value in np.arange(low, high + CURVE_STEPS[feature] / 2, CURVE_STEPS[feature]):
        try:
            valid.append(build_scenario(patient, {feature: float(value)}))
        except InvalidInputError:
            continue
    pipeline = (model or default_model())["pipeline"]
    X = pd.concat([p.to_frame() for p in valid], ignore_index=True)
    return pd.DataFrame({"value": X[feature], "probability": pipeline.predict_proba(X)[:, 1]})


def changes_from(patient: PatientInput, values: dict) -> dict:
    """Keep only adjustable inputs whose value differs from the patient's."""
    current = asdict(patient)
    return {k: v for k, v in values.items() if k in ADJUSTABLE and v != current[k]}
