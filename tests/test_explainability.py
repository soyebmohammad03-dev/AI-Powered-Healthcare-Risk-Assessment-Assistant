import ast
import math
from dataclasses import asdict

import numpy as np
import pytest

from src.data_loader import ROOT, load_dataset, split
from src.explainability import NEGATIVE, NEUTRAL, POSITIVE, ModelExplainer
from src.prediction import (DEMO_INPUTS, FINAL_MODEL_PATH, InvalidInputError, ModelArtifactError,
                            PatientInput, load_model, predict)
from src.preprocessing import CATEGORY_LABELS, FEATURE_LABELS, MODEL_FEATURES

VALID = asdict(DEMO_INPUTS["Example Patient B"])


@pytest.fixture(scope="module")
def bundle():
    if not FINAL_MODEL_PATH.exists():
        from src.train_models import main
        main()
    return load_model()


@pytest.fixture(scope="module")
def explainer(bundle):
    return ModelExplainer(bundle)


def test_explainer_initializes_on_persisted_model(explainer, bundle):
    assert explainer.pipeline is bundle["pipeline"]
    assert len(explainer.owners) == len(explainer.pre[-1].get_feature_names_out()) == 14
    assert set(explainer.owners) == set(MODEL_FEATURES)
    assert math.isfinite(explainer.base_value)


@pytest.mark.parametrize("label", list(DEMO_INPUTS))
def test_local_explanation_is_consistent_with_prediction(explainer, bundle, label):
    patient = DEMO_INPUTS[label]
    e, p = explainer.explain(patient), predict(patient, bundle)
    assert len(e.contributions) == len(MODEL_FEATURES)
    assert {c.feature for c in e.contributions} == set(MODEL_FEATURES)
    # Additivity: base + contributions == the logistic regression's own score (log-odds) ...
    assert e.base_value + sum(c.shap_value for c in e.contributions) == pytest.approx(e.model_output)
    Z = bundle["pipeline"].named_steps["pre"].transform(patient.to_frame())
    assert e.model_output == pytest.approx(explainer.lr.decision_function(Z)[0])
    # ... and calibrating that score reproduces the prediction service exactly
    assert e.probability_positive == pytest.approx(p.probability_positive)
    assert explainer.score_to_probability(e.model_output) == pytest.approx(p.probability_positive)
    assert sum(c.relative_importance for c in e.contributions) == pytest.approx(1.0)
    magnitudes = [abs(c.shap_value) for c in e.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_values_match_closed_form(explainer, bundle):
    """Independent check (no SHAP): for a linear model, SHAP_j = coef_j * (x_j - mean_j over background)."""
    pre, clf = bundle["pipeline"].named_steps["pre"], explainer.lr  # the LR inside the calibrator
    background = pre.transform(split(load_dataset())[0])
    patient = DEMO_INPUTS["Example Patient C"]
    per_column = clf.coef_[0] * (pre.transform(patient.to_frame())[0] - background.mean(axis=0))
    expected = {f: per_column[[o == f for o in explainer.owners]].sum() for f in MODEL_FEATURES}
    got = {c.feature: c.shap_value for c in explainer.explain(patient).contributions}
    assert got == pytest.approx(expected)
    assert explainer.base_value == pytest.approx(clf.intercept_[0] + clf.coef_[0] @ background.mean(axis=0))


def test_bmi_explained_instead_of_height_and_weight(explainer):
    features = {c.feature for c in explainer.explain(DEMO_INPUTS["Example Patient C"]).contributions}
    assert "bmi" in features and not {"height", "weight"} & features


def test_readable_names_values_and_wording(explainer):
    patient = DEMO_INPUTS["Example Patient C"]
    for c in explainer.explain(patient).contributions:
        assert c.label == FEATURE_LABELS[c.feature] and "__" not in c.label
        assert c.value == pytest.approx(getattr(patient, c.feature))  # bmi: derived from height/weight
        if c.feature in CATEGORY_LABELS:
            assert c.display_value == CATEGORY_LABELS[c.feature][int(c.value)]
        assert c.direction == (POSITIVE if c.shap_value > 0 else NEGATIVE if c.shap_value < 0 else NEUTRAL)
        assert c.label in c.text and "caus" not in c.text.lower() and "diagnos" not in c.text.lower()


def test_local_explanation_is_deterministic(explainer, bundle):
    patient = PatientInput(**VALID)
    assert explainer.explain(patient) == explainer.explain(patient) == ModelExplainer(bundle).explain(patient)


def test_global_importance(explainer):
    ranked = explainer.global_importance()
    assert {g.feature for g in ranked} == set(MODEL_FEATURES)
    values = [g.mean_abs_shap for g in ranked]
    assert all(math.isfinite(v) and v >= 0 for v in values)
    assert values == sorted(values, reverse=True)
    assert sum(g.relative_importance for g in ranked) == pytest.approx(1.0)
    assert all(g.label == FEATURE_LABELS[g.feature] for g in ranked)


def test_invalid_input_rejected_same_as_prediction(explainer):
    with pytest.raises(InvalidInputError, match="Cholesterol"):
        explainer.explain(PatientInput(**{**VALID, "cholesterol": 9}))
    with pytest.raises(TypeError):
        explainer.explain(VALID)  # raw dict bypasses validation, so it is refused


def test_non_linear_model_rejected(bundle):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    fake = {**bundle, "pipeline": Pipeline([("pre", bundle["pipeline"].named_steps["pre"]),
                                            ("clf", RandomForestClassifier())])}
    with pytest.raises(ModelArtifactError, match="LogisticRegression"):
        ModelExplainer(fake)


def test_no_hardcoded_shap_numbers():
    """The module must not contain float literals that could stand in for computed SHAP values."""
    tree = ast.parse((ROOT / "src" / "explainability.py").read_text())
    floats = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    assert floats <= {0.0, 1.0}


def test_calibration_is_monotone_and_explained_separately(explainer):
    """Isotonic calibration never reverses a contribution's direction: a higher score never gives a
    lower probability."""
    scores = np.linspace(-6, 6, 400)
    probs = [explainer.score_to_probability(z) for z in scores]
    assert all(b >= a for a, b in zip(probs, probs[1:]))
    assert 0 <= probs[0] <= probs[-1] <= 1
    e = explainer.explain(DEMO_INPUTS["Example Patient C"])
    assert e.uncalibrated_probability == pytest.approx(1 / (1 + math.exp(-e.model_output)))
    assert explainer.base_probability == pytest.approx(explainer.score_to_probability(explainer.base_value))
