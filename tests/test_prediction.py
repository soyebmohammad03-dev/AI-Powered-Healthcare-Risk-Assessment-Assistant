import json
import subprocess
import sys
from dataclasses import asdict, replace

import joblib
import numpy as np
import pytest

from src.data_loader import ROOT, load_dataset, split
from src.evaluate_models import evaluate
from src.prediction import (DEMO_INPUTS, FINAL_MODEL_PATH, InvalidInputError, ModelArtifactError,
                            PatientInput, load_model, predict)
from src.preprocessing import FEATURES, build_preprocessor

VALID = asdict(DEMO_INPUTS["Example Patient B"])


@pytest.fixture(scope="module")
def bundle():
    if not FINAL_MODEL_PATH.exists():  # tests use the real persisted artifact; create it once if absent
        from src.train_models import main
        main()
    return load_model()


def test_artifact_contains_full_pipeline(bundle):
    assert bundle["model_name"] == "logistic_regression"
    assert list(bundle["pipeline"].named_steps) == ["pre", "clf"]
    assert list(bundle["pipeline"].feature_names_in_) == FEATURES


@pytest.mark.parametrize("label", list(DEMO_INPUTS))
def test_valid_prediction(bundle, label):
    r = predict(DEMO_INPUTS[label], bundle)
    assert 0.0 <= r.probability_positive <= 1.0
    assert r.probability_positive + r.probability_negative == pytest.approx(1.0)
    assert r.predicted_class == int(r.probability_positive >= 0.5)
    assert r.inputs == asdict(DEMO_INPUTS[label])


def test_prediction_is_deterministic(bundle):
    p = DEMO_INPUTS["Example Patient B"]
    assert predict(p, bundle) == predict(p, bundle)


def test_matches_pipeline_and_ignores_key_order(bundle):
    shuffled = dict(reversed(list(VALID.items())))
    r = predict(PatientInput.from_dict(shuffled), bundle)
    expected = bundle["pipeline"].predict_proba(PatientInput(**VALID).to_frame())[0, 1]
    assert r.probability_positive == expected


def test_persisted_model_is_the_evaluated_model(bundle):
    """Same split + same preprocessing as training reproduces the stored test metrics exactly."""
    df = load_dataset()
    _, X_test, _, y_test = split(df)
    got = evaluate(bundle["pipeline"], X_test, y_test)
    assert got == bundle["test_metrics"]
    saved = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    assert got == pytest.approx(saved["models"]["logistic_regression"]["test"])


def test_inference_preprocessing_equals_training_preprocessing(bundle):
    df = load_dataset()
    X_train, X_test, _, _ = split(df)
    fresh = build_preprocessor().fit(X_train)
    np.testing.assert_array_equal(fresh.transform(X_test), bundle["pipeline"].named_steps["pre"].transform(X_test))


def test_saved_model_works_in_fresh_process(bundle):
    out = subprocess.run([sys.executable, "-m", "src.prediction"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert "Artifact OK" in out and out.count("P(pos)=") == len(DEMO_INPUTS)


@pytest.mark.parametrize("field, value, message", [
    ("cp", 9, "not a valid code"),
    ("thal", 4, "not a valid code"),
    ("sex", 0.5, "whole number"),
    ("chol", 5000, "outside the accepted range"),
    ("age", -3, "outside the accepted range"),
    ("trestbps", "120", "expected a number"),
    ("oldpeak", float("nan"), "expected a number"),
    ("exang", True, "expected a number"),
    ("ca", None, "required"),
])
def test_invalid_values_rejected(field, value, message):
    with pytest.raises(InvalidInputError, match=message):
        PatientInput(**{**VALID, field: value})


def test_missing_and_unknown_fields_rejected():
    data = {k: v for k, v in VALID.items() if k != "chol"}
    with pytest.raises(InvalidInputError, match="chol: value is required"):
        PatientInput.from_dict(data)
    with pytest.raises(InvalidInputError, match="bmi: unknown field"):
        PatientInput.from_dict({**VALID, "bmi": 25})


def test_all_errors_reported_together():
    with pytest.raises(InvalidInputError) as exc:
        PatientInput(**{**VALID, "cp": 9, "chol": 5000})
    assert len(exc.value.errors) == 2


def test_artifact_failures_are_explicit(bundle, tmp_path):
    with pytest.raises(ModelArtifactError, match="not found"):
        load_model(tmp_path / "missing.joblib")

    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a model")
    with pytest.raises(ModelArtifactError, match="could not be loaded"):
        load_model(corrupt)

    no_pre = tmp_path / "no_pre.joblib"
    joblib.dump({**bundle, "pipeline": bundle["pipeline"].named_steps["clf"]}, no_pre)
    with pytest.raises(ModelArtifactError, match="preprocessing"):
        load_model(no_pre)

    wrong_contract = tmp_path / "wrong.joblib"
    joblib.dump({**bundle, "features": FEATURES[:-1]}, wrong_contract)
    with pytest.raises(ModelArtifactError, match="feature contract"):
        load_model(wrong_contract)


def test_demo_inputs_are_valid_and_distinct():
    assert len({replace(p) for p in DEMO_INPUTS.values()}) == len(DEMO_INPUTS)
