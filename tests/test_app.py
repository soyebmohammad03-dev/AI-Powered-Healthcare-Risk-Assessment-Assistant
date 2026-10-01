"""UI tests with Streamlit's AppTest: content and flow, not visual formatting."""
import importlib
import re

import pytest
from streamlit.testing.v1 import AppTest

from src import prediction
from src.data_loader import ROOT
from src.explainability import explain
from src.prediction import DEMO_INPUTS, FINAL_MODEL_PATH, predict
from src.recommendations import DISCLAIMER, Category, generate

APP = str(ROOT / "app.py")


@pytest.fixture(scope="module", autouse=True)
def artifact():
    if not FINAL_MODEL_PATH.exists():
        from src.train_models import main
        main()


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def submit(at: AppTest) -> AppTest:
    next(b for b in at.button if b.label == "Assess Risk").click().run()
    assert not at.exception
    return at


def page_text(at: AppTest) -> str:
    parts = [e.value for kind in ("markdown", "caption", "error", "warning", "subheader", "title")
             for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts) + "\n".join(f"{m.label} {m.value}" for m in at.metric)


def test_app_module_imports():
    importlib.import_module("app")  # UI only runs under `streamlit run` / AppTest


def test_app_starts_with_form_and_disclaimer():
    at = start()
    assert {n.key for n in at.number_input} == {"age", "height", "weight", "ap_hi", "ap_lo"}
    assert {r.key for r in at.radio} == {"gender", "cholesterol", "gluc", "smoke", "alco", "active"}
    assert any(b.label == "Assess Risk" for b in at.button)
    assert DISCLAIMER in page_text(at)
    assert "Assessment result" not in page_text(at)  # no result before submitting


def test_categorical_options_are_readable():
    at = start()
    assert at.radio(key="cholesterol").options == ["Normal", "Above normal", "Well above normal"]
    assert at.radio(key="gender").options == ["Female", "Male"]


@pytest.mark.parametrize("label", list(DEMO_INPUTS))
def test_demo_assessment_end_to_end(label):
    at = start()
    at.button(key=f"demo_{label}").click().run()
    submit(at)
    patient = DEMO_INPUTS[label]
    result = predict(patient)
    guidance = generate(patient, result, explain(patient))
    text = page_text(at)

    assert f"{result.probability_positive:.1%}" in text                                  # probability
    assert f"Prototype estimate band: {guidance.risk_category.name.capitalize()}" in text  # band
    assert "not clinically validated" in text
    assert "Why did the model estimate this probability?" in text                         # SHAP section
    assert len(at.get("plotly_chart")) == 1
    assert "model behaviour, not medical causation" in text
    for rec in guidance.recommendations:                                                  # guidance
        if rec.category is not Category.MODEL_CONTEXT:
            assert rec.title in text
    assert "Assessment summary" in text and f"{result.bmi:.1f}" in text                   # summary
    assert DISCLAIMER in text
    for internal in ("cat__", "num__", "ap_hi", "ap_lo", "gluc", "alco"):               # no internal names shown
        assert not re.search(rf"\b{internal}\b", text), internal


def test_result_survives_unrelated_rerun():
    at = start()
    at.button(key="demo_Example Patient A").click().run()
    submit(at)
    at.run()  # any widget interaction triggers a plain rerun
    assert "Assessment result" in page_text(at)


def test_missing_values_are_reported_without_result():
    at = submit(start())
    errors = " ".join(e.value for e in at.error)
    assert "Age: value is required" in errors and "Glucose: value is required" in errors
    assert "Assessment result" not in page_text(at)


def test_invalid_input_clears_previous_result():
    at = start()
    at.button(key="demo_Example Patient B").click().run()
    submit(at)
    assert "Assessment result" in page_text(at)
    at.number_input(key="ap_hi").set_value(90)
    at.number_input(key="ap_lo").set_value(95)
    submit(at)
    assert "Systolic blood pressure must be higher" in " ".join(e.value for e in at.error)
    assert "Assessment result" not in page_text(at)  # no stale result for different inputs


def test_missing_model_shows_setup_message(monkeypatch, tmp_path):
    monkeypatch.setattr(prediction, "FINAL_MODEL_PATH", tmp_path / "missing.joblib")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "has not been generated" in " ".join(e.value for e in at.error)
    assert any("python -m src.train_models" in c.value for c in at.code)
    assert not at.button or not any(b.label == "Assess Risk" for b in at.button)


def _about_script():
    import app
    app.about_page()


def test_about_page_renders():
    # AppTest.switch_page only resolves file-based pages; render the callable page directly instead.
    at = AppTest.from_function(_about_script, default_timeout=120).run()
    assert not at.exception
    text = page_text(at)
    assert "Model comparison" in text and "Selected model: Logistic Regression" in text
    assert DISCLAIMER in text and "Limitations" in text
    assert len(at.get("plotly_chart")) == 1  # global SHAP importance

