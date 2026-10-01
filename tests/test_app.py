"""UI tests with Streamlit's AppTest: content and flow across all five pages, not visual formatting."""
import importlib
import re

import pytest
from streamlit.testing.v1 import AppTest

from src import prediction
from src.data_loader import ROOT
from src.explainability import explain
from src.prediction import DEMO_INPUTS, FINAL_MODEL_PATH, load_model, predict
from src.recommendations import DISCLAIMER, Category, generate, nearby_cutoffs

APP = str(ROOT / "app.py")


@pytest.fixture(scope="module", autouse=True)
def artifact():
    if not FINAL_MODEL_PATH.exists():
        from src.train_models import main
        main()


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    return at


def submit(at: AppTest) -> AppTest:
    next(b for b in at.button if b.label == "Run Assessment").click().run()
    assert not at.exception
    return at


def assessed(label: str = "Example Patient B") -> AppTest:
    at = start()
    at.button(key=f"demo_{label}").click().run()
    return submit(at)


def page_text(at: AppTest) -> str:
    """All visible text, including st.html blocks."""
    parts = [e.value for kind in ("markdown", "caption", "error", "warning", "info", "title") for e in getattr(at, kind)]
    parts += [h.proto.body for h in at.get("html")]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(str(p) for p in parts)


def visit(at: AppTest, page: str) -> AppTest:
    at.switch_page(f"ui/{page}.py").run()
    assert not at.exception, at.exception
    return at


def test_app_module_imports():
    importlib.import_module("app")  # UI only runs under `streamlit run` / AppTest


def test_assess_page_starts_with_grouped_form_and_disclaimer():
    at = start()
    text = page_text(at)
    assert "AI-Powered Healthcare Risk Assessment Assistant" in text
    assert "An Explainable AI-Based Human-Centered Healthcare Decision Support System" in text
    assert "AI that predicts — and explains why." in text
    assert "Not a medical device." in text  # disclaimer above the form, not only in the footer
    for group in ("Demographics", "Body measurements", "Blood pressure", "Laboratory indicators", "Lifestyle"):
        assert group in text
    assert {n.key for n in at.number_input} == {"age", "height", "weight", "ap_hi", "ap_lo"}
    assert {g.key for g in at.get("button_group")} == {"gender", "smoke", "alco", "active"}
    assert {s.key: s.options for s in at.selectbox} == {k: ["Normal", "Above normal", "Well above normal"]
                                                        for k in ("cholesterol", "gluc")}
    assert any(b.label == "Run Assessment" for b in at.button)
    assert DISCLAIMER in text
    assert "Prototype estimate band" not in text  # no result before submitting


@pytest.mark.parametrize("label", list(DEMO_INPUTS))
def test_demo_assessment_end_to_end(label):
    at = assessed(label)
    patient = DEMO_INPUTS[label]
    result = predict(patient)
    guidance = generate(patient, result, explain(patient))
    text = page_text(at)
    assert f"{result.probability_positive:.1%}" in text                        # headline probability
    assert guidance.risk_category.name.upper() in text and "Prototype estimate band" in text
    assert "not a clinically validated" in text
    assert "Why the model estimated this" in text and "Model contribution ≠ medical causation" in text
    assert load_model()["calibration"].capitalize() in text and "Reference baseline" in text  # secondary info
    assert "Input conformity" in text and "Within training distribution" in text
    for rec in guidance.recommendations:                                        # guidance
        if rec.category is not Category.MODEL_CONTEXT:
            assert rec.title in text
    assert "Assessment summary" in text and f"{result.bmi:.1f}" in text
    assert "Model-estimated probability" in text and "What the model does not know" in [e.label for e in at.expander]
    assert ("Close to a cut-off" in text) == bool(nearby_cutoffs(result.probability_positive))
    for internal in ("cat__", "num__", "ap_hi", "ap_lo", "gluc", "alco"):       # no internal names shown
        assert not re.search(rf"\b{internal}\b", text), internal


def test_result_survives_rerun_and_page_switch():
    at = assessed("Example Patient A")
    expected = f"{predict(DEMO_INPUTS['Example Patient A']).probability_positive:.1%}"
    at.run()
    visit(at, "model")
    visit(at, "assess")
    assert expected in page_text(at)
    assert at.number_input(key="age").value == DEMO_INPUTS["Example Patient A"].age  # form restored


def test_missing_values_are_reported_without_result():
    at = submit(start())
    errors = " ".join(e.value for e in at.error)
    assert "Age: value is required" in errors and "Glucose: value is required" in errors
    assert "Prototype estimate band" not in page_text(at)


def test_invalid_input_clears_previous_result():
    at = assessed()
    at.number_input(key="ap_hi").set_value(90)
    at.number_input(key="ap_lo").set_value(95)
    submit(at)
    assert "Systolic blood pressure must be higher" in " ".join(e.value for e in at.error)
    assert "Prototype estimate band" not in page_text(at)


def test_missing_model_shows_setup_message(monkeypatch, tmp_path):
    monkeypatch.setattr(prediction, "FINAL_MODEL_PATH", tmp_path / "missing.joblib")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "has not been generated" in " ".join(e.value for e in at.error)
    assert any("python -m src.train_models" in c.value for c in at.code)
    assert not any(b.label == "Run Assessment" for b in at.button)


@pytest.mark.parametrize("page", ["explain", "explore"])
def test_assessment_pages_show_empty_state_without_assessment(page):
    at = visit(start(), page)
    assert "No assessment yet" in page_text(at)


def test_explain_page():
    at = visit(assessed(), "explain")
    e = explain(DEMO_INPUTS["Example Patient B"])
    text = page_text(at)
    assert "Why did the model estimate this probability?" in text
    assert f"{e.base_probability:.1%}" in text and f"{e.probability_positive:.1%}" in text
    assert "Model contribution ≠ medical causation" in text
    assert len(at.get("plotly_chart")) == 1
    assert e.contributions[0].label in text
    assert "In plain language" in text and f"reference baseline of {e.base_probability:.1%}" in text


def test_explore_page_what_if_flow():
    at = visit(assessed(), "explore")
    base = predict(DEMO_INPUTS["Example Patient B"]).probability_positive
    assert "model sensitivity analysis, not a predicted medical outcome" in page_text(at)
    assert f"{base:.1%}" in page_text(at) and "+0.0 pp" in page_text(at)  # no change -> baseline reproduced
    at.slider(key="wi_ap_hi").set_value(150.0).run()
    text = page_text(at)
    assert "Under the model, changing" in text and "percentage points" in text
    assert "higher than baseline" in text or "lower than baseline" in text
    assert "non-monotone in systolic blood pressure" in text  # robustness finding surfaced, read from the artifact
    at.slider(key="wi_ap_lo").set_value(160.0).run()                    # impossible: diastolic > systolic
    assert "outside the validated input domain" in " ".join(e.value for e in at.error)
    next(b for b in at.button if b.label == "Reset to baseline").click().run()
    assert at.slider(key="wi_ap_hi").value == 130 and not at.error
    visit(at, "assess")
    assert f"{base:.1%}" in page_text(at)  # the stored assessment was not modified


def test_model_page_sections():
    at = visit(start(), "model")
    text = page_text(at)
    for heading in ("Selection protocol", "Methodological correction", "Model probability ≠ display band ≠ classification threshold",
                    "Subgroup Performance Analysis", "not clinical uncertainty", "Two different questions",
                    "Controlled distribution-shift experiment", "Model disagreement", "Explanation stability",
                    "Input conformity"):
        assert heading in text, heading
    assert len(at.get("plotly_chart")) >= 15
    clean = load_json_artifact("metrics.json")["dataset"]["clean_records"]
    assert f"{clean:,} cleaned records" in text and "not clinical performance" in text
    at.slider(key="threshold").set_value(0.3).run()
    assert not at.exception
    specs = " ".join(c.proto.spec for c in at.get("plotly_chart"))
    assert "At threshold 0.30" in specs and "Net benefit (exploratory" in specs


def test_methodology_page_with_model_card():
    at = visit(start(), "methodology")
    text = page_text(at)
    for stage in ("Dataset", "Cleaning", "Calibration", "Explainability", "Human-centred interface"):
        assert stage in text
    assert "Model card" in text and "Not intended for" in text and "Non-clinical status" in text
    assert "68,573" in text
    assert "Non-clinical scope" in text and "What does it not know?" in " ".join(
        str(d.value) for d in at.dataframe)


def load_json_artifact(name: str) -> dict:
    import json
    return json.loads((ROOT / "artifacts" / name).read_text())


def near_cutoff_patient():
    """Example Patient B with systolic BP varied until the estimate sits near a cut-off."""
    from dataclasses import replace
    for ap_hi in range(100, 200):
        p = replace(DEMO_INPUTS["Example Patient B"], ap_hi=ap_hi)
        if nearby_cutoffs(predict(p).probability_positive):
            return p
    pytest.skip("no near-cut-off profile found")


def test_near_cutoff_note_is_shown():
    p = near_cutoff_patient()
    at = start()
    at.button(key="demo_Example Patient B").click().run()
    at.number_input(key="ap_hi").set_value(p.ap_hi)
    text = page_text(submit(at))
    assert f"{predict(p).probability_positive:.1%}" in text and "Close to a cut-off" in text


@pytest.mark.parametrize("page", ["assess", "explain", "explore", "model", "methodology"])
def test_disclaimer_on_every_page(page):
    at = assessed()
    if page != "assess":
        visit(at, page)
    assert DISCLAIMER in page_text(at)
