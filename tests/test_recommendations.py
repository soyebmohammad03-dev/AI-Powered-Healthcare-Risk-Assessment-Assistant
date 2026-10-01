from dataclasses import asdict, replace

import pytest

from src.explainability import explain
from src.prediction import (DEMO_INPUTS, FINAL_MODEL_PATH, InvalidInputError, PatientInput,
                            PredictionResult, predict)
from src.recommendations import (BP_TRIGGER, CHOL_TRIGGER, DISCLAIMER, HIGHER_FROM, LOWER_BELOW, Category,
                                 Guidance, Priority, RiskCategory, generate, risk_category)

# Unremarkable values against every rule trigger (synthetic, not a real patient).
BASE = dict(age=45, sex=0, cp=3, trestbps=120, chol=180, fbs=0, restecg=0,
            thalach=165, exang=0, oldpeak=0.0, slope=1, ca=0, thal=3)
PROHIBITED = ["diagnos", "you have heart disease", "you are healthy", "you are safe", "healthy",
              "medication", "medicine", "drug", "prescri", "dose", "dosage", "treatment", "cure",
              "caused", "causes", "proves", "guarantee", "definitely"]


@pytest.fixture(scope="module", autouse=True)
def artifact():
    if not FINAL_MODEL_PATH.exists():
        from src.train_models import main
        main()


def run(**overrides) -> Guidance:
    patient = PatientInput(**{**BASE, **overrides})
    return generate(patient, predict(patient), explain(patient))


def fake_prediction(patient: PatientInput, p: float) -> PredictionResult:
    """Pins the probability to test band behaviour independently of what the model outputs."""
    return PredictionResult(int(p >= 0.5), p, 1 - p, "logistic_regression", asdict(patient))


def categories(g: Guidance) -> list[Category]:
    return [r.category for r in g.recommendations]


def test_normal_input():
    g = run()
    assert Category.BLOOD_PRESSURE not in categories(g) and Category.CHOLESTEROL not in categories(g)
    assert Category.ACTIVITY in categories(g)
    assert 3 <= len(g.recommendations) <= 6
    assert g.disclaimer == DISCLAIMER


def test_blood_pressure_rule_threshold():
    assert Category.BLOOD_PRESSURE not in categories(run(trestbps=BP_TRIGGER - 1))
    rec = next(r for r in run(trestbps=BP_TRIGGER).recommendations if r.category is Category.BLOOD_PRESSURE)
    assert "trestbps" in rec.reason and "healthcare professional" in rec.message


def test_cholesterol_rule_threshold():
    assert Category.CHOLESTEROL not in categories(run(chol=CHOL_TRIGGER - 1))
    assert Category.CHOLESTEROL in categories(run(chol=CHOL_TRIGGER))


def test_blood_sugar_rule():
    assert Category.BLOOD_SUGAR in categories(run(fbs=1))
    assert Category.BLOOD_SUGAR not in categories(run(fbs=0))


def test_exercise_indicator_replaces_activity_advice():
    g = run(exang=1)
    rec = next(r for r in g.recommendations if r.category is Category.EXERCISE)
    assert rec.priority is Priority.HIGH
    assert Category.ACTIVITY not in categories(g)  # no "be active" advice when exertional symptoms reported


def test_risk_bands():
    assert risk_category(0.0) is RiskCategory.LOWER
    assert risk_category(LOWER_BELOW - 1e-9) is RiskCategory.LOWER
    assert risk_category(LOWER_BELOW) is RiskCategory.MODERATE
    assert risk_category(HIGHER_FROM) is RiskCategory.HIGHER
    assert risk_category(1.0) is RiskCategory.HIGHER
    with pytest.raises(ValueError):
        risk_category(1.5)


def test_higher_probability_suggests_professional_evaluation():
    patient = PatientInput(**BASE)
    g = generate(patient, fake_prediction(patient, 0.9))
    assert g.risk_category is RiskCategory.HIGHER
    first = g.recommendations[0]
    assert first.category is Category.FOLLOW_UP and first.priority is Priority.HIGH
    assert "healthcare professional" in first.message and "not a medical finding" in first.message


def test_lower_probability_does_not_claim_health():
    patient = PatientInput(**BASE)
    g = generate(patient, fake_prediction(patient, 0.05))
    assert g.risk_category is RiskCategory.LOWER
    general = next(r for r in g.recommendations if r.category is Category.GENERAL)
    assert "does not rule out" in general.message


@pytest.mark.parametrize("label", list(DEMO_INPUTS))
def test_demo_guidance_is_generated_and_safe(label):
    patient = DEMO_INPUTS[label]
    p = predict(patient)
    g = generate(patient, p, explain(patient))
    assert g.risk_category is risk_category(p.probability_positive)
    assert g.recommendations[-1].category is Category.MODEL_CONTEXT
    assert all(isinstance(r.priority, Priority) and isinstance(r.category, Category) for r in g.recommendations)
    order = [list(Priority).index(r.priority) for r in g.recommendations]
    assert order == sorted(order)
    for r in g.recommendations:
        text = f"{r.title} {r.message}".lower()
        assert not [w for w in PROHIBITED if w in text], (r.title, text)


def test_model_context_uses_top_shap_feature():
    patient = DEMO_INPUTS["Example Patient C"]
    e = explain(patient)
    ctx = generate(patient, predict(patient), e).recommendations[-1]
    assert e.contributions[0].label in ctx.message and "not a medical cause" in ctx.message


def test_deterministic():
    patient = DEMO_INPUTS["Example Patient B"]
    a = generate(patient, predict(patient), explain(patient))
    b = generate(patient, predict(patient), explain(patient))
    assert a == b


def test_invalid_input_handling():
    with pytest.raises(InvalidInputError):
        run(chol=None)
    patient = DEMO_INPUTS["Example Patient A"]
    with pytest.raises(TypeError):
        generate(asdict(patient), predict(patient))
    with pytest.raises(ValueError, match="does not belong"):
        generate(replace(patient, chol=300), predict(patient))


@pytest.mark.parametrize("p", [0.05, 0.45, 0.9])
@pytest.mark.parametrize("triggers", [{}, dict(trestbps=160, chol=300, fbs=1, exang=1)])
def test_every_message_is_free_of_prohibited_language(p, triggers):
    patient = PatientInput(**{**BASE, **triggers})
    g = generate(patient, fake_prediction(patient, p), explain(patient))
    for r in g.recommendations:
        text = f"{r.title} {r.message}".lower()
        assert not [w for w in PROHIBITED if w in text], (r.title, text)
