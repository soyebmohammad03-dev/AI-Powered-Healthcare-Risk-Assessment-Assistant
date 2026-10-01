import re
from dataclasses import asdict, replace

import pytest

from src.data_loader import ROOT
from src.explainability import explain
from src.prediction import (DEMO_INPUTS, FINAL_MODEL_PATH, InvalidInputError, PatientInput,
                            PredictionResult, predict)
from src.preprocessing import FEATURES
from src.recommendations import (BMI_HIGH, BMI_LOW, DIASTOLIC_TRIGGER, DISCLAIMER, HIGHER_FROM, LOWER_BELOW,
                                 MAX_INPUT_ITEMS, SYSTOLIC_TRIGGER, Category, Guidance, Priority,
                                 RiskCategory, generate, risk_category)

# Below every rule trigger (synthetic, not a real patient). BMI = 60 / 1.65^2 = 22.0.
BASE = dict(age=40, gender=1, height=165, weight=60, ap_hi=115, ap_lo=75,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1)
ALL_TRIGGERS = dict(ap_hi=160, ap_lo=100, cholesterol=3, gluc=3, smoke=1, alco=1, active=0, weight=95)
PROHIBITED = ["diagnos", "you have", "you are healthy", "you are safe", "healthy", "free of",
              "medication", "medicine", "drug", "prescri", "dose", "dosage", "treatment", "cure",
              "nicotine", "caused", "causes", "proves", "guarantee", "definitely", "obese"]


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
    return PredictionResult(int(p >= 0.5), p, 1 - p, "logistic_regression", asdict(patient), patient.bmi)


def everything(g: Guidance) -> list:
    return g.recommendations + g.additional


def categories(g: Guidance) -> list[Category]:
    return [r.category for r in everything(g)]


def test_normal_input():
    g = run()
    assert categories(g) == [Category.GENERAL, Category.ACTIVITY, Category.MODEL_CONTEXT]
    assert g.disclaimer == DISCLAIMER and g.additional == []


@pytest.mark.parametrize("overrides, triggered", [
    (dict(ap_hi=SYSTOLIC_TRIGGER - 1, ap_lo=DIASTOLIC_TRIGGER - 1), False),
    (dict(ap_hi=SYSTOLIC_TRIGGER), True),
    (dict(ap_lo=DIASTOLIC_TRIGGER), True),
])
def test_blood_pressure_rule(overrides, triggered):
    assert (Category.BLOOD_PRESSURE in categories(run(**overrides))) is triggered


@pytest.mark.parametrize("field, category", [("cholesterol", Category.CHOLESTEROL), ("gluc", Category.GLUCOSE)])
def test_level_rules(field, category):
    assert category not in categories(run(**{field: 1}))
    for level, word in [(2, "above normal"), (3, "well above normal")]:
        rec = next(r for r in everything(run(**{field: level})) if r.category is category)
        assert f"reported as {word}." in rec.message


def test_bmi_rule():
    assert Category.WEIGHT not in categories(run())                          # BMI 22.0
    # height 200 cm -> BMI = weight / 4 exactly, so the boundaries are tested without float round-off
    assert Category.WEIGHT not in categories(run(height=200, weight=BMI_LOW * 4))      # 18.5
    assert Category.WEIGHT not in categories(run(height=200, weight=BMI_HIGH * 4 - 0.4))  # 24.9
    assert Category.WEIGHT in categories(run(height=200, weight=BMI_HIGH * 4))           # 25.0
    assert Category.WEIGHT in categories(run(height=200, weight=BMI_LOW * 4 - 0.4))      # 18.4


def test_lifestyle_rules():
    assert Category.SMOKING in categories(run(smoke=1))
    assert Category.ALCOHOL in categories(run(alco=1))
    inactive = next(r for r in everything(run(active=0)) if r.category is Category.ACTIVITY)
    assert inactive.title == "Consider regular physical activity"


def test_item_cap_keeps_every_triggered_rule():
    g = run(**ALL_TRIGGERS)
    shown_inputs = [r for r in g.recommendations if r.category not in (Category.FOLLOW_UP, Category.MODEL_CONTEXT)]
    assert len(shown_inputs) == MAX_INPUT_ITEMS and len(g.recommendations) == MAX_INPUT_ITEMS + 2
    assert len(shown_inputs) + len(g.additional) == 7  # BP, cholesterol, glucose, smoking, BMI, activity, alcohol
    order = [list(Priority).index(r.priority) for r in shown_inputs + g.additional]
    assert order == sorted(order)


def test_rules_only_reference_current_schema():
    for overrides in ({}, ALL_TRIGGERS):
        for r in everything(run(**overrides)):
            if r.category is Category.MODEL_CONTEXT or r.reason.startswith("Model probability"):
                continue
            fields = set(re.findall(r"(\w+) = ", r.reason))
            assert fields and fields <= set(FEATURES) | {"BMI"}, r.reason


def test_no_old_dataset_fields_in_source():
    old = ["trestbps", "thalach", "oldpeak", "restecg", "exang", "fbs", "thal", "Cleveland", "angina"]
    for path in (ROOT / "src").glob("*.py"):
        text = path.read_text()
        assert not [w for w in old if re.search(rf"\b{w}\b", text)], path.name


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
    first = g.recommendations[0]
    assert g.risk_category is RiskCategory.HIGHER
    assert first.category is Category.FOLLOW_UP and first.priority is Priority.HIGH
    assert "healthcare professional" in first.message and "not a medical finding" in first.message


def test_lower_probability_does_not_claim_health():
    patient = PatientInput(**BASE)
    g = generate(patient, fake_prediction(patient, 0.05))
    assert g.risk_category is RiskCategory.LOWER
    assert "does not rule out" in g.recommendations[0].message


@pytest.mark.parametrize("p", [0.05, 0.45, 0.9])
@pytest.mark.parametrize("overrides", [{}, ALL_TRIGGERS, dict(weight=45, smoke=1)])
def test_every_message_is_free_of_prohibited_language(p, overrides):
    patient = PatientInput(**{**BASE, **overrides})
    g = generate(patient, fake_prediction(patient, p), explain(patient))
    for r in everything(g):
        text = f"{r.title} {r.message}".lower()
        assert not [w for w in PROHIBITED if w in text], (r.title, text)


@pytest.mark.parametrize("label", list(DEMO_INPUTS))
def test_demo_guidance(label):
    patient = DEMO_INPUTS[label]
    p = predict(patient)
    g = generate(patient, p, explain(patient))
    assert g.risk_category is risk_category(p.probability_positive)
    assert 3 <= len(g.recommendations) <= 5
    assert g.recommendations[-1].category is Category.MODEL_CONTEXT
    assert "not a medical cause" in g.recommendations[-1].message
    assert all(isinstance(r.priority, Priority) and isinstance(r.category, Category) for r in everything(g))


def test_deterministic():
    patient = DEMO_INPUTS["Example Patient B"]
    assert generate(patient, predict(patient), explain(patient)) == generate(patient, predict(patient), explain(patient))


def test_invalid_input_handling():
    with pytest.raises(InvalidInputError):
        run(gluc=None)
    patient = DEMO_INPUTS["Example Patient A"]
    with pytest.raises(TypeError):
        generate(asdict(patient), predict(patient))
    with pytest.raises(ValueError, match="does not belong"):
        generate(replace(patient, ap_hi=150), predict(patient))


def test_nearby_cutoffs():
    from src.recommendations import nearby_cutoffs
    assert nearby_cutoffs(0.10) == [] and nearby_cutoffs(0.45) == []
    assert nearby_cutoffs(0.31) == [0.30] and nearby_cutoffs(0.49) == [0.5] and nearby_cutoffs(0.62) == [0.60]
