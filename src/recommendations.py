"""Rule-based informational guidance. No ML, no LLM: every rule is an explicit `if` below.

Run `python -m src.recommendations` to generate guidance for the demo inputs.

ML model = prediction | SHAP = explanation | this module = general informational guidance.
All thresholds here are PROTOTYPE presentation/trigger rules, not clinical diagnostic thresholds.
"""
from dataclasses import asdict, dataclass
from enum import Enum

from src.explainability import POSITIVE, LocalExplanation
from src.prediction import PatientInput, PredictionResult

DISCLAIMER = (
    "Educational AI prototype. Results are model-based estimates for information only. "
    "This is not a medical diagnosis and does not replace professional medical advice. "
    "If you have health concerns, please consult a qualified healthcare professional."
)

# Prototype risk bands on the model's probability. Presentation only: NOT clinically validated,
# not medical standards, and they do not change the model (predicted_class still uses 0.5).
LOWER_BELOW = 0.30
HIGHER_FROM = 0.60

# Prototype rule triggers, chosen to align with commonly cited reference points; not diagnostic cut-offs.
BP_TRIGGER = 130    # resting systolic mm Hg; ACC/AHA 2017 elevated/stage-1 systolic range starts at 130
CHOL_TRIGGER = 200  # total serum cholesterol mg/dl; NCEP ATP III "borderline high" starts at 200


class RiskCategory(Enum):
    LOWER = "Lower predicted risk"
    MODERATE = "Moderate predicted risk"
    HIGHER = "Higher predicted risk"


class Priority(Enum):  # declaration order = display order
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INFO = "info"


class Category(Enum):
    FOLLOW_UP = "Follow-up / professional consultation"
    GENERAL = "General health information"
    BLOOD_PRESSURE = "Blood pressure"
    CHOLESTEROL = "Cholesterol"
    BLOOD_SUGAR = "Blood sugar"
    EXERCISE = "Exercise-related indicators"
    ACTIVITY = "Physical activity"
    MODEL_CONTEXT = "Model explanation context"


@dataclass(frozen=True)
class Recommendation:
    category: Category
    title: str
    message: str
    priority: Priority
    reason: str  # which input / output triggered the rule


@dataclass(frozen=True)
class Guidance:
    risk_category: RiskCategory
    probability_positive: float
    recommendations: list[Recommendation]  # sorted by priority; model context last
    disclaimer: str = DISCLAIMER


def risk_category(probability: float) -> RiskCategory:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability!r}")
    if probability < LOWER_BELOW:
        return RiskCategory.LOWER
    return RiskCategory.MODERATE if probability < HIGHER_FROM else RiskCategory.HIGHER


def _follow_up(category: RiskCategory, p: float) -> Recommendation:
    reason = f"Model probability {p:.1%} -> {category.value} (prototype band)"
    if category is RiskCategory.HIGHER:
        return Recommendation(
            Category.FOLLOW_UP, "Consider a professional evaluation",
            "The model estimates a higher probability for these inputs, based on patterns in a historical "
            "research dataset. This is an estimate, not a medical finding. Consider discussing these results "
            "with a qualified healthcare professional.", Priority.HIGH, reason)
    if category is RiskCategory.MODERATE:
        return Recommendation(
            Category.FOLLOW_UP, "Consider discussing these results",
            "The model's estimate for these inputs falls in the middle band. Consider discussing these "
            "results and your health parameters with a qualified healthcare professional.",
            Priority.MODERATE, reason)
    return Recommendation(
        Category.GENERAL, "Keep up general preventive care",
        "The model's estimate is lower for these inputs. A lower estimate does not rule out any health "
        "condition. Regular check-ups with a healthcare professional are generally recommended, and you "
        "should seek advice if you notice symptoms.", Priority.LOW, reason)


def _input_rules(patient: PatientInput) -> list[Recommendation]:
    recs = []
    if patient.trestbps >= BP_TRIGGER:
        recs.append(Recommendation(
            Category.BLOOD_PRESSURE, "Review your blood pressure readings",
            f"The reported resting blood pressure ({patient.trestbps:g} mm Hg) is at or above the "
            f"{BP_TRIGGER} mm Hg level this prototype uses as a prompt. Consider discussing your blood "
            "pressure readings with a qualified healthcare professional.",
            Priority.MODERATE, f"trestbps = {patient.trestbps:g} >= {BP_TRIGGER}"))
    if patient.chol >= CHOL_TRIGGER:
        recs.append(Recommendation(
            Category.CHOLESTEROL, "Discuss your cholesterol level",
            f"The reported cholesterol ({patient.chol:g} mg/dl) is at or above the {CHOL_TRIGGER} mg/dl "
            "level this prototype uses as a prompt. Consider discussing cholesterol and other "
            "cardiovascular risk factors with a qualified healthcare professional.",
            Priority.MODERATE, f"chol = {patient.chol:g} >= {CHOL_TRIGGER}"))
    if patient.fbs == 1:
        recs.append(Recommendation(
            Category.BLOOD_SUGAR, "Discuss your blood sugar reading",
            "Fasting blood sugar above 120 mg/dl was reported. Consider discussing blood sugar results "
            "with a qualified healthcare professional.", Priority.MODERATE, "fbs = 1 (> 120 mg/dl)"))
    if patient.exang == 1:
        recs.append(Recommendation(
            Category.EXERCISE, "Discuss chest pain during exercise",
            "Exercise-induced angina was reported. Chest discomfort during exertion is worth discussing "
            "with a qualified healthcare professional, especially before starting or changing an "
            "exercise routine. Seek prompt medical help for severe or sudden chest pain.",
            Priority.HIGH, "exang = 1"))
    else:
        # Activity advice only when no exercise-related symptom is reported.
        recs.append(Recommendation(
            Category.ACTIVITY, "Stay physically active",
            "Regular physical activity is generally beneficial for heart health. A healthcare "
            "professional can advise what type and amount of activity suits you.",
            Priority.LOW, "exang = 0"))
    return recs


def _model_context(explanation: LocalExplanation) -> Recommendation:
    top = explanation.contributions[0]
    toward = "higher" if top.direction == POSITIVE else "lower"
    return Recommendation(
        Category.MODEL_CONTEXT, "What the model weighed most",
        f"The model placed the most weight on {top.label} ({top.display_value}), which moved its estimate "
        f"{toward}. This describes how the model works, not a medical cause.",
        Priority.INFO, f"Largest SHAP contribution: {top.feature} ({top.shap_value:+.3f} log-odds)")


def generate(patient: PatientInput, prediction: PredictionResult,
             explanation: LocalExplanation | None = None) -> Guidance:
    if not isinstance(patient, PatientInput):
        raise TypeError("generate() expects a validated PatientInput.")
    if prediction.inputs != asdict(patient):
        raise ValueError("prediction does not belong to this patient input.")
    category = risk_category(prediction.probability_positive)
    recs = [_follow_up(category, prediction.probability_positive), *_input_rules(patient)]
    order = list(Priority)
    recs.sort(key=lambda r: order.index(r.priority))  # stable: rule order kept within a priority
    if explanation is not None:
        recs.append(_model_context(explanation))
    return Guidance(category, prediction.probability_positive, recs)


if __name__ == "__main__":
    from src.explainability import explain
    from src.prediction import DEMO_INPUTS, predict

    print(DISCLAIMER)
    for name, patient in DEMO_INPUTS.items():
        p = predict(patient)
        g = generate(patient, p, explain(patient))
        print(f"\n{name}: P(pos)={p.probability_positive:.4f} -> {g.risk_category.value}")
        for r in g.recommendations:
            print(f"  [{r.priority.value:<8}] {r.category.value}: {r.title}\n             {r.message}\n"
                  f"             reason: {r.reason}")
