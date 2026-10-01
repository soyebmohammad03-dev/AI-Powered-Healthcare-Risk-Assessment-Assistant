"""Rule-based informational guidance. No ML, no LLM: every rule is an explicit `if` below.

Run `python -m src.recommendations` to generate guidance for the demo inputs.

ML model = prediction | SHAP = explanation | this module = general informational guidance.
Rules read the user's own inputs (and the model's probability band); they never use SHAP to decide
what to recommend. All thresholds are PROTOTYPE prompts, not clinical diagnostic thresholds.
"""
from dataclasses import asdict, dataclass, field
from enum import Enum

from src.explainability import POSITIVE, LocalExplanation
from src.prediction import PatientInput, PredictionResult
from src.preprocessing import CATEGORY_LABELS

DISCLAIMER = (
    "Educational AI prototype. Results are model-based estimates for information only. "
    "This is not a medical diagnosis and does not replace professional medical advice. "
    "If you have health concerns, please consult a qualified healthcare professional."
)

# Prototype risk bands on the model's probability. Presentation only: NOT clinically validated,
# not medical standards, and they do not change the model (predicted_class still uses 0.5).
LOWER_BELOW = 0.30
HIGHER_FROM = 0.60

# Prototype rule triggers aligned with commonly cited reference points; not diagnostic cut-offs.
SYSTOLIC_TRIGGER = 130   # mm Hg } ACC/AHA 2017: the elevated/stage-1 range begins at 130 systolic
DIASTOLIC_TRIGGER = 80   # mm Hg }               or 80 diastolic
BMI_LOW, BMI_HIGH = 18.5, 25.0  # WHO adult BMI categories: below 18.5 / 25 and above

MAX_INPUT_ITEMS = 3      # shown after the follow-up item; the rest go to Guidance.additional


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
    GLUCOSE = "Glucose"
    SMOKING = "Smoking"
    WEIGHT = "Body weight (BMI)"
    ACTIVITY = "Physical activity"
    ALCOHOL = "Alcohol"
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
    recommendations: list[Recommendation]  # follow-up, top input items by priority, model context last
    additional: list[Recommendation] = field(default_factory=list)  # further triggered items, not dropped
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
            "The model estimates a higher probability for these inputs, based on patterns in a research "
            "dataset. This is an estimate, not a medical finding. Consider discussing these results "
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
    """One `if` per rule, in tie-break order. Uses only fields of the current schema."""
    recs = []
    if patient.ap_hi >= SYSTOLIC_TRIGGER or patient.ap_lo >= DIASTOLIC_TRIGGER:
        recs.append(Recommendation(
            Category.BLOOD_PRESSURE, "Review your blood pressure readings",
            f"The reported blood pressure ({patient.ap_hi:g}/{patient.ap_lo:g} mm Hg) is at or above the "
            f"{SYSTOLIC_TRIGGER}/{DIASTOLIC_TRIGGER} mm Hg level this prototype uses as a prompt. A single "
            "reading says little on its own; consider discussing your blood pressure with a qualified "
            "healthcare professional.",
            Priority.MODERATE, f"ap_hi = {patient.ap_hi:g}, ap_lo = {patient.ap_lo:g}"))
    if patient.cholesterol >= 2:
        level = CATEGORY_LABELS["cholesterol"][patient.cholesterol].lower()
        recs.append(Recommendation(
            Category.CHOLESTEROL, "Discuss your cholesterol level",
            f"Cholesterol was reported as {level}. Consider discussing cholesterol and other "
            "cardiovascular risk factors with a qualified healthcare professional.",
            Priority.MODERATE, f"cholesterol = {patient.cholesterol} ({level})"))
    if patient.gluc >= 2:
        level = CATEGORY_LABELS["gluc"][patient.gluc].lower()
        recs.append(Recommendation(
            Category.GLUCOSE, "Discuss your glucose level",
            f"Glucose was reported as {level}. Consider discussing glucose results with a qualified "
            "healthcare professional.", Priority.MODERATE, f"gluc = {patient.gluc} ({level})"))
    if patient.smoke == 1:
        recs.append(Recommendation(
            Category.SMOKING, "Consider support to stop smoking",
            "Smoking was reported. Not smoking is widely recommended for heart and general health, and a "
            "healthcare professional can discuss support options.", Priority.MODERATE, "smoke = 1"))
    if not BMI_LOW <= patient.bmi < BMI_HIGH:
        recs.append(Recommendation(
            Category.WEIGHT, "Discuss your weight",
            f"Your height and weight give a BMI of {patient.bmi:.1f}, outside the {BMI_LOW}-{BMI_HIGH} range "
            "this prototype uses as a prompt. BMI is a rough measure; a healthcare professional can advise "
            "what is appropriate for you.", Priority.LOW, f"BMI = {patient.bmi:.1f}"))
    if patient.active == 0:
        recs.append(Recommendation(
            Category.ACTIVITY, "Consider regular physical activity",
            "Regular physical activity is generally beneficial for heart health. A healthcare "
            "professional can advise what type and amount of activity suits you.",
            Priority.LOW, "active = 0"))
    else:
        recs.append(Recommendation(
            Category.ACTIVITY, "Keep up your physical activity",
            "Regular physical activity was reported, which is generally beneficial for heart health.",
            Priority.LOW, "active = 1"))
    if patient.alco == 1:
        recs.append(Recommendation(
            Category.ALCOHOL, "Consider your alcohol intake",
            "Alcohol intake was reported. Consider discussing your alcohol intake with a qualified "
            "healthcare professional.", Priority.LOW, "alco = 1"))
    return recs


def _model_context(explanation: LocalExplanation) -> Recommendation:
    top = explanation.contributions[0]
    toward = "higher" if top.direction == POSITIVE else "lower"
    return Recommendation(
        Category.MODEL_CONTEXT, "What the model weighed most",
        f"The model placed the most weight on {top.label} ({top.display_value}), which moved its estimate "
        f"{toward}. This describes how the model weighted the inputs, not a medical cause.",
        Priority.INFO, f"Largest SHAP contribution: {top.feature} ({top.shap_value:+.3f} log-odds)")


def generate(patient: PatientInput, prediction: PredictionResult,
             explanation: LocalExplanation | None = None) -> Guidance:
    if not isinstance(patient, PatientInput):
        raise TypeError("generate() expects a validated PatientInput.")
    if prediction.inputs != asdict(patient):
        raise ValueError("prediction does not belong to this patient input.")
    category = risk_category(prediction.probability_positive)
    order = list(Priority)
    items = sorted(_input_rules(patient), key=lambda r: order.index(r.priority))  # stable within a priority
    shown = [_follow_up(category, prediction.probability_positive), *items[:MAX_INPUT_ITEMS]]
    if explanation is not None:
        shown.append(_model_context(explanation))
    return Guidance(category, prediction.probability_positive, shown, items[MAX_INPUT_ITEMS:])


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
        for r in g.additional:
            print(f"  (more) [{r.priority.value}] {r.category.value}: {r.title} - reason: {r.reason}")
