"""ASSESS: the patient-input workspace and the headline result."""
from dataclasses import asdict
from html import escape

import streamlit as st

from src.explainability import NEGATIVE, POSITIVE
from src.prediction import DEMO_INPUTS, InvalidInputError, PatientInput, predict
from src.preprocessing import CATEGORY_LABELS, FEATURE_LABELS, FEATURES, PLAUSIBLE
from src.recommendations import Category, Priority, generate
from ui.core import (MODEL_NAMES, PAGES, band_bar, band_color, band_name, conformity, engine, factor_rows, footer,
                     page_header, readable, section, tokens)

PRIORITY_COLORS = {Priority.HIGH: "red", Priority.MODERATE: "orange", Priority.LOW: "green", Priority.INFO: "gray"}


def set_field(name: str, value):
    st.session_state[name] = float(value) if name == "weight" and value is not None else value  # float widget


def fill_form(values: dict | None):
    for name in FEATURES:
        set_field(name, None if values is None else values[name])
    st.session_state.pop("assessment", None)


def input_form() -> bool:
    # Streamlit drops widget state when another page is shown; restore the assessed inputs so the form
    # and the displayed result always match.
    saved = asdict(st.session_state["assessment"][0]) if "assessment" in st.session_state else {}
    for name in FEATURES:
        if name not in st.session_state:
            set_field(name, saved.get(name))

    def number(name, label, **kw):
        low, high = PLAUSIBLE[name]
        if name == "weight":
            low, high = float(low), float(high)
        st.number_input(label, key=name, min_value=low, max_value=high, placeholder=f"{low:g}–{high:g}", **kw)

    def choice(name, help=None):
        if len(CATEGORY_LABELS[name]) > 2:  # three-level lab values: a select never clips "Well above normal"
            st.selectbox(FEATURE_LABELS[name], list(CATEGORY_LABELS[name]), key=name, help=help,
                         format_func=CATEGORY_LABELS[name].get, placeholder="Select level")
        else:
            st.segmented_control(FEATURE_LABELS[name], list(CATEGORY_LABELS[name]), key=name, help=help,
                                 format_func=CATEGORY_LABELS[name].get, width="stretch")

    with st.container(border=True):
        st.markdown("**Health inputs**  \n<span class='subtle'>All 11 fields are required. Or load a synthetic "
                    "example (not a real patient):</span>", unsafe_allow_html=True)
        cols = st.columns(len(DEMO_INPUTS) + 1, gap="small")
        for col, (label, patient) in zip(cols, DEMO_INPUTS.items()):
            col.button(label.replace("Example Patient", "Example"), key=f"demo_{label}", on_click=fill_form,
                       args=(asdict(patient),), width="stretch", icon=":material/person:",
                       help="Synthetic demonstration input, not a real patient.")
        cols[-1].button("Clear", key="clear", on_click=fill_form, args=(None,), width="stretch",
                        icon=":material/restart_alt:")

        with st.form("assessment_form", border=False):
            section("Person")
            a, b, c, d = st.columns([1, 1, 1, 1.25])
            with a:
                number("age", "Age (years)", step=1,
                       help="The training data covers ages 29–65, so only this range is accepted.")
            with b:
                number("height", "Height (cm)", step=1)
            with c:
                number("weight", "Weight (kg)", step=0.5, format="%.1f")
            with d:
                choice("gender")
            left, right = st.columns(2, gap="large")
            with left:
                section("Vitals")
                a, b = st.columns(2)
                with a:
                    number("ap_hi", "Systolic (mmHg)", step=1, help="The upper blood-pressure number.")
                with b:
                    number("ap_lo", "Diastolic (mmHg)", step=1, help="The lower blood-pressure number.")
            with right:
                section("Labs")
                choice("cholesterol", help="As reported by a test, relative to the normal range.")
                choice("gluc", help="Blood glucose as reported by a test, relative to the normal range.")
            section("Lifestyle")
            a, b, c = st.columns(3)
            with a:
                choice("smoke")
            with b:
                choice("alco")
            with c:
                choice("active", help="Whether you are regularly physically active.")
            return st.form_submit_button("Run Assessment", type="primary", icon=":material/play_arrow:",
                                         width="stretch")


def run_assessment(bundle, explainer):
    try:
        patient = PatientInput(**{name: st.session_state[name] for name in FEATURES})
    except InvalidInputError as exc:
        st.session_state.pop("assessment", None)  # never show results for different inputs
        st.error("Please check the following and try again:\n\n" + "\n".join(f"- {e}" for e in exc.errors),
                 icon=":material/warning:")
        return
    result = predict(patient, bundle)
    explanation = explainer.explain(patient)
    st.session_state["assessment"] = (patient, result, explanation, generate(patient, result, explanation))


def conformity_html(patient) -> tuple[str, str]:
    """(meta-grid value, caution block). An unusual input is a caution signal, never a validation error."""
    c = conformity(patient)
    if c is None:
        return "Not available", ""
    if not c["unusual"]:
        return "Within training distribution", ""
    names = ", ".join(FEATURE_LABELS[f].replace(" (from height and weight)", "") for f in c["unusual_features"])
    detail = (f" Values outside the central 99% of the reference data: {escape(names)}." if names else
              " No single value is extreme; the combination of values is uncommon.")
    t = tokens()
    return "Unusual profile", (
        f"<div class='note' style='border-left-color:{t['mid_band']}'><b>CAUTION.</b> This input profile is unusual "
        f"relative to the model's reference population (the training data).{detail} Interpret the model estimate "
        "cautiously. This is a statistical check, not a medical judgement.</div>")


def result_panel(bundle, patient, result, explanation):
    t = tokens()
    band = band_name(result.probability_positive)
    color = band_color(band)
    calibration = bundle.get("calibration", "none").capitalize()
    conformity_value, caution = conformity_html(patient)
    st.html(f"""
<div class='hero-label'>Model estimate</div>
<div style='display:flex;align-items:flex-end;gap:1rem;flex-wrap:wrap'>
  <div class='hero-number'>{result.probability_positive:.1%}</div>
  <div style='padding-bottom:.7rem'>
    <span class='chip chip-band' style='color:{color};border-color:{color}'>{band.upper()}</span>
    <div class='subtle' style='font-size:.78rem;margin-top:.3rem'>Prototype estimate band</div>
  </div>
</div>
<div class='subtle' style='font-size:.9rem'>Model-estimated probability of the dataset's cardiovascular-disease
label for these inputs. Not a diagnosis, and not a clinically validated individual risk.</div>
{band_bar(result.probability_positive)}
{caution}
<div class='meta-grid'>
  <div><div class='meta-k'>Model</div><div class='meta-v'>{MODEL_NAMES[result.model_name]}</div></div>
  <div><div class='meta-k'>Calibration</div><div class='meta-v'>{escape(calibration)} (protocol-selected)</div></div>
  <div><div class='meta-k'>Input conformity</div><div class='meta-v'>{conformity_value}</div></div>
  <div><div class='meta-k'>Reference baseline</div><div class='meta-v'>{explanation.base_probability:.1%}</div></div>
  <div><div class='meta-k'>Class at 0.50</div><div class='meta-v'>{result.predicted_class}
       <span class='subtle' style='font-weight:400'>({'present' if result.predicted_class else 'absent'} label)</span></div></div>
</div>""")
    with st.expander("Probability, band and threshold are three different things"):
        st.markdown(
            "- **Probability**: the model's calibrated estimate for these inputs.\n"
            "- **Display band**: Lower (<30%), Moderate (30–60%), Higher (≥60%), a presentation category made "
            "for this prototype. It is not a clinical threshold.\n"
            "- **Classification threshold**: the 0.50 cut-off that turns the probability into the class output. "
            "Other cut-offs trade false positives against false negatives; see **Model → Thresholds**.\n"
            f"- **Reference baseline**: the model's estimate for an average record in the training data.\n"
            "- **Input conformity**: whether these inputs resemble the training data statistically. An unusual "
            "profile still gets an estimate, with a caution; see **Explain → Assessment reliability**.")

    section("How the model arrived here")
    up = [c for c in explanation.contributions if c.direction == POSITIVE][:3]
    down = [c for c in explanation.contributions if c.direction == NEGATIVE][:3]
    st.html(f"<div style='font-size:.82rem;font-weight:600;margin:.1rem 0 .1rem 0'>▲ Moved the estimate higher</div>"
            f"{factor_rows(up, t['higher'])}"
            f"<div style='font-size:.82rem;font-weight:600;margin:.7rem 0 .1rem 0'>▼ Moved the estimate lower</div>"
            f"{factor_rows(down, t['lower'])}")
    st.caption("Percentages are each input's share of the model's total weighting for this assessment. "
               "Model contribution ≠ medical causation.")
    a, b = st.columns(2)
    a.page_link(PAGES["explain"], label="Full explanation", icon=":material/insights:")
    b.page_link(PAGES["explore"], label="What-if explorer", icon=":material/tune:")


def guidance_section(guidance):
    section("General guidance")
    st.caption("General information from fixed, transparent rules applied to your inputs. Not medical advice; "
               "following it is not a guarantee of any outcome.")
    items = [r for r in guidance.recommendations if r.category is not Category.MODEL_CONTEXT]
    cols = st.columns(2, gap="medium")
    for i, rec in enumerate(items):
        with cols[i % 2].container(border=True):
            head, tag = st.columns([4, 1])
            head.markdown(f"**{rec.title}**")
            with tag:
                st.badge(rec.priority.value.capitalize(), color=PRIORITY_COLORS[rec.priority])
            st.markdown(rec.message)
            st.caption(rec.category.value)
    if guidance.additional:
        with st.expander(f"More guidance ({len(guidance.additional)})"):
            for rec in guidance.additional:
                st.markdown(f"**{rec.title}**: {rec.message}")


def summary_section(patient, result):
    section("Assessment summary")
    values = {
        "Age": readable("age", patient.age), "Gender": readable("gender", patient.gender),
        "Height": readable("height", patient.height), "Weight": readable("weight", patient.weight),
        "BMI": f"{result.bmi:.1f}", "Blood pressure": f"{patient.ap_hi:g}/{patient.ap_lo:g} mmHg",
        "Cholesterol": readable("cholesterol", patient.cholesterol), "Glucose": readable("gluc", patient.gluc),
        "Smoking": readable("smoke", patient.smoke), "Alcohol intake": readable("alco", patient.alco),
        "Physical activity": readable("active", patient.active),
    }
    cells = "".join(f"<div><div class='meta-k'>{k}</div><div class='meta-v'>{escape(v)}</div></div>"
                    for k, v in values.items())
    st.html(f"<div class='meta-grid' style='border-top:none;margin-top:0;padding-top:0'>{cells}</div>")


page_header("Assess", "AI-Powered Healthcare Risk Assessment Assistant",
            "Explainable cardiovascular risk estimation from structured health data.")
st.html("<span class='chip'>Educational research prototype</span><span class='chip'>Not a diagnostic tool</span>"
        "<span class='chip'>Explainable by design</span>")
bundle, explainer = engine()

left, right = st.columns([2, 1], gap="large")
with left:
    if input_form():
        run_assessment(bundle, explainer)
with right:
    with st.container(border=True):
        if "assessment" in st.session_state:
            patient, result, explanation, _ = st.session_state["assessment"]
            result_panel(bundle, patient, result, explanation)
        else:
            st.html("<div class='hero-label'>Model estimate</div>"
                    "<div class='hero-number subtle' style='opacity:.35'>—</div>")
            st.markdown("Enter the 11 inputs, or load a synthetic example, then select **Run Assessment**.\n\n"
                        "You will see the model-estimated probability, a prototype estimate band, the inputs that "
                        "moved the estimate most, and general guidance.")

if "assessment" in st.session_state:
    patient, result, explanation, guidance = st.session_state["assessment"]
    st.divider()
    guidance_section(guidance)
    st.divider()
    summary_section(patient, result)
footer()
