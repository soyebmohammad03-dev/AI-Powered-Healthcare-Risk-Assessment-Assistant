"""Streamlit UI: Cardiovascular Risk Assessment Assistant (educational prototype).

Run: streamlit run app.py

The UI only collects input and presents results. Validation, prediction, SHAP and guidance all come
from the existing src modules; no model logic lives here.
"""
import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import prediction
from src.data_loader import ROOT
from src.explainability import POSITIVE, ModelExplainer
from src.prediction import DEMO_INPUTS, InvalidInputError, ModelArtifactError, PatientInput, load_model, predict
from src.preprocessing import CATEGORY_LABELS, FEATURE_LABELS, FEATURES, PLAUSIBLE
from src.recommendations import DISCLAIMER, HIGHER_FROM, LOWER_BELOW, Category, Priority, generate

HIGHER_COLOR, LOWER_COLOR = "#C2410C", "#0F766E"  # estimate moved higher / lower
UNITS = {"age": "years", "height": "cm", "weight": "kg", "ap_hi": "mmHg", "ap_lo": "mmHg"}
BAND_COLORS = {"LOWER": "green", "MODERATE": "orange", "HIGHER": "red"}
PRIORITY_COLORS = {Priority.HIGH: "red", Priority.MODERATE: "orange", Priority.LOW: "green", Priority.INFO: "gray"}
TRAIN_COMMAND = "python -m src.train_models"
MODEL_NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}


# ---- model loading ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the trained model…")
def load_engine(model_path: str):
    """Persisted pipeline + SHAP explainer, loaded once per server. Errors are not cached."""
    bundle = load_model(Path(model_path))
    return bundle, ModelExplainer(bundle)


def engine():
    """Return (bundle, explainer) or stop the page with a setup message instead of a traceback."""
    try:
        return load_engine(str(prediction.FINAL_MODEL_PATH))
    except ModelArtifactError:
        st.error("The trained model has not been generated yet, or the saved file cannot be used.",
                 icon=":material/error:")
    except Exception as exc:  # e.g. dataset download for the SHAP background failed while offline
        st.error(f"The assessment engine could not be started ({type(exc).__name__}).", icon=":material/error:")
    st.markdown("From the project folder, run this once, then reload the page:")
    st.code(TRAIN_COMMAND, language="bash")
    st.stop()


# ---- formatting helpers -----------------------------------------------------------------------

def readable(feature: str, value) -> str:
    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature][int(value)]
    return f"{value:g} {UNITS[feature]}" if feature in UNITS else f"{value:g}"


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def band_bar(p: float) -> str:
    """Static 0-100% track with the three prototype bands and a marker at p."""
    lo, hi = LOWER_BELOW * 100, HIGHER_FROM * 100
    seg = "height:12px;display:inline-block"
    return f"""
<div style="position:relative;margin:0.4rem 0 0.2rem 0">
  <div style="border-radius:6px;overflow:hidden;white-space:nowrap;font-size:0">
    <span style="{seg};width:{lo}%;background:#D1FAE5"></span><span style="{seg};width:{hi - lo}%;background:#FEF3C7"></span><span style="{seg};width:{100 - hi}%;background:#FEE2E2"></span>
  </div>
  <div style="position:absolute;top:-5px;left:calc({p * 100:.1f}% - 2px);width:4px;height:22px;background:#1F2933;border-radius:2px"></div>
  <div style="display:flex;font-size:0.75rem;color:#52606D;margin-top:4px">
    <span style="width:{lo}%">Lower (&lt;{lo:.0f}%)</span><span style="width:{hi - lo}%">Moderate ({lo:.0f}–{hi:.0f}%)</span><span>Higher (≥{hi:.0f}%)</span>
  </div>
</div>"""


# ---- assessment page --------------------------------------------------------------------------

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

    st.markdown("Fill in all fields, or load a synthetic example:")
    cols = st.columns(len(DEMO_INPUTS) + 1)
    for col, (label, patient) in zip(cols, DEMO_INPUTS.items()):
        col.button(label, key=f"demo_{label}", on_click=fill_form, args=(asdict(patient),),
                   icon=":material/person:", width="stretch")
    cols[-1].button("Clear form", key="clear", on_click=fill_form, args=(None,),
                    icon=":material/restart_alt:", width="stretch")
    st.caption("Examples A, B and C are hand-written demonstration inputs, not real patients.")

    def bounds(name):
        return dict(min_value=PLAUSIBLE[name][0], max_value=PLAUSIBLE[name][1],
                    placeholder=f"{PLAUSIBLE[name][0]}–{PLAUSIBLE[name][1]}")

    def choice(name, help=None):
        st.radio(FEATURE_LABELS[name], list(CATEGORY_LABELS[name]), key=name, horizontal=True,
                 format_func=CATEGORY_LABELS[name].get, help=help)

    with st.form("assessment_form", border=True):
        st.markdown("##### :material/person: About you")
        c = st.columns(4)
        with c[0]:
            st.number_input("Age (years)", key="age", step=1, **bounds("age"),
                            help="The training data covers ages 29–65, so only this range is accepted.")
        with c[1]:
            choice("gender")
        with c[2]:
            st.number_input("Height (cm)", key="height", step=1, **bounds("height"))
        with c[3]:
            st.number_input("Weight (kg)", key="weight", step=0.5, format="%.1f",
                            min_value=float(PLAUSIBLE["weight"][0]), max_value=float(PLAUSIBLE["weight"][1]),
                            placeholder="30–250")

        st.markdown("##### :material/cardiology: Blood pressure")
        c = st.columns(4)
        with c[0]:
            st.number_input("Systolic (upper number, mmHg)", key="ap_hi", step=1, **bounds("ap_hi"))
        with c[1]:
            st.number_input("Diastolic (lower number, mmHg)", key="ap_lo", step=1, **bounds("ap_lo"))

        st.markdown("##### :material/bloodtype: Blood test results")
        c = st.columns(2)
        with c[0]:
            choice("cholesterol", help="As reported by a test, compared with the normal range.")
        with c[1]:
            choice("gluc", help="Blood glucose as reported by a test, compared with the normal range.")

        st.markdown("##### :material/directions_walk: Lifestyle")
        c = st.columns(3)
        with c[0]:
            choice("smoke")
        with c[1]:
            choice("alco")
        with c[2]:
            choice("active", help="Whether you are regularly physically active.")

        return st.form_submit_button("Assess Risk", type="primary", icon=":material/monitoring:")


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


def show_result(result, guidance, explanation):
    band = guidance.risk_category.name.capitalize()
    st.subheader(":material/analytics: Assessment result")
    with st.container(border=True):
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.metric("Model-estimated probability", f"{result.probability_positive:.1%}",
                      help="Probability of the dataset's 'cardiovascular disease present' label, as estimated "
                           "by the trained model for these inputs.")
            st.caption("of cardiovascular disease, as estimated by the model")
            st.badge(f"Prototype estimate band: {band}", color=BAND_COLORS[guidance.risk_category.name])
        with right:
            st.markdown(band_bar(result.probability_positive), unsafe_allow_html=True)
            st.markdown("This is the probability estimated by the trained model for the supplied inputs. "
                        "It is **not a diagnosis** or a clinically validated individual risk score.")
            st.caption("The Lower / Moderate / Higher bands are display categories created for this prototype. "
                       "They are not clinically validated thresholds.")
        with st.expander("Why do you see this? / technical details"):
            st.markdown(
                "The model uses patterns learned from a public research dataset of about 68,500 records to "
                "estimate the probability associated with the supplied inputs.")
            st.markdown(
                f"- Model: {MODEL_NAMES[result.model_name]}\n"
                f"- Class output at the 0.5 threshold: **{result.predicted_class}** (the dataset labels "
                f"1 = disease present, 0 = absent). Shown for transparency only; it is not a diagnosis.\n"
                f"- Model output: {explanation.model_output:+.3f} log-odds")


def show_explanation(explanation, result):
    st.subheader(":material/insights: Why did the model estimate this probability?")
    st.markdown("These values show how the trained model weighted the supplied features. "
                "They describe **model behaviour, not medical causation**.")
    with st.container(border=True):
        rows = list(reversed(explanation.contributions))  # largest effect at the top
        fig = go.Figure(go.Bar(
            x=[c.shap_value for c in rows],
            y=[f"{c.label}: {readable(c.feature, c.value) if c.feature != 'bmi' else c.display_value}" for c in rows],
            orientation="h",
            marker_color=[HIGHER_COLOR if c.shap_value > 0 else LOWER_COLOR for c in rows],
            customdata=["Moved the model estimate higher" if c.shap_value > 0 else
                        "Moved the model estimate lower" for c in rows],
            hovertemplate="%{y}<br>%{customdata}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="#9AA5B1", line_width=1)
        fig.update_layout(height=60 + 34 * len(rows), margin=dict(l=10, r=10, t=10, b=40),
                          xaxis_title="← moved the estimate lower      |      moved the estimate higher →",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                          font=dict(size=13))
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        up = [c for c in explanation.contributions if c.direction == POSITIVE][:3]
        down = [c for c in explanation.contributions if c.shap_value < 0][:3]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**:material/arrow_upward: Moved the model estimate higher**")
            st.markdown("\n".join(f"- {c.label} ({c.display_value})" for c in up) or "- None")
        with c2:
            st.markdown("**:material/arrow_downward: Moved the model estimate lower**")
            st.markdown("\n".join(f"- {c.label} ({c.display_value})" for c in down) or "- None")

    with st.expander("How to read this chart"):
        st.markdown(
            f"The model starts from its estimate for an average record in the training data "
            f"(**{sigmoid(explanation.base_value):.1%}**). Each bar shows how much one input moved the "
            f"estimate up or down from there; together they produce this result "
            f"(**{result.probability_positive:.1%}**). Longer bars had more influence on the model.\n\n"
            "Height and weight are used through BMI. Bar lengths are on the model's internal (log-odds) "
            "scale, so they are not percentage points. A feature can move the estimate in a direction that "
            "differs from medical knowledge when the training data contains unusual patterns; see "
            "*About & Methodology*.")


def show_guidance(guidance):
    st.subheader(":material/lightbulb: General guidance")
    st.caption("General informational guidance generated by fixed, transparent rules from your inputs. "
               "It is not medical advice.")
    items = [r for r in guidance.recommendations if r.category is not Category.MODEL_CONTEXT]
    for rec in items:
        with st.container(border=True):
            head, tag = st.columns([5, 1])
            head.markdown(f"**{rec.title}**")
            with tag:
                st.badge(rec.priority.value.capitalize(), color=PRIORITY_COLORS[rec.priority])
            st.markdown(rec.message)
            st.caption(rec.category.value)
    if guidance.additional:
        with st.expander(f"More guidance ({len(guidance.additional)})"):
            for rec in guidance.additional:
                st.markdown(f"**{rec.title}**: {rec.message}")


def show_summary(patient, result):
    st.subheader(":material/assignment: Assessment summary")
    values = {
        "Age": readable("age", patient.age),
        "Gender": readable("gender", patient.gender),
        "Height": readable("height", patient.height),
        "Weight": readable("weight", patient.weight),
        "BMI": f"{result.bmi:.1f}",
        "Blood pressure": f"{patient.ap_hi:g}/{patient.ap_lo:g} mmHg",
        "Cholesterol": readable("cholesterol", patient.cholesterol),
        "Glucose": readable("gluc", patient.gluc),
        "Smoking": readable("smoke", patient.smoke),
        "Alcohol intake": readable("alco", patient.alco),
        "Physical activity": readable("active", patient.active),
    }
    with st.container(border=True):
        cols = st.columns(3)
        for i, (label, value) in enumerate(values.items()):
            cols[i % 3].markdown(f"<span style='color:#52606D'>{label}</span><br>**{value}**",
                                 unsafe_allow_html=True)


def assessment_page():
    st.title("Cardiovascular Risk Assessment Assistant")
    st.markdown("An explainable AI prototype that estimates a cardiovascular disease probability from "
                "11 health inputs, shows how the model reached it, and offers general information.")
    st.warning(DISCLAIMER, icon=":material/info:")
    bundle, explainer = engine()

    if input_form():
        run_assessment(bundle, explainer)

    if "assessment" in st.session_state:
        patient, result, explanation, guidance = st.session_state["assessment"]
        st.divider()
        show_result(result, guidance, explanation)
        show_explanation(explanation, result)
        show_guidance(guidance)
        show_summary(patient, result)

    st.divider()
    st.caption(f"**Limitations.** {DISCLAIMER} The model was trained on one public dataset with self-reported "
               "lifestyle fields and coarse cholesterol/glucose categories; see *About & Methodology*.")


# ---- about page -------------------------------------------------------------------------------

@st.cache_data(show_spinner="Computing global feature importance…")
def global_importance(model_path: str) -> pd.DataFrame:
    _, explainer = load_engine(model_path)
    return pd.DataFrame([(g.label, g.mean_abs_shap, g.relative_importance) for g in explainer.global_importance()],
                        columns=["Feature", "Mean |SHAP|", "Share"])


def about_page():
    st.title("About & Methodology")
    st.warning(DISCLAIMER, icon=":material/info:")
    engine()
    report = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    data = report["dataset"]

    st.subheader("Purpose")
    st.markdown("An educational, human-centred AI product: it estimates a model probability, explains it "
                "with SHAP, and adds rule-based general information. It does not diagnose, prescribe or "
                "replace a healthcare professional, and it has not been clinically validated.")

    st.subheader("Dataset")
    st.markdown(
        f"[Cardiovascular Disease dataset]({data['source']}) (Kaggle): **{data['raw_records']:,}** records, "
        f"**{data['clean_records']:,}** after removing duplicates and implausible measurements "
        f"(e.g. blood pressure readings of 16,020 mmHg). Target: cardiovascular disease present / absent "
        f"(classes {data['class_counts']['0']:,} / {data['class_counts']['1']:,}). Smoking, alcohol and "
        "activity are self-reported; cholesterol and glucose are three-level categories.")
    with st.expander("Rows removed per data-quality rule"):
        st.dataframe(pd.DataFrame(data["removed_by_rule"].items(), columns=["Rule", "Rows removed"]),
                     hide_index=True)

    st.subheader("Model comparison")
    st.markdown(f"Stratified 80/20 split ({report['split']['train']:,} train / {report['split']['test']:,} "
                "test) and 5-fold cross-validation on the training set. All values are measured.")
    rows = []
    for name, r in report["models"].items():
        for split, m in (("5-fold CV", {k: v["mean"] for k, v in r["cv_train_5fold"].items()}), ("Test", r["test"])):
            rows.append({"Model": MODEL_NAMES[name], "Evaluation": split,
                         **{k: round(m[k], 3) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}})
    st.dataframe(pd.DataFrame(rows).rename(columns={"accuracy": "Accuracy", "precision": "Precision",
                                                    "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")})
    final = report["final_model"]
    st.markdown(f"**Selected model: {MODEL_NAMES[final]}.** {report['final_model_reason']}")
    cm = report["models"][final]["test"]["confusion_matrix"]
    with st.expander("Confusion matrix of the selected model (test set)"):
        st.dataframe(pd.DataFrame(cm, index=["Actual: absent", "Actual: present"],
                                  columns=["Predicted: absent", "Predicted: present"]))

    st.subheader("Explainable AI (SHAP)")
    st.markdown(
        "SHAP splits one prediction into a contribution per input: the model's starting point for an "
        "average record plus all contributions gives its output. For this linear model the values are "
        "exact. Contributions describe how the **model** weighted the inputs, not what causes disease.")
    imp = global_importance(str(prediction.FINAL_MODEL_PATH))
    fig = go.Figure(go.Bar(x=imp["Share"][::-1], y=imp["Feature"][::-1], orientation="h",
                           marker_color=LOWER_COLOR, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=30), xaxis_tickformat=".0%",
                      xaxis_title="Share of average influence across the dataset",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.subheader("General guidance")
    st.markdown(
        "Guidance comes from fixed, readable rules on your inputs (for example blood pressure of 130/80 "
        "mmHg or more, cholesterol above normal, smoking, BMI outside 18.5–25) and on the model's band. "
        "No language model is used. Thresholds are prototype prompts, not diagnostic cut-offs.")

    st.subheader("Limitations")
    st.markdown(
        "- Not clinically validated; about 0.79 ROC-AUC on one public dataset with limited provenance.\n"
        "- The prototype bands (<30%, 30–60%, ≥60%) are display categories, not clinical thresholds.\n"
        "- In this dataset, smokers and drinkers show slightly *lower* disease rates (a confounded pattern), "
        "so the model gives smoking a small negative contribution. This is model behaviour, not medical "
        "evidence; the guidance still addresses smoking.\n"
        "- Inputs are limited to ages 29–65, the range covered by the data.\n"
        "- SHAP treats features as independent; correlated inputs can share credit unintuitively.")


if __name__ == "__main__":
    st.set_page_config(page_title="Cardiovascular Risk Assessment Assistant",
                       page_icon=":material/monitor_heart:", layout="wide")
    st.navigation([
        st.Page(assessment_page, title="Risk Assessment", icon=":material/monitor_heart:", default=True),
        st.Page(about_page, title="About & Methodology", icon=":material/info:", url_path="about"),
    ]).run()
