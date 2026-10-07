"""EXPLORE: what-if / model sensitivity analysis around the current assessment."""
import plotly.graph_objects as go
import streamlit as st

from src.prediction import InvalidInputError
from src.preprocessing import CATEGORY_LABELS, FEATURE_LABELS, PLAUSIBLE
from src.what_if import ADJUSTABLE, changes_from, compare, response_curve
from ui.core import (MODEL_NAMES, artifact, assessment_context, chart, engine, footer, kpis, note, page_header, readable,
                     require_assessment, section, step_note, tokens)

patient, result, _, _ = require_assessment()
bundle, _ = engine()
t = tokens()
page_header("Explore", "Explore what changes the model estimate",
            "Change selected inputs and compare the model's estimate with your assessment.")
assessment_context(patient, result, "explore")
note("<b>This is model sensitivity analysis, not a predicted medical outcome.</b> It shows how the trained "
     "model's estimate responds when one or more inputs change while all other inputs stay fixed.")

KEY = "wi_"


def reset():
    for name in ADJUSTABLE:
        st.session_state[KEY + name] = float(getattr(patient, name)) if name in ("ap_hi", "ap_lo", "weight") \
            else getattr(patient, name)


# New assessment -> start from the baseline. Returning from another page (Streamlit drops widget state
# for pages not shown) -> restore the scenario from the plain copy kept below.
if st.session_state.get(KEY + "baseline") != result.inputs:
    reset()
    st.session_state[KEY + "baseline"] = result.inputs
elif any(KEY + n not in st.session_state for n in ADJUSTABLE):
    for name, value in st.session_state.get(KEY + "saved", {}).items():
        st.session_state[KEY + name] = value

controls, results = st.columns([1, 1.35], gap="large")
with controls:
    with st.container(border=True):
        section("Scenario inputs")
        for name, step in (("ap_hi", 1.0), ("ap_lo", 1.0), ("weight", 0.5)):
            low, high = PLAUSIBLE[name]
            st.slider(f"{FEATURE_LABELS[name]} ({'kg' if name == 'weight' else 'mmHg'})", float(low), float(high),
                      step=step, key=KEY + name, format="%.1f" if name == "weight" else "%.0f")
        for name in ("cholesterol", "gluc", "smoke", "active", "alco"):
            st.segmented_control(FEATURE_LABELS[name], list(CATEGORY_LABELS[name]), key=KEY + name,
                                 format_func=CATEGORY_LABELS[name].get, width="stretch")
        st.button("Reset to baseline", on_click=reset, icon=":material/restart_alt:", width="stretch")
        st.caption("Age, gender and height stay as assessed. Combinations outside the validated input domain "
                   "(e.g. diastolic ≥ systolic, implausible BMI) are rejected.")

values = {name: st.session_state[KEY + name] for name in ADJUSTABLE}
st.session_state[KEY + "saved"] = dict(values)
with results:
    if any(v is None for v in values.values()):
        st.warning("Select a value for every scenario input.", icon=":material/warning:")
        st.stop()
    try:
        outcome = compare(patient, changes_from(patient, values), bundle)
    except InvalidInputError as exc:
        st.error("This scenario is outside the validated input domain:\n\n" +
                 "\n".join(f"- {e}" for e in exc.errors), icon=":material/block:")
        st.stop()

    base, scen = outcome.baseline.probability_positive, outcome.scenario.probability_positive
    kpis([("Baseline", f"{base:.1%}", "your assessment"), ("Scenario", f"{scen:.1%}", "with the changes"),
          ("Change", f"{outcome.delta_pp:+.1f} pp",
           "higher than baseline" if outcome.delta_pp >= 0.05 else "lower than baseline"
           if outcome.delta_pp <= -0.05 else "no change")])
    if outcome.changes:
        changed = ", ".join(f"{FEATURE_LABELS[k]} {readable(k, a)} → {readable(k, b)}"
                            for k, (a, b) in outcome.changes.items())
        st.markdown(f"Under the model, changing **{changed}** while keeping other inputs fixed changed the "
                    f"estimated probability by **{outcome.delta_pp:+.1f} percentage points**.")
        if len(outcome.changes) > 1:
            section("Each change on its own")
            items = [(FEATURE_LABELS[k], v) for k, v in outcome.single_effects.items()]
            fig = go.Figure(go.Bar(
                y=[k for k, v in items], x=[v if v is not None else 0 for k, v in items], orientation="h",
                marker_color=[t["higher"] if (v or 0) > 0 else t["lower"] for _, v in items],
                text=[f"{v:+.1f} pp" if v is not None else "not valid alone" for _, v in items],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{text}<extra></extra>"))
            chart(fig, height=60 + 40 * len(items), showlegend=False, xaxis_title="change in percentage points",
                  margin=dict(l=8, r=60, t=10, b=40))
            st.caption("Single-change effects need not add up to the combined change: effects combine on the "
                       "model's internal score, and the probability is not linear in it.")
    else:
        st.markdown("Adjust a scenario input to compare it with your assessment.")

    section("Model estimate across one input")
    feature = st.segmented_control("Input", ["ap_hi", "ap_lo", "weight"], default="ap_hi",
                                   format_func=FEATURE_LABELS.get, key=KEY + "curve", label_visibility="collapsed")
    feature = feature or "ap_hi"
    curve = response_curve(patient, feature, bundle)
    fig = go.Figure(go.Scatter(x=curve["value"], y=curve["probability"], mode="lines", line_shape="hv",
                               line=dict(color=t["primary"], width=2.5), name="Model estimate",
                               hovertemplate="%{x:g}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[getattr(patient, feature)], y=[base], mode="markers", name="Baseline",
                             marker=dict(size=11, color=t["text"], symbol="circle")))
    fig.add_trace(go.Scatter(x=[values[feature]], y=[scen], mode="markers", name="Scenario",
                             marker=dict(size=12, color=t["higher"], symbol="diamond")))
    chart(fig, height=320, yaxis_tickformat=".0%", yaxis_range=[0, 1],
          xaxis_title=f"{FEATURE_LABELS[feature]} (other inputs as in your assessment)")
    st.caption("The curve varies only this input, holding every other input at your assessed values; values "
               f"that would be invalid (e.g. systolic not above diastolic) are omitted. {step_note(bundle)}")
    mono = (artifact("reliability.json") or {}).get("monotonicity", {}).get("features", {}).get(feature)
    counts = mono and mono["models"].get(bundle["model_name"], {}).get("counts")
    if counts and counts["non-monotone"]:
        n = sum(counts.values())
        note(f"<b>Not always \"more is higher\".</b> In the robustness analysis, {MODEL_NAMES[bundle['model_name']]}'s "
             f"estimate was non-monotone in {FEATURE_LABELS[feature].lower()} for {counts['non-monotone']} of {n} "
             "tested profiles: raising this input sometimes lowered the estimate. Tree-based models can respond "
             "non-linearly like this. It is model behaviour, not a medical effect; see "
             "<b>Model → Robustness</b>.")
footer()
