"""EXPLAIN: detailed SHAP explanation of the current assessment."""
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import reliability as rel
from src.explainability import NEGATIVE, POSITIVE
from ui.core import (MODEL_NAMES, artifact, candidates, chart, conformity, engine, footer, kpis, note, page_header,
                     require_assessment, section, stability_for, tokens)

patient, result, explanation, _ = require_assessment()
bundle, explainer = engine()
t = tokens()
page_header("Explain", "Why did the model estimate this probability?",
            "How the trained model weighted each of your inputs for this assessment.")

kpis([
    ("Reference baseline", f"{explanation.base_probability:.1%}", "average record in the training data"),
    ("This assessment", f"{result.probability_positive:.1%}", "model-estimated probability"),
    ("Difference", f"{(result.probability_positive - explanation.base_probability) * 100:+.1f} pp",
     "from baseline to this estimate"),
])


def named(cs) -> str:
    items = [f"{c.label} ({c.display_value})" for c in cs]
    return " and ".join(items) if len(items) < 3 else ", ".join(items[:-1]) + " and " + items[-1]


up = [c for c in explanation.contributions if c.direction == POSITIVE][:3]
down = [c for c in explanation.contributions if c.direction == NEGATIVE][:3]
section("In plain language")
st.markdown(
    f"The model starts from a **reference baseline of {explanation.base_probability:.1%}**, its estimate for an "
    f"average record in the training data, and arrives at **{result.probability_positive:.1%}** for these inputs. "
    + (f"The inputs that moved the estimate **higher** most were {named(up)}. " if up else "")
    + (f"The inputs that moved it **lower** most were {named(down)}. " if down else "")
    + "Contributions are measured on the model's internal score, so the chart below shows direction and "
      "relative size, not percentage points.")
note("<b>Model contribution ≠ medical causation.</b> These values show how the trained model weighted the "
     "supplied features. They describe model behaviour, not what causes cardiovascular disease.")

section("Feature contributions")
rows = list(reversed(explanation.contributions))  # largest effect at the top
fig = go.Figure(go.Bar(
    x=[c.shap_value for c in rows],
    y=[f"{'▲' if c.direction == POSITIVE else '▼' if c.direction == NEGATIVE else '•'} {c.label} · {c.display_value}"
       for c in rows],
    orientation="h",
    marker_color=[t["higher"] if c.shap_value > 0 else t["lower"] for c in rows],
    text=[f"{c.shap_value:+.2f}" for c in rows], textposition="outside", cliponaxis=False,
    customdata=[f"{c.relative_importance:.0%} of total weighting" for c in rows],
    hovertemplate="%{y}<br>contribution %{x:+.3f} (score units)<br>%{customdata}<extra></extra>",
))
fig.add_vline(x=0, line_color=t["faint"], line_width=1)
chart(fig, height=70 + 38 * len(rows), xaxis_title="◀ moved the estimate lower      |      moved the estimate higher ▶",
      showlegend=False, margin=dict(l=8, r=40, t=10, b=40))

a, b = st.columns(2, gap="large")
with a:
    section("▲ Moved the estimate higher")
    for c in [c for c in explanation.contributions if c.direction == POSITIVE]:
        st.markdown(f"- **{c.label}** ({c.display_value}): {c.relative_importance:.0%} of the weighting")
with b:
    section("▼ Moved the estimate lower")
    for c in [c for c in explanation.contributions if c.direction == NEGATIVE]:
        st.markdown(f"- **{c.label}** ({c.display_value}): {c.relative_importance:.0%} of the weighting")

with st.expander("All contributions as a table"):
    st.dataframe(pd.DataFrame([{
        "Feature": c.label, "Your value": c.display_value, "Contribution (score)": round(c.shap_value, 4),
        "Share of weighting": c.relative_importance,
        "Direction": "Higher" if c.direction == POSITIVE else "Lower" if c.direction == NEGATIVE else "None",
    } for c in explanation.contributions]), hide_index=True, width="stretch",
        column_config={"Share of weighting": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

score = "log-odds score" if explainer.log_odds else "probability score (average of the trees)"
method = (f"SHAP for the logistic-regression score. For a linear model it is exact: each bar is *coefficient × "
          f"(your value − training average)* after preprocessing." if explainer.linear else
          f"TreeSHAP for {MODEL_NAMES[result.model_name]}, exact for the trees' own output. Unlike a linear "
          f"model, a tree model's contribution for one input can depend on the values of the others (interactions).")
calibration = (f"then **{bundle['calibration']} calibration**, a monotone map learned with cross-validation"
               if explainer.calibrator is not None else "with no calibrator (the protocol kept raw probabilities)")
with st.expander("How this explanation is computed"):
    st.markdown(
        f"- **Method:** {method}\n"
        f"- **Score vs probability:** contributions add up exactly on the model's internal {score} "
        f"({explanation.base_value:+.3f} at baseline → {explanation.model_output:+.3f} here). The displayed "
        f"probability is that score passed through {'the logistic link, ' if explainer.log_odds else ''}{calibration}. "
        f"Every step keeps each contribution's direction but not its size in percentage points, so bar lengths "
        f"are not percentages.\n"
        f"- **Height and weight** enter the model only through BMI.\n"
        f"- **Unexpected directions:** in this dataset smokers and drinkers have slightly lower disease rates "
        f"(a confounded pattern), so the model can show smoking moving the estimate lower. That is model "
        f"behaviour, not medical evidence; see **Methodology → Model card**.")

# ---- assessment reliability ------------------------------------------------------------------
section("Assessment reliability")
st.caption("Separate signals about this estimate, shown side by side on purpose. They are not combined into a "
           "single score: they measure different things, and any weighting would be arbitrary.")
R = artifact("reliability.json")
conf = conformity(patient)
stab = stability_for(patient, explainer)
spread = None
if R:
    spread = rel.model_disagreement(patient, candidates(), R["disagreement"]["spread_percentiles"])
    cal = R["calibration"]["models"][result.model_name]["variants"][bundle["calibration"]]["stats"]
cells = [
    ("Probability", f"{result.probability_positive:.1%}", f"{MODEL_NAMES[result.model_name]}, model estimate"),
    ("Input conformity", "Not available" if conf is None else "Unusual profile" if conf["unusual"]
     else "Within reference distribution",
     "" if conf is None else f"distance larger than {min(conf['reference_percentile'], 99.9):.1f}% of reference records"),
    ("Explanation stability",
     "Stable under small perturbations" if stab["share_full_top_k_overlap"] == 1
     else f"Top-5 changed in {round((1 - stab['share_full_top_k_overlap']) * stab['n'])} of {stab['n']}",
     f"±1–5% of one input at a time; probability moved up to {stab['max_change_pp']:.1f} pp"),
    ("Model disagreement", "Not available" if spread is None else f"{spread['spread_pp']:.1f} pp spread",
     "" if spread is None else f"across 3 candidate models; larger than {spread['test_percentile']:.0f}% of test records"),
    ("Calibration status", f"{bundle['calibration'].capitalize()} probabilities",
     "" if not R else f"out-of-fold ECE {cal['ece']['mean']:.3f}, slope {cal['slope']['mean']:.2f}"),
]
st.html("<div class='meta-grid' style='border-top:none;margin-top:0;padding-top:0'>" + "".join(
    f"<div><div class='meta-k'>{escape(k)}</div><div class='meta-v'>{escape(v)}</div>"
    f"<div class='subtle' style='font-size:.78rem'>{escape(d)}</div></div>" for k, v, d in cells) + "</div>")
if spread:
    with st.expander("Prediction variation across candidate models"):
        st.dataframe(pd.DataFrame([{"Model": MODEL_NAMES[n] + (" (final)" if n == result.model_name else ""),
                                    "Model estimate": p} for n, p in spread["probabilities"].items()]),
                     hide_index=True, width="stretch",
                     column_config={"Model estimate": st.column_config.NumberColumn(format="percent")})
        st.caption("Models that fit this data almost equally well can still give different estimates for the same "
                   "person. This is model disagreement, a consequence of model assumptions, not a measure of "
                   "medical uncertainty.")
with st.expander("What these signals mean"):
    st.markdown(
        "- **Input conformity:** a Mahalanobis distance of the inputs from the training data, flagged above the "
        "99th percentile of reference records. Unusual is not invalid.\n"
        "- **Explanation stability:** whether the five largest contributions stay the same when one numeric input "
        "is moved by ±1%, ±2% or ±5%.\n"
        "- **Model disagreement:** highest minus lowest estimate among Logistic Regression, Random Forest and "
        "XGBoost, compared with the spread on the held-out test records.\n"
        "- **Calibration status:** how closely this model's probabilities matched observed frequencies in "
        "cross-validation on this dataset. It is not clinical accuracy.")
footer()
