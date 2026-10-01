"""EXPLAIN: detailed SHAP explanation of the current assessment."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explainability import NEGATIVE, POSITIVE
from ui.core import chart, footer, kpis, note, page_header, require_assessment, section, tokens

patient, result, explanation, _ = require_assessment()
t = tokens()
page_header("Explain", "Why did the model estimate this probability?",
            "How the trained model weighted each of your inputs for this assessment.")

kpis([
    ("Reference baseline", f"{explanation.base_probability:.1%}", "average record in the training data"),
    ("This assessment", f"{result.probability_positive:.1%}", "model-estimated probability"),
    ("Difference", f"{(result.probability_positive - explanation.base_probability) * 100:+.1f} pp",
     "from baseline to this estimate"),
])
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

with st.expander("How this explanation is computed"):
    st.markdown(
        f"- **Method:** SHAP (Shapley additive explanations) for the logistic-regression score. For a linear "
        f"model it is exact: each bar is *coefficient × (your value − training average)* after preprocessing.\n"
        f"- **Score vs probability:** contributions add up exactly on the model's internal score "
        f"({explanation.base_value:+.3f} at baseline → {explanation.model_output:+.3f} here). The displayed "
        f"probability is that score passed through **isotonic calibration**, a monotone map learned with "
        f"cross-validation. Calibration keeps every contribution's direction but not its size in percentage "
        f"points, so bar lengths are not percentages. Before calibration the score corresponds to "
        f"{explanation.uncalibrated_probability:.1%}.\n"
        f"- **Height and weight** enter the model only through BMI.\n"
        f"- **Unexpected directions:** in this dataset smokers and drinkers have slightly lower disease rates "
        f"(a confounded pattern), so the model can show smoking moving the estimate lower. That is model "
        f"behaviour, not medical evidence; see **Methodology → Model card**.")
footer()
