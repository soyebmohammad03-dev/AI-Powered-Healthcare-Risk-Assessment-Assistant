"""METHODOLOGY: pipeline, data-quality lab, model card, limitations."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import ROOT
from src.preprocessing import FEATURE_LABELS
from ui.core import analysis, chart, footer, kpis, metrics_report, note, page_header, section, tokens

A, R = analysis(), metrics_report()
DQ = A["data_quality"]
t = tokens()
page_header("Methodology", "How the system was built and evaluated",
            "From raw data to a human-centred interface, with every stage inspectable.")

STAGES = [
    ("Dataset", f"{DQ['raw_rows']:,} records, 11 inputs, binary cardiovascular-disease label (Kaggle)."),
    ("Cleaning", f"Documented data-quality rules remove {DQ['removed_rows']:,} implausible or duplicate rows."),
    ("Feature engineering", "Age in years; BMI from height and weight. Nothing else derived."),
    ("Preprocessing", "Scaling and one-hot encoding inside the model pipeline, fit on training data only."),
    ("Cross-validation", "Stratified 80/20 split; 5-fold CV repeated 5 times (25 splits) on the training split."),
    ("Model training", "Logistic Regression, Random Forest and XGBoost, each raw, sigmoid and isotonic."),
    ("Selection", "Pre-declared protocol: corrected resampled t-tests, no tunable margins, test set unused."),
    ("Evaluation & reliability", "Test-set metrics with bootstrap intervals, subgroups, conformity, stability, shift."),
    ("Explainability", "Exact SHAP (linear or TreeSHAP), permutation importance, partial dependence, what-if."),
    ("Human-centred interface", "Plain-language results, prototype bands, reliability signals and limitations."),
]
st.html("<div class='pipeline'>" + "".join(
    f"<div class='step'><div class='step-n'>{i:02d}</div><div class='step-t'>{name}</div>"
    f"<div class='step-d'>{desc}</div></div>" for i, (name, desc) in enumerate(STAGES, 1)) + "</div>")

tabs = st.tabs(["Data quality lab", "Model card", "Validation strategy", "Limitations",
                "Scope & human-centred design"])

with tabs[0]:
    kpis([("Raw rows", f"{DQ['raw_rows']:,}", "as downloaded"),
          ("Clean rows", f"{DQ['clean_rows']:,}", "used for modelling"),
          ("Removed", f"{DQ['removed_rows']:,}", f"{DQ['removed_rows'] / DQ['raw_rows']:.1%} of raw"),
          ("Missing values", f"{sum(DQ['missing_values_raw'].values())}", f"duplicate ids: {DQ['duplicate_ids_raw']}")])
    section("Exclusions by rule (applied in this order)")
    rules = pd.DataFrame([{"Rule": k.replace("_", " "), "Rows removed": v} for k, v in DQ["removed_by_rule"].items()])
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        st.dataframe(rules, hide_index=True, width="stretch")
    with b:
        rule = st.selectbox("Example excluded rows", list(DQ["examples"]), format_func=lambda r: r.replace("_", " "))
        st.dataframe(pd.DataFrame(DQ["examples"][rule]), hide_index=True, width="stretch")
        st.caption("Up to five rows per rule (age already converted to years). These are recording errors such "
                   "as negative or four-digit blood pressures, not medical judgements.")

    section("Raw vs clean ranges (min, 1st pct, median, 99th pct, max)")
    st.dataframe(pd.DataFrame([{"Input": FEATURE_LABELS[c], "Raw": " · ".join(f"{v:g}" for v in r["raw"]),
                                "Clean": " · ".join(f"{v:g}" for v in r["clean"])} for c, r in DQ["ranges"].items()]),
                 hide_index=True, width="stretch")

    section("Distributions by outcome (clean data)")
    feature = st.segmented_control("Feature", list(DQ["histograms"]), default="ap_hi", format_func=FEATURE_LABELS.get,
                                   key="dq_feature", label_visibility="collapsed") or "ap_hi"
    h = DQ["histograms"][feature]
    centers = [(a + b) / 2 for a, b in zip(h["edges"], h["edges"][1:])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=h["target_0"], name="Label: absent", marker_color=t["lower"], opacity=0.75))
    fig.add_trace(go.Bar(x=centers, y=h["target_1"], name="Label: present", marker_color=t["higher"], opacity=0.75))
    chart(fig, height=300, barmode="overlay", xaxis_title=FEATURE_LABELS[feature], yaxis_title="Records")

    a, b = st.columns(2, gap="large")
    with a:
        section("Disease-label rate by category")
        rows = [{"Input": FEATURE_LABELS[col], "Value": v["label"], "Records": v["n"], "Label rate": v["disease_rate"]}
                for col, vals in DQ["categorical"].items() for v in vals]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=380,
                     column_config={"Label rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    with b:
        section("Correlation between numeric inputs")
        corr = A["data_quality"]["correlation"]
        names = [FEATURE_LABELS[f].replace(" (from height and weight)", "") for f in corr["features"]]
        fig = go.Figure(go.Heatmap(z=corr["matrix"], x=names, y=names, zmin=-1, zmax=1,
                                   colorscale=[[0, t["lower"]], [0.5, t["sunken"]], [1, t["higher"]]],
                                   text=[[f"{v:.2f}" for v in r] for r in corr["matrix"]], texttemplate="%{text}"))
        chart(fig, height=380, yaxis_autorange="reversed")
    note(f"<b>Target balance:</b> raw {DQ['target_raw']['0']:,} absent / {DQ['target_raw']['1']:,} present; "
         f"clean {DQ['target_clean']['0']:,} / {DQ['target_clean']['1']:,}. Smokers and drinkers have slightly "
         "lower label rates than non-smokers and non-drinkers in this data, a confounded pattern the model inherits.")

with tabs[1]:
    st.markdown((ROOT / "docs" / "model_card.md").read_text())

with tabs[2]:
    st.markdown(f"""
- **Split:** stratified 80/20, seed {R['seed']}: {R['split']['train']:,} training and {R['split']['test']:,} test
  records. The test set is scored once per model, after selection; no threshold, calibrator or detector is fit on it.
- **Cross-validation:** stratified 5-fold repeated 5 times (`RepeatedStratifiedKFold`, seed {R['seed']}) on the
  training split, 25 splits per variant. Every learned step (scaler, encoder, model, calibrator) is fit inside
  each training fold; calibrators use a further internal 3-fold split, so no out-of-fold probability comes from
  a model or calibrator that saw that record.
- **Selection protocol:** declared before the repeated-CV run; corrected resampled t-tests at α = 0.05, keeping
  the more interpretable model unless the next is significantly better on both ROC-AUC and Brier. It replaces
  earlier margins that were set after seeing results.
- **Metrics:** accuracy, precision, recall and F1 at 0.50; ROC-AUC and PR-AUC for ranking; log loss, Brier,
  ECE and calibration slope/intercept for probability quality.
- **Uncertainty:** paired percentile bootstrap (1,000 resamples) on test predictions.
- **Subgroups:** gender and age bands 29–39, 40–49, 50–59, 60–65, with within-group bootstrap intervals;
  groups with fewer than 30 records of either class are reported without metrics.
- **Reliability:** input conformity (Mahalanobis distance, fit on training rows only), prediction and
  explanation stability under ±1–5% input changes, monotonicity checks, model disagreement, and a synthetic
  distribution-shift experiment.
- **Reproduce:** `python -m src.train_models && python -m src.analysis && python -m src.reliability &&
  python -m src.shift_analysis`. Detailed numbers: `docs/evaluation.md`.
""")

AGE_AUC = {r["group"]: r["roc_auc"] for r in A["subgroups"]["age_group"] if r["reliable"]}
with tabs[3]:
    st.markdown(f"""
- **Not clinically validated.** One public dataset with limited provenance; test ROC-AUC
  {A['models'][A['final_model']]['test']['roc_auc']:.3f}. Probabilities agree with this dataset's labels, not with
  any clinical population.
- **Step-wise, non-monotone final model.** XGBoost's estimate can jump by many percentage points for a 1–2%
  input change near a split point, and it is not monotone in blood pressure, age or weight for every profile.
- **Explanations are less stable than with a linear model.** Small input changes alter XGBoost's top-5
  contributions more often than Logistic Regression's.
- **Self-reported and coarse inputs.** Smoking, alcohol and activity are self-reported; cholesterol and glucose
  are three-level categories, not lab values.
- **Confounded patterns.** Smoking and alcohol show slightly lower label rates in this data, so the model can
  move the estimate lower for them. This is model behaviour, never medical evidence.
- **Weaker in older ages.** Test ROC-AUC is {AGE_AUC.get('60–65', float('nan')):.2f} for ages 60–65 versus
  {AGE_AUC.get('40–49', float('nan')):.2f} for 40–49.
- **Explanations describe the model.** SHAP, permutation importance, partial dependence and what-if results are
  about this trained model, not about causes of disease. Correlated inputs (systolic/diastolic) share credit.
- **Prototype bands and thresholds.** The <30/30–60/≥60% bands and the 0.50 threshold are presentation and
  analysis devices, not clinical cut-offs.
- **Age coverage.** Inputs are restricted to ages 29–65, the range covered by the data.
""")
with tabs[4]:
    note("<b>Non-clinical scope.</b> This is an educational prototype. It does not diagnose, screen, triage or "
         "recommend treatment, and it is not affiliated with any hospital, clinic or medical institution. Any "
         "health question belongs with a qualified healthcare professional.")
    section("How the interface answers a user's questions")
    st.dataframe(pd.DataFrame([
        {"Question": "What does the system know?", "Where it is answered":
         "Assess: the 11 inputs and their accepted ranges. Methodology: the dataset and cleaning rules."},
        {"Question": "What does it predict?", "Where it is answered":
         "Assess: a model-estimated probability of the dataset label, with a prototype band. It is not a diagnosis."},
        {"Question": "Why does it predict that?", "Where it is answered":
         "Assess: the top contributions. Explain: the full SHAP breakdown, in plain language and as a chart."},
        {"Question": "What does it not know?", "Where it is answered":
         "Assess: \"What the model does not know\". Limitations tab. Input conformity flags unusual profiles."},
        {"Question": "Where is it uncertain?", "Where it is answered":
         "Assess: near-cut-off notes. Explain: reliability signals. Model: bootstrap intervals, subgroups, robustness."},
        {"Question": "What happens if inputs change?", "Where it is answered":
         "Explore: model sensitivity only, including non-monotone responses. It never predicts a medical outcome."},
    ]), hide_index=True, width="stretch")
    st.markdown("""
- **No pressure, no fear.** The app uses neutral wording and shows no alarms or countdowns. A caution never blocks
  an estimate, and nothing asks the user to take a medical action.
- **Separate signals.** Probability, band, threshold, conformity, stability and model disagreement are shown
  separately. Combining them into one "trust score" would need arbitrary weights.
- **Transparent rules.** The guidance comes from fixed, readable rules, with no language model and no named treatments.
- **Accessible presentation.** Every coloured cue has a text label or a ▲/▼ marker. The app supports light and dark
  themes and adapts its layout to narrow screens.
""")
footer()
