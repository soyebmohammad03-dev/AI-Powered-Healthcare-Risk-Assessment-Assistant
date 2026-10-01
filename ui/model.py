"""MODEL: comparison, calibration, discrimination, thresholds, explainability, subgroups, uncertainty.
Everything is read from artifacts/metrics.json and artifacts/analysis.json (precomputed, measured)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.core import (MODEL_COLORS, MODEL_NAMES, analysis, chart, footer, kpis, metrics_report, note, page_header,
                     section, tokens)

A, R = analysis(), metrics_report()
t = tokens()
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
                 "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "log_loss": "Log loss", "brier": "Brier"}
NUM = {m: st.column_config.NumberColumn(format="%.3f") for m in METRIC_LABELS.values()}


def short(label: str) -> str:
    return label.replace(" (from height and weight)", "")

page_header("Model", "Model analytics",
            f"How the three candidate models compare, how trustworthy the probabilities are, and where the selected "
            f"model behaves differently. Test set: {A['test_rows']:,} held-out records.")
tabs = st.tabs(["Overview", "Calibration", "ROC & PR", "Thresholds", "Explainability", "Subgroups", "Uncertainty"])

# ---- overview ---------------------------------------------------------------------------------
with tabs[0]:
    sel = R["selection"]
    kpis([("Selected model", MODEL_NAMES[sel["model"]], f"{sel['variant']} calibration"),
          ("CV ROC-AUC", f"{R['comparison'][sel['model']][sel['variant']]['cv']['roc_auc']['mean']:.3f}",
           "5-fold, training split"),
          ("Test ROC-AUC", f"{A['models'][sel['model']]['test']['roc_auc']:.3f}", "held-out"),
          ("Test Brier", f"{A['models'][sel['model']]['test']['brier']:.3f}", "lower is better")])
    section("Comparison (each model in its selected calibration variant)")
    rows = []
    for name, comp in R["comparison"].items():
        v = comp["selected"]
        for split, m in (("5-fold CV (mean)", {k: x["mean"] for k, x in comp[v]["cv"].items()}),
                         ("Test", comp[v]["test"])):
            rows.append({"Model": MODEL_NAMES[name], "Variant": v, "Evaluation": split,
                         **{METRIC_LABELS[k]: m[k] for k in METRIC_LABELS}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=NUM)
    st.caption("Accuracy, precision, recall and F1 use the 0.50 threshold. ROC-AUC and PR-AUC measure ranking; "
               "log loss and Brier measure probability quality (lower is better).")

    section("Stability across the 5 CV folds (ROC-AUC)")
    fig = go.Figure()
    for name, comp in R["comparison"].items():
        folds = comp[comp["selected"]]["cv"]["roc_auc"]["folds"]
        fig.add_trace(go.Scatter(x=[f"Fold {i + 1}" for i in range(len(folds))], y=folds, mode="lines+markers",
                                 name=MODEL_NAMES[name], line=dict(color=MODEL_COLORS[name])))
    chart(fig, height=280, yaxis_title="ROC-AUC")

    section("Selection framework")
    st.markdown(
        f"1. **Calibration per model:** keep raw probabilities unless a calibrator lowers mean CV Brier by at least "
        f"{sel['brier_margin']} *and* in every fold.\n"
        f"2. **Model:** start from the most interpretable model ({', '.join(MODEL_NAMES[m] for m in sel['interpretability_order'])}) "
        f"and move to a less transparent one only if it improves CV ROC-AUC by at least {sel['auc_margin']} "
        "without worse calibration.\n3. The test set is never used to choose.")
    for step in sel["steps"]:
        st.markdown(f"- `{step}`")
    for name, comp in R["comparison"].items():
        st.markdown(f"- **{MODEL_NAMES[name]}:** {comp['calibration_reason']}")
    note("The margins (0.02 ROC-AUC, 0.001 Brier) are judgement calls about practical relevance, fixed during "
         "this upgrade after the first comparison was known. The tree models are measurably better at ranking "
         "(see Uncertainty); Logistic Regression is kept because its explanations are exact and monotone.")

# ---- calibration ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("**Question:** when the model says 70%, does the outcome occur about 70% of the time? "
                "Points on the diagonal mean well-calibrated probabilities. Out-of-fold predictions on the "
                "training split, so no point was predicted by a model that saw it.")
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                               line=dict(color=t["faint"], dash="dot")))
    series = [("logistic_regression", "raw", "Logistic Regression (raw)", "dash"),
              ("logistic_regression", "isotonic", "Logistic Regression (isotonic, selected)", "solid"),
              ("random_forest", "raw", "Random Forest (raw)", "solid"), ("xgboost", "raw", "XGBoost (raw)", "solid")]
    for name, variant, label, dash in series:
        bins = R["comparison"][name][variant]["oof_calibration"]
        fig.add_trace(go.Scatter(x=[b["mean_predicted"] for b in bins], y=[b["observed_rate"] for b in bins],
                                 mode="lines+markers", name=label, line=dict(color=MODEL_COLORS[name], dash=dash),
                                 customdata=[b["count"] for b in bins],
                                 hovertemplate="predicted %{x:.2f}, observed %{y:.2f} (n=%{customdata})<extra></extra>"))
    chart(fig, height=440, xaxis_title="Mean predicted probability", yaxis_title="Observed rate",
          xaxis_range=[0, 1], yaxis_range=[0, 1])
    rows = [{"Model": MODEL_NAMES[n], "Variant": v, "CV Brier": R["comparison"][n][v]["cv"]["brier"]["mean"],
             "CV log loss": R["comparison"][n][v]["cv"]["log_loss"]["mean"],
             "CV ROC-AUC": R["comparison"][n][v]["cv"]["roc_auc"]["mean"],
             "Selected": "✓" if R["comparison"][n]["selected"] == v else ""}
            for n in R["comparison"] for v in ("raw", "sigmoid", "isotonic")]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("CV Brier", "CV log loss", "CV ROC-AUC")})
    note("<b>Finding.</b> Raw logistic regression over-estimated in the 25–35% range and at the top, and "
         "under-estimated around 55–75%. Isotonic calibration corrected this (lower Brier and log loss in every "
         "fold) without changing ranking (ROC-AUC); its step function slightly lowers PR-AUC. Sigmoid "
         "calibration changes nothing for logistic regression, which already uses a sigmoid. Calibration here "
         "means agreement with this dataset's labels, not clinical validity.")

# ---- ROC & PR ---------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("**Question:** how well does each model rank records with the disease label above those without, "
                "at every possible threshold?")
    a, b = st.columns(2, gap="large")
    with a:
        fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=t["faint"], dash="dot"),
                                   name="Chance"))
        for name, m in A["models"].items():
            fig.add_trace(go.Scatter(x=m["roc"]["x"], y=m["roc"]["y"], mode="lines", line=dict(color=MODEL_COLORS[name]),
                                     name=f"{MODEL_NAMES[name]} ({m['test']['roc_auc']:.3f})"))
        chart(fig, height=380, xaxis_title="False positive rate", yaxis_title="True positive rate (recall)",
              title=dict(text="ROC curve (test)", font=dict(size=14)))
    with b:
        prevalence = A["data_quality"]["target_clean"]["1"] / A["data_quality"]["clean_rows"]
        fig = go.Figure(go.Scatter(x=[0, 1], y=[prevalence] * 2, mode="lines", line=dict(color=t["faint"], dash="dot"),
                                   name="Chance (prevalence)"))
        for name, m in A["models"].items():
            fig.add_trace(go.Scatter(x=m["pr"]["x"], y=m["pr"]["y"], mode="lines", line=dict(color=MODEL_COLORS[name]),
                                     name=f"{MODEL_NAMES[name]} ({m['test']['pr_auc']:.3f})"))
        chart(fig, height=380, xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0.4, 1],
              title=dict(text="Precision–recall curve (test)", font=dict(size=14)))
    section("Confusion matrices at the 0.50 threshold (test)")
    cols = st.columns(3)
    for col, (name, m) in zip(cols, A["models"].items()):
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Predicted absent", "Predicted present"],
                                   y=["Actual absent", "Actual present"], colorscale=[[0, t["sunken"]], [1, t["primary"]]],
                                   showscale=False, text=[[f"TN {tn:,}", f"FP {fp:,}"], [f"FN {fn:,}", f"TP {tp:,}"]],
                                   texttemplate="%{text}", hoverinfo="skip"))
        with col:
            chart(fig, height=250, title=dict(text=MODEL_NAMES[name], font=dict(size=13)), yaxis_autorange="reversed")

# ---- thresholds -------------------------------------------------------------------------------
with tabs[3]:
    note("<b>Probability ≠ display band ≠ classification threshold.</b> The probability is the model's estimate. "
         "The Lower/Moderate/Higher bands are prototype presentation categories. A classification threshold "
         "turns probabilities into yes/no decisions; moving it trades false positives against false "
         "negatives. No threshold here is clinically optimal.")
    grid = pd.DataFrame(A["thresholds"]["grid"])
    th = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.01, key="threshold")
    row = grid.iloc[(grid["threshold"] - th).abs().idxmin()]
    kpis([("Recall (sensitivity)", f"{row['recall']:.1%}", f"FNR {row['fnr']:.1%}"),
          ("Specificity", f"{row['specificity']:.1%}", f"FPR {row['fpr']:.1%}"),
          ("Precision", f"{row['precision']:.1%}", "of predicted positives"),
          ("F1", f"{row['f1']:.3f}", f"{row['predicted_positive_rate']:.0%} flagged positive")])
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        fig = go.Figure()
        for key, label, color in (("recall", "Recall", t["higher"]), ("specificity", "Specificity", t["lower"]),
                                  ("precision", "Precision", "#6366F1"), ("f1", "F1", t["faint"])):
            fig.add_trace(go.Scatter(x=grid["threshold"], y=grid[key], mode="lines", name=label, line=dict(color=color)))
        fig.add_vline(x=th, line_color=t["text"], line_dash="dot")
        chart(fig, height=340, xaxis_title="Threshold", yaxis_tickformat=".0%")
    with b:
        fig = go.Figure(go.Heatmap(
            z=[[row["tn"], row["fp"]], [row["fn"], row["tp"]]], x=["Predicted absent", "Predicted present"],
            y=["Actual absent", "Actual present"], colorscale=[[0, t["sunken"]], [1, t["primary"]]], showscale=False,
            text=[[f"TN {row['tn']:,.0f}", f"FP {row['fp']:,.0f}"], [f"FN {row['fn']:,.0f}", f"TP {row['tp']:,.0f}"]],
            texttemplate="%{text}", hoverinfo="skip"))
        chart(fig, height=300, yaxis_autorange="reversed", title=dict(text=f"At threshold {th:.2f}", font=dict(size=13)))
    section("Threshold table (selected model, test set)")
    st.dataframe(pd.DataFrame(A["thresholds"]["table"])[
        ["threshold", "precision", "recall", "specificity", "f1", "fpr", "fnr", "tp", "fp", "tn", "fn"]],
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.3f")
                       for c in ("precision", "recall", "specificity", "f1", "fpr", "fnr")})

# ---- explainability ---------------------------------------------------------------------------
with tabs[4]:
    a, b = st.columns(2, gap="large")
    with a:
        section("Global SHAP importance (selected model)")
        g = A["shap_global"][::-1]
        fig = go.Figure(go.Bar(x=[r["share"] for r in g], y=[short(r["label"]) for r in g], orientation="h",
                               marker_color=t["primary"], hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        chart(fig, height=380, xaxis_tickformat=".0%", xaxis_title="Share of mean |SHAP| (all cleaned records)")
    with b:
        section("Permutation importance (test set)")
        p = A["permutation_importance"][::-1]
        fig = go.Figure(go.Bar(x=[r["mean"] for r in p], y=[r["label"] for r in p], orientation="h",
                               marker_color="#6366F1", error_x=dict(type="data", array=[r["std"] for r in p]),
                               hovertemplate="%{y}: ROC-AUC drop %{x:.4f}<extra></extra>"))
        chart(fig, height=380, xaxis_title="Drop in ROC-AUC when the input is shuffled")
    st.markdown(
        "**Two different questions.** SHAP importance: *how much does each feature move this model's score, on "
        "average?* Permutation importance: *how much worse does the model rank records if this input is "
        "scrambled?* They agree on the top feature (systolic blood pressure) but can differ elsewhere: "
        "permutation works on the 11 raw inputs (height and weight separately), SHAP on the 10 model features "
        "(BMI), and a feature can move scores without adding much ranking information.")
    with st.expander("Direction of each feature's effect (global SHAP)"):
        for r in A["shap_global"]:
            detail = r.get("direction") or ", ".join(f"{k}: {v:+.2f}" for k, v in r["levels"].items())
            st.markdown(f"- **{r['label']}** — {detail}")
        st.caption("Categorical values: mean score contribution of each level. Smoking and alcohol 'Yes' have "
                   "negative contributions because of a confounded pattern in this dataset.")

    section("Feature response: partial dependence and ICE")
    fr = A["feature_response"]
    feature = st.segmented_control("Feature", list(fr), default="ap_hi", format_func=lambda f: fr[f]["label"],
                                   key="pd_feature", label_visibility="collapsed") or "ap_hi"
    resp = fr[feature]
    fig = go.Figure()
    for i, line in enumerate(resp["models"]["logistic_regression"]["ice"]):
        fig.add_trace(go.Scatter(x=resp["grid"], y=line, mode="lines", line=dict(color=t["primary"], width=0.7),
                                 opacity=0.25, showlegend=i == 0, name="Individual records (ICE)",
                                 hoverinfo="skip"))
    for name, dash in (("logistic_regression", "solid"), ("xgboost", "dash")):
        fig.add_trace(go.Scatter(x=resp["grid"], y=resp["models"][name]["average"], mode="lines",
                                 line=dict(color=MODEL_COLORS[name], width=3, dash=dash),
                                 name=f"Average — {MODEL_NAMES[name]}"))
    chart(fig, height=380, yaxis_tickformat=".0%", xaxis_title=resp["label"], yaxis_title="Model estimate")
    st.caption("2,000 test records; each line varies one input over its 2nd–98th percentile with the others fixed. "
               "Grid points that would create an impossible record (systolic ≤ diastolic, implausible BMI) are "
               "skipped. Systolic and diastolic pressure are correlated (r = 0.73), so varying one alone partly "
               "creates unusual combinations; read these as model behaviour, not physiology.")

    section("Feature interactions (XGBoost)")
    it = A["interactions"]
    pairs = it["pairs"][::-1]
    fig = go.Figure(go.Bar(x=[p["strength"] for p in pairs], y=[f"{short(p['a'])} × {short(p['b'])}" for p in pairs],
                           orientation="h", marker_color=MODEL_COLORS["xgboost"]))
    chart(fig, height=340, xaxis_title="Mean |SHAP interaction| (score units)")
    st.caption(f"SHAP interaction values on {it['rows']:,} test records. Interactions account for about "
               f"{it['interaction_share']:.0%} of XGBoost's total attribution. The selected logistic regression "
               "has no interaction terms by construction. Interactions describe the model, not biology.")

# ---- subgroups --------------------------------------------------------------------------------
with tabs[5]:
    st.markdown("**Subgroup Performance Analysis.** Does the selected model behave differently across groups? "
                "This identifies differences; it does not establish fairness or clinical validity.")
    for key, title in (("gender", "Gender"), ("age_group", "Age group (years)")):
        rows = pd.DataFrame(A["subgroups"][key])
        section(title)
        a, b = st.columns([1.9, 1], gap="large")
        with a:
            show = rows.rename(columns={"group": "Group", "n": "n", "observed_rate": "Observed rate",
                                        "mean_predicted": "Mean predicted", "roc_auc": "ROC-AUC",
                                        "precision": "Precision", "recall": "Recall", "f1": "F1", "brier": "Brier"})
            st.dataframe(show.drop(columns=["positives", "reliable"]), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                        ("Observed rate", "Mean predicted", "ROC-AUC", "Precision", "Recall", "F1", "Brier")})
        with b:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=rows["group"], y=rows["observed_rate"], name="Observed rate", marker_color=t["faint"]))
            fig.add_trace(go.Bar(x=rows["group"], y=rows["mean_predicted"], name="Mean predicted", marker_color=t["primary"]))
            chart(fig, height=250, yaxis_tickformat=".0%", barmode="group")
    note("<b>Finding.</b> ROC-AUC falls with age, from about 0.82 (40–49) to 0.69 (60–65). Part of this is "
         "expected, because age itself is a strong signal and varies little inside a narrow band, but it means "
         "the model separates outcomes less well for older people in this data. Calibration-in-the-large "
         "(observed vs mean predicted) stays close in every group; the youngest group (n≈370) is slightly "
         "over-estimated. Gender groups perform similarly.")

# ---- uncertainty ------------------------------------------------------------------------------
with tabs[6]:
    bs = A["bootstrap"]
    st.markdown(f"**Question:** how much could the test metrics move with a different sample of the same size? "
                f"Percentile bootstrap, {bs['n_resamples']:,} resamples of the {A['test_rows']:,} test rows "
                f"(seed {bs['seed']}), {bs['level']:.0%} intervals. All models share the same resamples, so their "
                "differences get their own intervals. This is evaluation uncertainty, not clinical uncertainty.")
    labels = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1 (0.50)", "brier": "Brier"}
    cols = st.columns(4)
    for col, (metric, label) in zip(cols, labels.items()):
        fig = go.Figure()
        for name, m in bs["models"].items():
            ci = m[metric]
            fig.add_trace(go.Scatter(x=[ci["estimate"]], y=[MODEL_NAMES[name]], mode="markers",
                                     marker=dict(color=MODEL_COLORS[name], size=10), showlegend=False,
                                     error_x=dict(type="data", symmetric=False, array=[ci["upper"] - ci["estimate"]],
                                                  arrayminus=[ci["estimate"] - ci["lower"]]),
                                     hovertemplate=f"{ci['estimate']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]<extra></extra>"))
        with col:
            chart(fig, height=230, title=dict(text=label, font=dict(size=13)), yaxis_autorange="reversed")
    section("Difference from Logistic Regression (paired bootstrap)")
    rows = [{"Model": MODEL_NAMES[n], "Metric": labels[m], "Difference": ci["estimate"], "Lower": ci["lower"],
             "Upper": ci["upper"], "Interval excludes 0": "yes" if ci["lower"] > 0 or ci["upper"] < 0 else "no"}
            for n, d in bs["differences"].items() for m, ci in d.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%+.4f") for c in ("Difference", "Lower", "Upper")})
    st.caption("For Brier, a negative difference means the other model's probabilities are better.")
footer()
