"""MODEL: selection, calibration, discrimination, thresholds, explainability, subgroups, uncertainty, robustness.
Everything is read from the precomputed artifacts (metrics, analysis, reliability, shift_analysis)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.preprocessing import FEATURE_LABELS
from ui.core import (MODEL_COLORS, MODEL_NAMES, TRAIN_COMMAND, analysis, artifact, chart, footer, kpis, metrics_report,
                     note, page_header, section, tokens)

A, R = analysis(), metrics_report()
REL, SHIFT = artifact("reliability.json"), artifact("shift_analysis.json")
t = tokens()
FINAL, VARIANT = R["selection"]["model"], R["selection"]["variant"]
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
                 "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "log_loss": "Log loss", "brier": "Brier"}
NUM = {m: st.column_config.NumberColumn(format="%.3f") for m in METRIC_LABELS.values()}
VARIANT_DASH = {"raw": "dot", "sigmoid": "dash", "isotonic": "solid"}


def short(label: str) -> str:
    return label.replace(" (from height and weight)", "")


def name(model: str) -> str:
    return MODEL_NAMES[model] + (" (final)" if model == FINAL else "")


def ci(value, interval) -> str:
    return f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"


page_header("Model", "Model analytics",
            f"How the three candidate models compare, how trustworthy the probabilities are, and where the selected "
            f"model behaves differently. Test set: {A['test_rows']:,} held-out records.")
tabs = st.tabs(["Overview", "Calibration", "ROC & PR", "Thresholds", "Explainability", "Subgroups", "Uncertainty",
                "Robustness"])

# ---- overview ---------------------------------------------------------------------------------
with tabs[0]:
    cv = R["cv"]
    k = cv["n_splits"] * cv["n_repeats"]
    sel_cv = R["comparison"][FINAL][VARIANT]["cv"]
    kpis([("Selected model", MODEL_NAMES[FINAL], f"{VARIANT} probabilities"),
          ("CV ROC-AUC", f"{sel_cv['roc_auc']['mean']:.3f}", f"± {sel_cv['roc_auc']['std']:.3f} over {k} splits"),
          ("Test ROC-AUC", f"{A['models'][FINAL]['test']['roc_auc']:.3f}", "held-out, evaluated once"),
          ("Test Brier", f"{A['models'][FINAL]['test']['brier']:.3f}", "lower is better")])
    section(f"Comparison: {cv['n_splits']}-fold CV repeated {cv['n_repeats']} times ({k} splits), training split")
    rows = []
    for m, comp in R["comparison"].items():
        c = comp[comp["selected"]]["cv"]
        for stat in ("mean", "std", "min", "max"):
            rows.append({"Model": name(m), "Variant": comp["selected"], "Statistic": f"CV {stat}",
                         **{METRIC_LABELS[x]: c[x][stat] for x in METRIC_LABELS}})
        rows.append({"Model": name(m), "Variant": comp["selected"], "Statistic": "Test",
                     **{METRIC_LABELS[x]: comp["test"][x] for x in METRIC_LABELS}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=NUM)
    st.caption("Accuracy, precision, recall and F1 use the 0.50 threshold. ROC-AUC and PR-AUC measure ranking; "
               "log loss and Brier measure probability quality (lower is better). The test row is computed once, "
               "after selection, and was not used to choose anything.")

    section("Stability across the repeated CV splits")
    fig = make_subplots(rows=1, cols=4, subplot_titles=["ROC-AUC (higher better)", "PR-AUC (higher better)",
                                                        "Brier (lower better)", "F1 at 0.50 (higher better)"])
    for col, metric in enumerate(("roc_auc", "pr_auc", "brier", "f1"), 1):
        for m, comp in R["comparison"].items():
            fig.add_trace(go.Box(y=comp[comp["selected"]]["cv"][metric]["folds"], name=MODEL_NAMES[m],
                                 marker_color=MODEL_COLORS[m], boxpoints="all", jitter=0.4, pointpos=0,
                                 showlegend=col == 1, legendgroup=m), row=1, col=col)
    fig.update_xaxes(showticklabels=False)
    chart(fig, height=330, yaxis_title="Score on one validation split")
    st.caption(f"Each point is one of the {k} validation splits (same splits for every model). Narrow boxes mean "
               "the result depends little on which rows happened to be held out.")

    section("Selection protocol")
    sel = R["selection"]
    st.markdown(
        f"Declared before the repeated-CV comparison was run. No tunable margins: each comparison is a **{sel['test']}** "
        f"on the same {k} splits, at α = {sel['alpha']} (the conventional level). The test set is never used.\n"
        f"1. **Calibration per model:** keep raw probabilities unless a calibrator lowers CV Brier significantly.\n"
        f"2. **Explanation-quality gate:** an exact, additive SHAP explanation must exist, mapping monotonically "
        f"to the displayed probability.\n"
        f"3. **Model:** start from the most interpretable model ({', '.join(MODEL_NAMES[m] for m in sel['interpretability_order'])}) "
        "and move on only if the next model is significantly better on **both** ROC-AUC and Brier.")
    st.dataframe(pd.DataFrame([{"Comparison": c.replace("_vs_", " vs ").replace("_", " "), "Metric": METRIC_LABELS[m],
                                "Mean difference": v["mean_difference"], "t": v["t"], "p": v["p_value"]}
                               for c, d in sel["tests"].items() for m, v in d.items()]),
                 hide_index=True, width="stretch",
                 column_config={"Mean difference": st.column_config.NumberColumn(format="%+.4f"),
                                "t": st.column_config.NumberColumn(format="%.2f"),
                                "p": st.column_config.NumberColumn(format="%.2g")})
    for step in sel["steps"]:
        st.markdown(f"- `{step}`")
    for m, comp in R["comparison"].items():
        st.markdown(f"- **{MODEL_NAMES[m]}:** {comp['calibration_reason']}")
    note("<b>Methodological correction.</b> The previous version chose Logistic Regression with margins "
         "(0.02 ROC-AUC, 0.001 Brier) that were set after the first comparison was known. Those margins are no "
         "longer used. Under the pre-declared protocol the evidence selects the model above; the differences are "
         "small in absolute terms (about 0.01 ROC-AUC) but consistent across splits. See <b>Robustness</b> for "
         "the costs of this choice (less smooth responses, less stable explanations).")

# ---- calibration ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("**Question:** how well do predicted probabilities agree with observed outcome frequencies in this "
                "dataset? Points on the diagonal mean close agreement. Out-of-fold predictions of the training split: "
                "every calibrator was fit inside the training folds, so no point was predicted by a model (or "
                "calibrator) that saw it.")
    if not REL:
        st.info(f"Run `{TRAIN_COMMAND}` to generate the reliability artifacts.")
    else:
        cal = REL["calibration"]
        model = st.segmented_control("Model", list(cal["models"]), default=FINAL, format_func=name,
                                     key="cal_model", label_visibility="collapsed") or FINAL
        fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28], shared_xaxes=True, vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect agreement",
                                 line=dict(color=t["faint"], dash="dot")), row=1, col=1)
        for v, d in cal["models"][model]["variants"].items():
            b = pd.DataFrame(d["bins"])
            label = f"{v}{' (selected)' if v == cal['models'][model]['selected'] else ''}"
            fig.add_trace(go.Scatter(
                x=b["mean_predicted"], y=b["observed_rate"], mode="lines+markers", name=label,
                line=dict(color=MODEL_COLORS[model], dash=VARIANT_DASH[v]),
                marker=dict(size=9, symbol=["circle-open" if s else "circle" for s in b["sparse"]]),
                customdata=b["count"], hovertemplate="predicted %{x:.3f}, observed %{y:.3f} (n=%{customdata:,})<extra></extra>"),
                row=1, col=1)
            fig.add_trace(go.Bar(x=(b["bin_low"] + b["bin_high"]) / 2, y=b["count"], name=f"{v} rows per bin",
                                 marker_color=MODEL_COLORS[model], opacity={"raw": .3, "sigmoid": .55, "isotonic": .85}[v],
                                 showlegend=False, hovertemplate=f"{v}: %{{y:,}} rows<extra></extra>"), row=2, col=1)
        fig.update_xaxes(range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text="Mean predicted probability (bin)", range=[0, 1], row=2, col=1)
        fig.update_yaxes(title_text="Observed outcome frequency", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Rows", row=2, col=1)
        chart(fig, height=560, barmode="group")
        st.caption(f"{cal['bins']} bins on repeat 1 of the out-of-fold predictions (each training row once). Hollow "
                   f"markers: fewer than {cal['sparse_bin_rows']} rows, too few to interpret. Lower panel: rows per bin.")
        rows = [{"Variant": v + (" ✓" if v == cal["models"][model]["selected"] else ""),
                 **{lbl: f"{d['stats'][k]['mean']:.4f} ± {d['stats'][k]['std']:.4f}" for k, lbl in
                    (("brier", "Brier"), ("log_loss", "Log loss"), ("ece", "ECE"), ("slope", "Slope"),
                     ("intercept", "Intercept"))}} for v, d in cal["models"][model]["variants"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Mean ± std over the 5 CV repeats. ECE: count-weighted mean |observed − predicted| over the 10 bins. "
                   "Slope and intercept: logistic recalibration of the outcome on logit(p); slope 1 and intercept 0 "
                   "mean no linear miscalibration. They cannot see S-shaped miscalibration, which ECE and the curve can.")
        if model == "logistic_regression":
            note("<b>Finding.</b> Raw Logistic Regression has slope ≈ 1 and intercept ≈ 0, yet ECE 0.034: its "
                 "miscalibration is S-shaped (over-estimates around 25–35% and at the top, under-estimates around "
                 "55–75%), which a logistic recalibration cannot represent. Sigmoid calibration therefore changes "
                 "nothing; isotonic calibration lowers ECE to about 0.002 and Brier in every repeat.")
        else:
            note(f"<b>Finding.</b> {MODEL_NAMES[model]}'s raw probabilities already agree closely with observed "
                 "frequencies out-of-fold; neither calibrator lowered CV Brier significantly, so the protocol kept "
                 "raw probabilities. Agreement here means agreement with this dataset's labels, not clinical accuracy.")
        section("Held-out test set (final evaluation, reported only)")
        st.dataframe(pd.DataFrame([{"Model / variant": k.replace("__", " · ").replace("_", " "), **{
            lbl: v[key] for key, lbl in (("brier", "Brier"), ("log_loss", "Log loss"), ("ece", "ECE"),
                                         ("slope", "Slope"), ("intercept", "Intercept"))}}
            for k, v in cal["test"].items()]), hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.4f")
                           for c in ("Brier", "Log loss", "ECE", "Slope", "Intercept")})

# ---- ROC & PR ---------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("**Question:** how well does each model rank records with the disease label above those without, "
                "at every possible threshold?")
    a, b = st.columns(2, gap="large")
    with a:
        fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=t["faint"], dash="dot"),
                                   name="Chance"))
        for m, d in A["models"].items():
            fig.add_trace(go.Scatter(x=d["roc"]["x"], y=d["roc"]["y"], mode="lines", line=dict(color=MODEL_COLORS[m]),
                                     name=f"{MODEL_NAMES[m]} ({d['test']['roc_auc']:.3f})"))
        chart(fig, height=380, xaxis_title="False positive rate", yaxis_title="True positive rate (recall)",
              title=dict(text="ROC curve (test)", font=dict(size=14)))
    with b:
        prevalence = A["data_quality"]["target_clean"]["1"] / A["data_quality"]["clean_rows"]
        fig = go.Figure(go.Scatter(x=[0, 1], y=[prevalence] * 2, mode="lines", line=dict(color=t["faint"], dash="dot"),
                                   name="Chance (prevalence)"))
        for m, d in A["models"].items():
            fig.add_trace(go.Scatter(x=d["pr"]["x"], y=d["pr"]["y"], mode="lines", line=dict(color=MODEL_COLORS[m]),
                                     name=f"{MODEL_NAMES[m]} ({d['test']['pr_auc']:.3f})"))
        chart(fig, height=380, xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0.4, 1],
              title=dict(text="Precision–recall curve (test)", font=dict(size=14)))
    section("Confusion matrices at the 0.50 threshold (test)")
    cols = st.columns(3)
    for col, (m, d) in zip(cols, A["models"].items()):
        (tn, fp), (fn, tp) = d["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Predicted absent", "Predicted present"],
                                   y=["Actual absent", "Actual present"], colorscale=[[0, t["sunken"]], [1, t["primary"]]],
                                   showscale=False, text=[[f"TN {tn:,}", f"FP {fp:,}"], [f"FN {fn:,}", f"TP {tp:,}"]],
                                   texttemplate="%{text}", hoverinfo="skip"))
        with col:
            chart(fig, height=250, title=dict(text=name(m), font=dict(size=13)), yaxis_autorange="reversed")


# ---- thresholds -------------------------------------------------------------------------------
def confusion_fig(r, title):
    fig = go.Figure(go.Heatmap(
        z=[[r["tn"], r["fp"]], [r["fn"], r["tp"]]], x=["Predicted absent", "Predicted present"],
        y=["Actual absent", "Actual present"], colorscale=[[0, t["sunken"]], [1, t["primary"]]], showscale=False,
        text=[[f"TN {r['tn']:,.0f}", f"FP {r['fp']:,.0f}"], [f"FN {r['fn']:,.0f}", f"TP {r['tp']:,.0f}"]],
        texttemplate="%{text}", hoverinfo="skip"))
    return fig, dict(yaxis_autorange="reversed", title=dict(text=title, font=dict(size=13)))


with tabs[3]:
    note("<b>Model probability ≠ display band ≠ classification threshold.</b> The probability is the model's "
         "estimate. The Lower/Moderate/Higher bands (30% / 60%) are prototype presentation categories, not medical "
         "thresholds. A classification threshold turns probabilities into yes/no decisions; moving it trades false "
         "positives against false negatives. Everything here is exploratory: no threshold is selected, and none is "
         "clinically optimal.")
    sources = {"test": "Held-out test set (descriptive)"}
    if REL:
        sources = {"oof": "Out-of-fold, training split (for exploration)", **sources}
    source = st.segmented_control("Data", list(sources), default=list(sources)[0], format_func=sources.get,
                                  key="threshold_source", label_visibility="collapsed") or list(sources)[0]
    data = REL["thresholds_oof"] if source == "oof" else A["thresholds"]
    grid = pd.DataFrame(data["grid"])
    th = st.slider("Classification threshold (exploratory)", 0.05, 0.95, 0.50, 0.01, key="threshold")
    row = grid.iloc[(grid["threshold"] - th).abs().idxmin()]
    kpis([("Recall (sensitivity)", f"{row['recall']:.1%}", f"FNR {row['fnr']:.1%}"),
          ("Specificity", f"{row['specificity']:.1%}", f"FPR {row['fpr']:.1%}"),
          ("Precision", f"{row['precision']:.1%}", "of predicted positives"),
          ("F1", f"{row['f1']:.3f}", f"accuracy {row['accuracy']:.1%}")])
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        fig = go.Figure()
        for key, label, color in (("recall", "Recall (sensitivity)", t["higher"]), ("specificity", "Specificity", t["lower"]),
                                  ("precision", "Precision", "#6366F1"), ("f1", "F1", t["faint"]),
                                  ("accuracy", "Accuracy", t["text"])):
            fig.add_trace(go.Scatter(x=grid["threshold"], y=grid[key], mode="lines", name=label,
                                     line=dict(color=color, dash="dot" if key == "accuracy" else "solid")))
        fig.add_vline(x=th, line_color=t["text"], line_dash="dot")
        chart(fig, height=340, xaxis_title="Classification threshold", yaxis_title="Rate", yaxis_tickformat=".0%")
    with b:
        fig, layout = confusion_fig(row, f"At threshold {th:.2f}")
        chart(fig, height=300, **layout)

    section("Decision trade-off")
    a, b = st.columns(2, gap="large")
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid["threshold"], y=grid["fp"], mode="lines", name="False positives",
                                 line=dict(color=t["higher"])))
        fig.add_trace(go.Scatter(x=grid["threshold"], y=grid["fn"], mode="lines", name="False negatives",
                                 line=dict(color=t["lower"])))
        fig.add_vline(x=th, line_color=t["text"], line_dash="dot")
        chart(fig, height=320, xaxis_title="Classification threshold", yaxis_title="Records",
              title=dict(text="Errors of each kind", font=dict(size=13)))
    with b:
        dc = pd.DataFrame(data["decision_curve"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dc["threshold"], y=dc["model"], mode="lines", name=MODEL_NAMES[FINAL],
                                 line=dict(color=MODEL_COLORS[FINAL])))
        fig.add_trace(go.Scatter(x=dc["threshold"], y=dc["flag_all"], mode="lines", name="Flag everyone",
                                 line=dict(color=t["faint"], dash="dash")))
        fig.add_trace(go.Scatter(x=dc["threshold"], y=[0] * len(dc), mode="lines", name="Flag no one",
                                 line=dict(color=t["faint"], dash="dot")))
        chart(fig, height=320, xaxis_title="Threshold probability t", yaxis_title="Net benefit",
              yaxis_range=[-0.05, max(dc["model"].max(), dc["flag_all"].max()) + 0.05],
              title=dict(text="Net benefit (exploratory, this dataset only)", font=dict(size=13)))
    st.caption("Left: lowering the threshold converts false negatives into false positives. Right: net benefit = "
               "TP/n − FP/n × t/(1 − t), a decision-curve summary that weighs a false positive t/(1 − t) times "
               "a true positive. It depends on this dataset's prevalence (about 50%) and says nothing about clinical "
               "utility.")
    section("Confusion matrices at selected thresholds")
    cols = st.columns(3)
    for col, thr in zip(cols, (0.3, 0.5, 0.7)):
        r = grid.iloc[(grid["threshold"] - thr).abs().idxmin()]
        fig, layout = confusion_fig(r, f"Threshold {thr:.2f}")
        with col:
            chart(fig, height=250, **layout)
    section("Threshold grid")
    st.dataframe(grid[["threshold", "recall", "specificity", "precision", "f1", "accuracy", "fpr", "fnr",
                       "tp", "fp", "tn", "fn"]], hide_index=True, width="stretch", height=300,
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ("recall", "specificity", "precision", "f1", "accuracy", "fpr", "fnr")})

# ---- explainability ---------------------------------------------------------------------------
with tabs[4]:
    a, b = st.columns(2, gap="large")
    with a:
        section("Global SHAP importance (selected model)")
        g = A["shap_global"][::-1]
        fig = go.Figure(go.Bar(x=[r["share"] for r in g], y=[short(r["label"]) for r in g], orientation="h",
                               marker_color=t["primary"], hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        chart(fig, height=380, xaxis_tickformat=".0%",
              xaxis_title=f"Share of mean |SHAP| ({A['shap_rows']:,} cleaned records)")
    with b:
        section("Permutation importance (test set)")
        p = A["permutation_importance"][::-1]
        fig = go.Figure(go.Bar(x=[r["mean"] for r in p], y=[r["label"] for r in p], orientation="h",
                               marker_color="#6366F1", error_x=dict(type="data", array=[r["std"] for r in p]),
                               hovertemplate="%{y}: ROC-AUC drop %{x:.4f}<extra></extra>"))
        chart(fig, height=380, xaxis_title="Drop in test ROC-AUC when the input is shuffled")
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
        st.caption("Numeric features: sign of the overall value-vs-contribution slope (a tree model's response can "
                   "still be non-monotone, see Robustness). Categorical values: mean score contribution of each "
                   "level. Smoking and alcohol 'Yes' get negative contributions because of a confounded pattern in "
                   "this dataset.")

    section("Feature response: partial dependence and ICE")
    fr = A["feature_response"]
    feature = st.segmented_control("Feature", list(fr), default="ap_hi", format_func=lambda f: fr[f]["label"],
                                   key="pd_feature", label_visibility="collapsed") or "ap_hi"
    resp = fr[feature]
    fig = go.Figure()
    for i, line in enumerate(resp["models"][FINAL]["ice"]):
        fig.add_trace(go.Scatter(x=resp["grid"], y=line, mode="lines", line=dict(color=MODEL_COLORS[FINAL], width=0.7),
                                 opacity=0.25, showlegend=i == 0, name=f"Individual records (ICE, {MODEL_NAMES[FINAL]})",
                                 hoverinfo="skip"))
    for m, d in resp["models"].items():
        fig.add_trace(go.Scatter(x=resp["grid"], y=d["average"], mode="lines",
                                 line=dict(color=MODEL_COLORS[m], width=3, dash="solid" if m == FINAL else "dash"),
                                 name=f"Average — {name(m)}"))
    chart(fig, height=380, yaxis_tickformat=".0%", xaxis_title=resp["label"], yaxis_title="Model-estimated probability")
    st.caption("2,000 test records; each line varies one input over its 2nd–98th percentile with the others fixed. "
               "Grid points that would create an impossible record (systolic ≤ diastolic, implausible BMI) are "
               "skipped. Systolic and diastolic pressure are correlated (r = 0.73), so varying one alone partly "
               "creates unusual combinations; read these as model behaviour, not physiology.")

    section("Feature interactions (XGBoost)")
    it = A["interactions"]
    pairs = it["pairs"][::-1]
    fig = go.Figure(go.Bar(x=[p["strength"] for p in pairs], y=[f"{short(p['a'])} × {short(p['b'])}" for p in pairs],
                           orientation="h", marker_color=MODEL_COLORS["xgboost"]))
    chart(fig, height=340, xaxis_title="Mean |SHAP interaction| (log-odds)")
    st.caption(f"SHAP interaction values on {it['rows']:,} test records. Interactions account for about "
               f"{it['interaction_share']:.0%} of XGBoost's total attribution, so one input's contribution depends "
               "partly on the others. Logistic Regression has no interaction terms by construction. Interactions "
               "describe the model, not biology.")

# ---- subgroups --------------------------------------------------------------------------------
with tabs[5]:
    st.markdown(f"**Subgroup Performance Analysis** ({name(FINAL)}, test set). Does the selected model behave "
                "differently across groups? This identifies differences; it is not a fairness certification and does "
                "not establish clinical validity. Nothing was tuned on these results.")
    for key, title in (("gender", "Gender"), ("age_group", "Age group (years)")):
        rows = A["subgroups"][key]
        section(title)
        a, b = st.columns([1.9, 1], gap="large")
        with a:
            table = []
            for r in rows:
                base = {"Group": r["group"], "N": r["n"], "Prevalence": f"{r['prevalence']:.3f}",
                        "Mean predicted": f"{r['mean_predicted']:.3f}"}
                if r["reliable"]:
                    base.update({"ROC-AUC [95% CI]": ci(r["roc_auc"], r["ci"]["roc_auc"]),
                                 "PR-AUC [95% CI]": ci(r["pr_auc"], r["ci"]["pr_auc"]),
                                 "Recall [95% CI]": ci(r["recall"], r["ci"]["recall"]),
                                 "Precision": f"{r['precision']:.3f}", "F1": f"{r['f1']:.3f}",
                                 "Brier [95% CI]": ci(r["brier"], r["ci"]["brier"])})
                else:
                    base["ROC-AUC [95% CI]"] = r["unavailable"]
                table.append(base)
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        with b:
            df = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["group"], y=df["observed_rate"], name="Observed rate", marker_color=t["faint"]))
            fig.add_trace(go.Bar(x=df["group"], y=df["mean_predicted"], name="Mean predicted", marker_color=t["primary"]))
            chart(fig, height=250, yaxis_tickformat=".0%", barmode="group", yaxis_title="Rate")
    st.caption("Intervals: percentile bootstrap within each group (500 resamples, seed 42). Recall, precision and F1 "
               "use the 0.50 threshold. A group with fewer than 30 records of either outcome would show "
               "'Metric unavailable for this subgroup' instead of numbers.")
    ages = {r["group"]: r for r in A["subgroups"]["age_group"] if r["reliable"]}
    if ages:
        trend = ", ".join(f"{g}: {r['roc_auc']:.2f}" for g, r in ages.items())
        note(f"<b>Finding.</b> ROC-AUC falls with age ({trend}). Part of this is expected, because age itself is a "
             "strong signal and varies little inside a narrow band, but the model separates outcomes less well for "
             "older people in this data. Mean predicted stays close to the observed rate in every group. Gender "
             "groups perform similarly.")

# ---- uncertainty ------------------------------------------------------------------------------
with tabs[6]:
    bs = A["bootstrap"]
    st.markdown(f"**Question:** how much could the test metrics move with a different sample of the same size? "
                f"Percentile bootstrap, {bs['n_resamples']:,} resamples of the {A['test_rows']:,} test rows "
                f"(seed {bs['seed']}), {bs['level']:.0%} intervals. All models share the same resamples, so their "
                "differences get their own intervals. This is uncertainty around evaluation metrics on the available "
                "sample, not clinical uncertainty and not confidence that the model is correct.")
    labels = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1 (0.50)", "brier": "Brier", "ece": "ECE"}
    cols = st.columns(5)
    for col, (metric, label) in zip(cols, labels.items()):
        fig = go.Figure()
        for m, d in bs["models"].items():
            c = d[metric]
            fig.add_trace(go.Scatter(x=[c["estimate"]], y=[MODEL_NAMES[m]], mode="markers",
                                     marker=dict(color=MODEL_COLORS[m], size=10), showlegend=False,
                                     error_x=dict(type="data", symmetric=False, array=[max(c["upper"] - c["estimate"], 0)],
                                                  arrayminus=[max(c["estimate"] - c["lower"], 0)]),
                                     hovertemplate=f"{c['estimate']:.4f} [{c['lower']:.4f}, {c['upper']:.4f}]<extra></extra>"))
        with col:
            chart(fig, height=230, title=dict(text=label, font=dict(size=13)), yaxis_autorange="reversed",
                  xaxis_title=label)
    st.caption("ECE is a binned statistic that is biased upward under resampling (sampling noise adds to every bin's "
               "gap), so its bootstrap interval can sit above the point estimate. Read it as variability, not as a "
               "bracket around the true value.")
    section(f"Difference from the final model ({MODEL_NAMES[FINAL]}, paired bootstrap)")
    rows = [{"Model": MODEL_NAMES[m], "Metric": labels[k], "Difference": c["estimate"], "Lower": c["lower"],
             "Upper": c["upper"], "Interval excludes 0": "yes" if c["lower"] > 0 or c["upper"] < 0 else "no"}
            for m, d in bs["differences"].items() for k, c in d.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%+.4f") for c in ("Difference", "Lower", "Upper")})
    st.caption("Other model minus final model. For Brier and ECE, a positive difference means the other model's "
               "probabilities agree less well with observed outcomes.")

# ---- robustness -------------------------------------------------------------------------------
with tabs[7]:
    if not REL or not SHIFT:
        st.info(f"Run `{TRAIN_COMMAND}` to generate the reliability artifacts.")
        st.stop()
    models = list(MODEL_NAMES)

    section("Input conformity (novelty detection)")
    nv = REL["novelty"]
    st.dataframe(pd.DataFrame([{"Method": m.replace("_", " ").capitalize() + (" ✓ chosen" if m == nv["chosen"] else ""),
                                "Flag rate, reference rows": d["flag_rate_reference"],
                                "Flag rate, test rows": d["flag_rate_test"],
                                "Flag rate, atypical combinations": d["flag_rate_atypical"],
                                "AUC test vs atypical": d["auc_test_vs_atypical"]} for m, d in nv["methods"].items()]),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ("Flag rate, reference rows", "Flag rate, test rows", "Flag rate, atypical combinations",
                                 "AUC test vs atypical")})
    st.caption(f"Both detectors were fit on {nv['fit_rows']:,} training rows; the threshold is the 99th percentile of "
               f"{nv['reference_rows']:,} held-back training rows. 'Atypical combinations' ({nv['atypical_rows']:,} "
               "records) are test values recombined at random, kept only if valid. Pre-declared rule: higher AUC "
               "wins, a tie goes to the simpler method. Neither separates atypical combinations strongly, so the "
               "check mainly catches extreme values and is a statistical caution, not a medical abnormality detector.")

    section("Prediction stability under small input changes")
    pr = REL["perturbation"]["prediction"]
    fig = go.Figure()
    for m in models:
        ds = [k for k in pr[m] if k.endswith("%")]
        fig.add_trace(go.Bar(x=[f"±{d}" for d in ds], y=[pr[m][d]["median"] for d in ds], name=f"{MODEL_NAMES[m]} median",
                             marker_color=MODEL_COLORS[m],
                             error_y=dict(type="data", symmetric=False, array=[pr[m][d]["p95"] - pr[m][d]["median"] for d in ds],
                                          arrayminus=[0] * len(ds))))
    chart(fig, height=320, barmode="group", xaxis_title="Size of input change (one input alone, or all five jointly)",
          yaxis_title="|Change in probability| (pp)")
    st.caption(f"{REL['perturbation']['profiles']} baseline profiles (3 demo inputs + random test records), "
               f"{REL['perturbation']['rows']:,} valid perturbed versions; age, height, weight, systolic and diastolic "
               "changed by ±1/2/5% one at a time and jointly. Bars: median; whiskers: 95th percentile. Categorical "
               "inputs have no small change and are excluded.")
    st.dataframe(pd.DataFrame([{"Model": name(m), **{f"{FEATURE_LABELS[f]} ±5%": f"{v['median']:.1f} / {v['p95']:.1f} / {v['max']:.1f}"
                                                     for f, v in pr[m]["by_feature_5pct"].items()}} for m in models]),
                 hide_index=True, width="stretch")
    st.caption("Median / 95th percentile / maximum change in percentage points. Tree models respond in steps at "
               "their split points, and recorded blood pressures cluster at round values (120/80), so a 1–2% change "
               "that crosses a split can move the estimate by many points.")

    section("Explanation stability")
    es = REL["explanation_stability"]
    rows = []
    for m in models:
        for d, v in es["models"][m].items():
            rows.append({"Model": name(m), "Change": f"±{d}", "Spearman ρ (mean)": v["spearman"]["mean"],
                         f"Top-{es['top_k']} overlap (mean)": v["top_k_overlap"]["mean"],
                         f"Top-{es['top_k']} unchanged": v["share_full_top_k_overlap"],
                         "Same top feature": v["share_same_top1"], "L1 distance (median)": v["l1"]["median"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ("Spearman ρ (mean)", f"Top-{es['top_k']} overlap (mean)", "L1 distance (median)")} |
                 {c: st.column_config.NumberColumn(format="percent") for c in (f"Top-{es['top_k']} unchanged", "Same top feature")})
    st.caption("SHAP recomputed for every perturbed version and compared with its baseline: Spearman rank correlation "
               "of |SHAP|, share of the top-5 features kept, and Σ|φ−φ'| / Σ|φ|. Descriptive; there is no universal "
               "'stable enough' threshold.")
    st.dataframe(pd.DataFrame([{"Model": name(m), "Rows": v["rows"], "Max |base + ΣSHAP − score|": v["max_score_error"],
                                "Max |probability from SHAP − pipeline probability|": v["max_probability_error"]}
                               for m, v in REL["reconciliation"].items()]), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.1e") for c in
                                ("Max |base + ΣSHAP − score|", "Max |probability from SHAP − pipeline probability|")})
    st.caption("SHAP reconciliation on 1,000 test records; documented tolerance 1e-5 (XGBoost computes in float32).")

    section("Controlled sensitivity (monotonicity check)")
    mono = REL["monotonicity"]
    f = st.segmented_control("Input", list(mono["features"]), default="ap_hi", format_func=FEATURE_LABELS.get,
                             key="mono_feature", label_visibility="collapsed") or "ap_hi"
    a, b = st.columns([1.5, 1], gap="large")
    with a:
        fig = go.Figure()
        for m in models:
            for i, c in enumerate(mono["features"][f]["models"][m]["demo_curves"]):
                fig.add_trace(go.Scatter(x=c["x"], y=c["p"], mode="lines", name=f"{MODEL_NAMES[m]} · {c['profile']}",
                                         line=dict(color=MODEL_COLORS[m], dash=["solid", "dash", "dot"][i], width=2)))
        chart(fig, height=380, xaxis_title=f"{FEATURE_LABELS[f]} (all other inputs fixed)",
              yaxis_title="Model-estimated probability", yaxis_tickformat=".0%")
    with b:
        st.dataframe(pd.DataFrame([{"Model": name(m), **d["counts"], "Largest drop (pp)": d["largest_single_step_decrease_pp"]}
                                   for m, d in ((m, mono["features"][f]["models"][m]) for m in models)]),
                     hide_index=True, width="stretch",
                     column_config={"Largest drop (pp)": st.column_config.NumberColumn(format="%.1f")})
        st.caption(f"{mono['profiles']} profiles (3 demo inputs + 20 test records). Counts of profiles whose estimate "
                   "only rose, only fell, or did both as this input increased. 'Largest drop': biggest single-step "
                   "decrease along the grid.")
    note("This describes how each model's estimate changed under controlled input variation. Nothing is imposed: "
         "the models are not constrained to be monotone, and a direction here is model behaviour, not evidence "
         "that an input causes disease.")

    section("Model disagreement (prediction variation across candidate models)")
    dis = REL["disagreement"]
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        fig = go.Figure(go.Scatter(x=dis["spread_percentiles"], y=list(range(101)), mode="lines",
                                   line=dict(color=t["primary"], width=2.5), hovertemplate="%{y}% of records ≤ %{x:.1f} pp<extra></extra>"))
        chart(fig, height=300, xaxis_title="Spread: highest − lowest estimate of the 3 models (pp)",
              yaxis_title="Share of test records (%)", showlegend=False)
    with b:
        sp = dis["spread_pp"]
        kpis([("Median spread", f"{sp['median']:.1f} pp", f"95th pct {sp['p95']:.1f} pp"),
              ("Same class at 0.50", f"{dis['share_same_class_at_050']:.1%}", "all three models agree")])
    st.dataframe(pd.DataFrame([{"Case": c["case"], **{MODEL_NAMES[m]: c[m] for m in models},
                                "Spread (pp)": c["spread_pp"]} for c in dis["cases"]]), hide_index=True, width="stretch",
                 column_config={MODEL_NAMES[m]: st.column_config.NumberColumn(format="percent") for m in models} |
                 {"Spread (pp)": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Three models with similar overall accuracy can still give different estimates for one person. This "
               "is model disagreement, a consequence of model assumptions; it is not a calibrated measure of "
               "uncertainty.")

    section("Controlled distribution-shift experiment (synthetic)")
    note(f"<b>Synthetic experiment, not a validation.</b> {SHIFT['note']} Population-mix rows resample the test set "
         "toward higher or lower values of one feature (each record keeps its label). Measurement-offset rows "
         "record an input systematically higher or lower while labels stay the same.")
    feat = st.segmented_control("Shifted feature", ["age", "ap_hi", "bmi"], default="age",
                                format_func=lambda x: {"age": "Age", "ap_hi": "Systolic BP", "bmi": "BMI"}[x],
                                key="shift_feature", label_visibility="collapsed") or "age"
    mix = [s for s in SHIFT["population_mix"] if s["feature"] == feat]
    xs = [s["feature_mean"] for s in mix]
    base_x = mix[0]["feature_mean_baseline"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC-AUC", "Mean predicted − observed rate"])
    for m in models:
        ys = [s["metrics"][m]["roc_auc"] for s in mix]
        bias = [s["metrics"][m]["mean_predicted"] - s["metrics"][m]["observed_rate"] for s in mix]
        pts = sorted(zip(xs + [base_x], ys + [SHIFT["baseline"][m]["roc_auc"]],
                         bias + [SHIFT["baseline"][m]["mean_predicted"] - SHIFT["baseline"][m]["observed_rate"]]))
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=MODEL_NAMES[m],
                                 line=dict(color=MODEL_COLORS[m])), row=1, col=1)
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[2] for p in pts], mode="lines+markers", showlegend=False,
                                 line=dict(color=MODEL_COLORS[m])), row=1, col=2)
    for c in (1, 2):
        fig.add_vline(x=base_x, line_color=t["faint"], line_dash="dot", row=1, col=c)
        fig.update_xaxes(title_text=f"Mean {mix[0]['label'].lower()} of the resampled set (dotted: unshifted)", row=1, col=c)
    fig.update_yaxes(title_text="ROC-AUC", row=1, col=1)
    fig.update_yaxes(title_text="Predicted − observed", tickformat="+.1%", row=1, col=2)
    chart(fig, height=340)
    st.dataframe(pd.DataFrame([{"Measurement offset": s["label"], "Dropped (invalid)": s["dropped_invalid"],
                                **{f"{MODEL_NAMES[m]} ROC-AUC": s["metrics"][m]["roc_auc"] for m in models},
                                f"{MODEL_NAMES[FINAL]} predicted": s["metrics"][FINAL]["mean_predicted"],
                                "Observed rate": s["metrics"][FINAL]["observed_rate"]} for s in SHIFT["measurement_offset"]]),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                [f"{MODEL_NAMES[m]} ROC-AUC" for m in models] + [f"{MODEL_NAMES[FINAL]} predicted", "Observed rate"]})
    st.caption("A systematic offset barely changes ranking (ROC-AUC) but shifts every estimate, so mean predicted "
               "drifts away from the observed rate. Rows that become invalid (e.g. age above 65) are dropped, which "
               "itself changes the population.")
footer()
