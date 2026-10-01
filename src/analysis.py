"""Post-hoc analysis of the trained models, precomputed once into artifacts/analysis.json.

Run: python -m src.analysis   (after python -m src.train_models; takes about a minute)
Robustness analyses (OOF calibration, novelty, stability, disagreement) live in src/reliability.py.

Everything here uses the held-out TEST split for model behaviour (never used for fitting or selection)
and the cleaned dataset only for descriptive data-quality statistics. The UI reads the JSON; it never
recomputes these analyses per request.
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.data_loader import ROOT, SEED, TARGET, clean_with_exclusions, load_raw, split
from src.evaluate_models import (bootstrap_ci, calibration_data, decision_curve, full_metrics, subgroup_metrics,
                                 threshold_metrics)
from src.explainability import ModelExplainer, _column_owners
from src.prediction import load_model
from src.preprocessing import (CATEGORICAL, CATEGORY_LABELS, FEATURE_LABELS, FEATURES, MODEL_FEATURES, NUMERIC,
                               add_bmi, plausible_mask)

ARTIFACT = ROOT / "artifacts" / "analysis.json"
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
TABLE_THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.91, 0.1)]
GRID_THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.951, 0.01)]
SHAP_ROWS = 5000  # global SHAP on a fixed random sample of cleaned records (TreeSHAP on all 68k is slow)
AGE_BINS, AGE_LABELS = [29, 40, 50, 60, 66], ["29–39", "40–49", "50–59", "60–65"]
RESPONSE_FEATURES = ["ap_hi", "age", "ap_lo", "weight"]
DESCRIBE = ["age", "height", "weight", "ap_hi", "ap_lo"]


def _thin(xs, ys, n: int = 200) -> dict:
    idx = np.unique(np.linspace(0, len(xs) - 1, n).astype(int))
    return {"x": np.asarray(xs)[idx].round(5).tolist(), "y": np.asarray(ys)[idx].round(5).tolist()}


def data_quality(raw: pd.DataFrame, df: pd.DataFrame, removed: dict, excluded: dict) -> dict:
    raw_years = raw.assign(age=raw["age"] / 365.25)
    examples = {rule: rows.head(5).round(2).to_dict("records") for rule, rows in excluded.items() if len(rows)}
    hist = {}
    for col in ["age", "height", "weight", "bmi", "ap_hi", "ap_lo"]:
        values = add_bmi(df)[col]
        edges = np.histogram_bin_edges(values, bins=30)
        hist[col] = {"edges": edges.round(2).tolist(),
                     **{f"target_{t}": np.histogram(values[df[TARGET] == t], bins=edges)[0].tolist() for t in (0, 1)}}
    categorical = {col: [{"code": int(code), "label": CATEGORY_LABELS[col][int(code)], "n": int(len(g)),
                          "disease_rate": float(g[TARGET].mean())} for code, g in df.groupby(col)]
                   for col in CATEGORICAL}
    numeric = add_bmi(df)[["age", "height", "weight", "bmi", "ap_hi", "ap_lo"]]
    return {
        "raw_rows": len(raw), "clean_rows": len(df), "removed_rows": len(raw) - len(df),
        "removed_by_rule": removed, "examples": examples,
        "missing_values_raw": {c: int(v) for c, v in raw.isna().sum().items()},
        "duplicate_ids_raw": int(raw["id"].duplicated().sum()),
        "target_raw": {str(k): int(v) for k, v in raw[TARGET].value_counts().sort_index().items()},
        "target_clean": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "ranges": {c: {"raw": raw_years[c].quantile([0, .01, .5, .99, 1]).round(1).tolist(),
                       "clean": df[c].quantile([0, .01, .5, .99, 1]).round(1).tolist()} for c in DESCRIBE},
        "histograms": hist, "categorical": categorical,
        "correlation": {"features": list(numeric.columns), "matrix": numeric.corr().round(3).values.tolist()},
    }


def model_evaluation(models: dict, X_test, y_test) -> dict:
    out = {}
    for name, model in models.items():
        p = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        out[name] = {"test": full_metrics(y_test, p), "roc": _thin(fpr, tpr), "pr": _thin(rec, prec),
                     "calibration_test": calibration_data(y_test, p),
                     "confusion_matrix": confusion_matrix(y_test, p >= 0.5).tolist()}
    return out


def shap_global(explainer: ModelExplainer, df: pd.DataFrame) -> list[dict]:
    """Mean |SHAP| (score units of the final model) plus direction: for numeric features the sign of the value-vs-contribution slope,
    for categorical features the mean contribution of each level."""
    values = explainer.grouped_shap(df)
    data = add_bmi(df)
    mean_abs = np.abs(values).mean(axis=0)
    rows = []
    for j, f in enumerate(MODEL_FEATURES):
        row = {"feature": f, "label": FEATURE_LABELS[f], "mean_abs_shap": float(mean_abs[j]),
               "share": float(mean_abs[j] / mean_abs.sum())}
        if f in NUMERIC:
            slope = np.polyfit(data[f], values[:, j], 1)[0]
            row["direction"] = f"higher {FEATURE_LABELS[f].lower()} -> {'higher' if slope > 0 else 'lower'} estimate"
        else:
            row["levels"] = {CATEGORY_LABELS[f][int(k)]: float(values[(data[f] == k).to_numpy(), j].mean())
                             for k in sorted(data[f].unique())}
        rows.append(row)
    return sorted(rows, key=lambda r: -r["mean_abs_shap"])


def feature_response(models: dict, X: pd.DataFrame, n_ice: int = 30, points: int = 25) -> dict:
    """Partial dependence + ICE over the 2nd-98th percentile of each feature. Grid points that would
    create an invalid record (e.g. systolic <= diastolic, implausible BMI) are dropped for that row,
    so the averages never include impossible inputs; `n_valid` reports how many rows remain."""
    out = {}
    for f in RESPONSE_FEATURES:
        grid = np.unique(np.quantile(X[f], np.linspace(0.02, 0.98, points)).round(1))
        out[f] = {"label": FEATURE_LABELS[f], "grid": grid.tolist(), "models": {}}
        for name, model in models.items():
            avg, n_valid, ice = [], [], np.full((n_ice, len(grid)), np.nan)
            for i, g in enumerate(grid):
                Xg = X.assign(**{f: g})
                ok = plausible_mask(Xg).to_numpy()
                p = np.full(len(X), np.nan)
                p[ok] = model.predict_proba(Xg[ok])[:, 1]
                avg.append(float(np.nanmean(p)))
                n_valid.append(int(ok.sum()))
                ice[:, i] = p[:n_ice]
            out[f]["models"][name] = {"average": avg, "n_valid": n_valid,
                                      "ice": [[None if np.isnan(v) else round(float(v), 4) for v in r] for r in ice]}
    return out


def tree_interactions(pipeline, X: pd.DataFrame) -> dict:
    """XGBoost SHAP interaction values, grouped to model features. A pair's strength is the mean of
    |phi_ij + phi_ji| (the interaction is split across both off-diagonal cells)."""
    Z = pipeline.named_steps["pre"].transform(X)
    phi = np.asarray(shap.TreeExplainer(pipeline.named_steps["clf"]).shap_interaction_values(Z))
    owners = _column_owners(pipeline)
    k = len(MODEL_FEATURES)
    G = np.zeros((len(X), k, k))
    for a, oa in enumerate(owners):
        for b, ob in enumerate(owners):
            G[:, MODEL_FEATURES.index(oa), MODEL_FEATURES.index(ob)] += phi[:, a, b]
    main = np.abs(np.einsum("nii->ni", G)).mean(axis=0)
    pairs = [{"a": FEATURE_LABELS[MODEL_FEATURES[i]], "b": FEATURE_LABELS[MODEL_FEATURES[j]],
              "strength": float(np.abs(G[:, i, j] + G[:, j, i]).mean())}
             for i in range(k) for j in range(i + 1, k)]
    pairs.sort(key=lambda r: -r["strength"])
    return {"rows": len(X), "pairs": pairs[:10],
            "main_effects": {FEATURE_LABELS[f]: float(m) for f, m in zip(MODEL_FEATURES, main)},
            "interaction_share": float(sum(p["strength"] for p in pairs) / (sum(p["strength"] for p in pairs) + main.sum()))}


def main() -> dict:
    raw = load_raw()
    df, removed, excluded = clean_with_exclusions(raw)
    X_train, X_test, y_train, y_test = split(df)
    bundle = load_model()
    final = bundle["pipeline"]
    models = {n: joblib.load(ROOT / "models" / f"{n}.joblib") for n in MODEL_NAMES}
    p_final = final.predict_proba(X_test)[:, 1]
    rng = np.random.default_rng(SEED)
    sample = X_test.iloc[rng.choice(len(X_test), 2000, replace=False)]

    age_group = pd.cut(X_test["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    gender = X_test["gender"].map(CATEGORY_LABELS["gender"])
    permutation = permutation_importance(final, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                         random_state=SEED)
    report = {
        "final_model": bundle["model_name"], "calibration": bundle["calibration"],
        "test_rows": len(X_test), "seed": SEED,
        "data_quality": data_quality(raw, df, removed, excluded),
        "models": model_evaluation(models, X_test, y_test),
        # Descriptive only: no threshold is chosen from these test-set rows.
        "thresholds": {"table": threshold_metrics(y_test, p_final, TABLE_THRESHOLDS),
                       "grid": threshold_metrics(y_test, p_final, GRID_THRESHOLDS),
                       "decision_curve": decision_curve(y_test, p_final, GRID_THRESHOLDS)},
        "subgroups": {"gender": subgroup_metrics(y_test, p_final, gender),
                      "age_group": subgroup_metrics(y_test, p_final, age_group.astype(str))},
        "bootstrap": bootstrap_ci(y_test, {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()},
                                  n_resamples=1000, seed=SEED, reference=bundle["model_name"]),
        "shap_rows": SHAP_ROWS,
        "shap_global": shap_global(ModelExplainer(bundle), df.sample(SHAP_ROWS, random_state=SEED)),
        "permutation_importance": sorted(
            [{"feature": f, "label": FEATURE_LABELS[f], "mean": float(m), "std": float(s)}
             for f, m, s in zip(FEATURES, permutation.importances_mean, permutation.importances_std)],
            key=lambda r: -r["mean"]),
        "feature_response": feature_response(models, sample),
        "interactions": tree_interactions(models["xgboost"], sample.iloc[:1000]),
    }
    ARTIFACT.write_text(json.dumps(report, indent=1, allow_nan=False, default=float))
    return report


if __name__ == "__main__":
    r = main()
    print(f"Wrote {ARTIFACT} ({ARTIFACT.stat().st_size / 1e6:.2f} MB)")
    for n, m in r["bootstrap"]["models"].items():
        print(n, {k: f"{v['estimate']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]" for k, v in m.items()})
    for n, d in r["bootstrap"]["differences"].items():
        print("diff", n, "- final", {k: f"{v['estimate']:+.4f} [{v['lower']:+.4f}, {v['upper']:+.4f}]" for k, v in d.items()})
    for kind, rows in r["subgroups"].items():
        for g in rows:
            print(kind, g["group"], g["n"], g.get("roc_auc"), round(g["observed_rate"], 3), round(g["mean_predicted"], 3))
    print("perm", [(p["feature"], round(p["mean"], 4)) for p in r["permutation_importance"]])
    print("interactions", r["interactions"]["pairs"][:3], round(r["interactions"]["interaction_share"], 3))
