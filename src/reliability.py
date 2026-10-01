"""Reliability and robustness analyses, precomputed into artifacts/reliability.json, plus the lightweight
per-assessment checks the UI runs (input conformity, local explanation stability, model disagreement).

Run: python -m src.reliability   (after src.train_models and src.analysis; a few minutes)

Data use: out-of-fold (OOF) predictions of the TRAINING split for calibration and threshold exploration;
the held-out TEST split only for descriptive, final-model behaviour (nothing here is tuned on it). The
input-conformity detector is fit on training rows only and persisted to models/novelty_detector.joblib.
None of these analyses is a clinical validation.
"""
import json
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_loader import ROOT, SEED, load_dataset, provenance, split
from src.evaluate_models import calibration_data, calibration_stats, decision_curve, threshold_metrics
from src.explainability import ModelExplainer
from src.prediction import DEMO_INPUTS, PatientInput, load_model
from src.preprocessing import FEATURES, NUMERIC, PLAUSIBLE, add_bmi, build_preprocessor, plausible_mask
from src.train_models import build_models, build_variant

ARTIFACT = ROOT / "artifacts" / "reliability.json"
OOF_PATH = ROOT / "artifacts" / "oof_predictions.npz"
DETECTOR_PATH = ROOT / "models" / "novelty_detector.joblib"
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
GRID = [round(t, 2) for t in np.arange(0.05, 0.951, 0.01)]
CONFUSION_AT = [0.3, 0.5, 0.7]
SPARSE_BIN = 100              # reliability bins with fewer rows are drawn hollow and not interpreted
REFERENCE_FLAG_RATE = 0.01    # detector threshold = 99th percentile of held-back REFERENCE (training) scores
PERTURB_FEATURES = ["age", "height", "weight", "ap_hi", "ap_lo"]  # categorical inputs have no "1%" change
DELTAS = [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]
N_JOINT = 4                   # random joint perturbations per |delta| per profile
N_PROFILES = 150              # random test records used as baseline profiles (plus the 3 demo inputs)
TOP_K = 5
CURVE = {"ap_hi": 2.0, "ap_lo": 2.0, "age": 1.0, "weight": 1.0}  # monotonicity grid step per feature
N_CURVE_PROFILES = 20


def load_candidates() -> dict:
    """Each candidate in its protocol-selected variant (fit on the training split)."""
    return {n: joblib.load(ROOT / "models" / f"{n}.joblib") for n in MODEL_NAMES}


# ---- calibration deep dive (OOF, training split) ---------------------------------------------
def calibration_deep_dive(oof, y, metrics: dict, final: str) -> dict:
    """Raw / sigmoid / isotonic for Logistic Regression and for the final model, from repeated-CV OOF
    probabilities. Every probability comes from a model (and calibrator) fit without that row. Statistics
    are computed per repeat (each repeat predicts every row once) and summarised as mean +- std; the
    reliability curve uses repeat 1, so bin counts are real row counts."""
    out = {}
    for name in dict.fromkeys(["logistic_regression", final]):
        out[name] = {"selected": metrics["comparison"][name]["selected"], "variants": {}}
        for variant in ("raw", "sigmoid", "isotonic"):
            p = oof[f"{name}__{variant}"]
            per_repeat = [calibration_stats(y, r) for r in p]
            out[name]["variants"][variant] = {
                "stats": {k: {"mean": float(np.mean([r[k] for r in per_repeat])),
                              "std": float(np.std([r[k] for r in per_repeat]))} for k in per_repeat[0]},
                "bins": [{**b, "sparse": b["count"] < SPARSE_BIN} for b in calibration_data(y, p[0])],
            }
    return {"source": "repeated-CV out-of-fold predictions, training split", "bins": "10 equal-width",
            "sparse_bin_rows": SPARSE_BIN, "models": out}


def calibration_on_test(X_train, y_train, X_test, y_test, candidates: dict) -> dict:
    """Final-evaluation calibration statistics on the untouched test set (reported only). The Logistic
    Regression raw/sigmoid/isotonic variants are refit on the training split for this comparison."""
    lr = build_models()["logistic_regression"]
    out = {f"logistic_regression__{v}": calibration_stats(
        y_test, build_variant(lr, v).fit(X_train, y_train).predict_proba(X_test)[:, 1]) for v in ("raw", "sigmoid", "isotonic")}
    out.update({f"{n}__selected": calibration_stats(y_test, m.predict_proba(X_test)[:, 1]) for n, m in candidates.items()})
    return out


# ---- thresholds on OOF (exploratory) ---------------------------------------------------------
def oof_thresholds(oof, y, final: str, variant: str) -> dict:
    p = oof[f"{final}__{variant}"][0]
    grid = threshold_metrics(y, p, GRID)
    return {"source": "out-of-fold predictions of the final model, training split, repeat 1",
            "grid": grid, "confusion": [r for r in grid if r["threshold"] in CONFUSION_AT],
            "decision_curve": decision_curve(y, p, GRID)}


# ---- input conformity / novelty detection ----------------------------------------------------
# Two detectors, stored as plain scikit-learn objects (higher score = more unusual):
#   mahalanobis: squared Mahalanobis distance on the scaled continuous model features (age, BMI, systolic,
#     diastolic), classical covariance. The robust MCD estimate degenerates on this data (blood pressures are
#     heavily rounded, e.g. 120/80, so its half-sample subsets become near-singular).
#   isolation_forest: Isolation Forest on every preprocessed model column (continuous + one-hot).
DETECTORS = ["mahalanobis", "isolation_forest"]


def fit_detector(method: str, Z: np.ndarray):
    if method == "mahalanobis":
        return EmpiricalCovariance().fit(Z[:, :len(NUMERIC)])
    return IsolationForest(n_estimators=200, random_state=SEED).fit(Z)


def score_detector(method: str, model, Z: np.ndarray) -> np.ndarray:
    return model.mahalanobis(Z[:, :len(NUMERIC)]) if method == "mahalanobis" else -model.score_samples(Z)


def atypical_combinations(X: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Evaluation set: every column of X permuted independently, then only VALID records kept. Each
    value is realistic on its own; the combination is often not (e.g. 64 years, BMI 19, 170/95)."""
    rng = np.random.default_rng(seed)
    shuffled = pd.DataFrame({c: rng.permutation(X[c].to_numpy()) for c in X.columns})
    return shuffled[plausible_mask(shuffled).to_numpy()].reset_index(drop=True)


def novelty_detection(X_train, X_test) -> tuple[dict, dict]:
    """Pre-declared procedure: fit each detector on 80% of the TRAINING split, set its threshold at the
    99th percentile of the other 20% (the "reference" flag rate is 1% by construction), then compare the
    detectors on test-set records vs atypical combinations of test values (ROC-AUC of the score). The higher
    AUC wins; a tie goes to the simpler Mahalanobis distance. The user's input is never used to fit."""
    X_fit, X_ref = train_test_split(X_train, test_size=0.2, random_state=SEED)
    pre = build_preprocessor().fit(X_fit)
    atypical = atypical_combinations(X_test)
    Z_fit, Z_ref, Z_test, Z_atyp = (pre.transform(x) for x in (X_fit, X_ref, X_test, atypical))
    report, fitted = {}, {}
    for name in DETECTORS:
        det = fit_detector(name, Z_fit)
        ref = score_detector(name, det, Z_ref)
        threshold = float(np.quantile(ref, 1 - REFERENCE_FLAG_RATE))
        s_test, s_atyp = score_detector(name, det, Z_test), score_detector(name, det, Z_atyp)
        report[name] = {"threshold": threshold, "flag_rate_reference": float((ref > threshold).mean()),
                        "flag_rate_test": float((s_test > threshold).mean()),
                        "flag_rate_atypical": float((s_atyp > threshold).mean()),
                        "auc_test_vs_atypical": float(roc_auc_score(
                            np.r_[np.zeros(len(s_test)), np.ones(len(s_atyp))], np.r_[s_test, s_atyp]))}
        fitted[name] = {"detector": det, "threshold": threshold, "reference_scores": np.sort(ref)}
    chosen = max(DETECTORS, key=lambda n: (round(report[n]["auc_test_vs_atypical"], 3), n == "mahalanobis"))
    raw_ref = add_bmi(X_ref)
    bundle = {"method": chosen, "pre": pre, **fitted[chosen], "fit_index": X_fit.index.to_numpy(),
              "ranges": {f: [float(raw_ref[f].quantile(0.005)), float(raw_ref[f].quantile(0.995))]
                         for f in ["age", "bmi", "ap_hi", "ap_lo"]}}
    summary = {"space": "model preprocessing (BMI derived, numerics standard-scaled, categoricals one-hot), "
                        "refit on the detector's own fitting rows",
               "fit_rows": len(X_fit), "reference_rows": len(X_ref), "reference_flag_rate": REFERENCE_FLAG_RATE,
               "atypical_rows": len(atypical), "methods": report, "chosen": chosen,
               "unusual_value_range": "0.5th-99.5th percentile of reference rows", "ranges": bundle["ranges"]}
    return summary, bundle


def input_conformity(patient: PatientInput, detector: dict) -> dict:
    """Statistical check of a validated input against the training reference population. Never blocks a
    prediction and is not a medical judgement."""
    score = float(detector_scores(detector, patient.to_frame())[0])
    values = {**asdict(patient), "bmi": patient.bmi}
    unusual = [f for f, (lo, hi) in detector["ranges"].items() if not lo <= values[f] <= hi]
    ref = detector["reference_scores"]
    return {"unusual": score > detector["threshold"], "score": score, "threshold": detector["threshold"],
            "reference_percentile": float(np.searchsorted(ref, score) / len(ref) * 100),
            "unusual_features": unusual, "method": detector["method"]}


def detector_scores(detector: dict, X: pd.DataFrame) -> np.ndarray:
    return score_detector(detector["method"], detector["detector"], detector["pre"].transform(X[FEATURES]))


# ---- perturbation sets -----------------------------------------------------------------------
def perturbations(base: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Deterministic small perturbations of each baseline row: every numeric input alone by each delta,
    plus N_JOINT random joint perturbations (each numeric input scaled by U(-|d|, |d|)) per |delta|.
    Invalid results (outside the input domain) are dropped, never repaired."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid, (_, r) in enumerate(base.iterrows()):
        for f in PERTURB_FEATURES:
            for d in DELTAS:
                rows.append({**r, f: r[f] * (1 + d), "profile": pid, "kind": "single", "feature": f, "delta": d})
        for d in sorted({abs(d) for d in DELTAS}):
            for _ in range(N_JOINT):
                scale = 1 + rng.uniform(-d, d, len(PERTURB_FEATURES))
                rows.append({**r, **{f: r[f] * s for f, s in zip(PERTURB_FEATURES, scale)},
                             "profile": pid, "kind": "joint", "feature": "all", "delta": d})
    out = pd.DataFrame(rows)
    return out[plausible_mask(out).to_numpy()].reset_index(drop=True)


def baseline_profiles(X_test) -> pd.DataFrame:
    demos = pd.concat([p.to_frame() for p in DEMO_INPUTS.values()], ignore_index=True)
    sample = X_test.sample(N_PROFILES, random_state=SEED)
    return pd.concat([demos, sample], ignore_index=True)[FEATURES].astype(float)


def _q(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
            "max": float(v.max()), "min": float(v.min())}


def prediction_stability(candidates: dict, base: pd.DataFrame, pert: pd.DataFrame) -> dict:
    """|change in probability| (percentage points) under the perturbations, per model and |delta|."""
    out = {}
    for name, model in candidates.items():
        p0 = model.predict_proba(base)[:, 1]
        dp = np.abs(model.predict_proba(pert[FEATURES])[:, 1] - p0[pert["profile"]]) * 100
        out[name] = {}
        for d in sorted(pert["delta"].abs().unique()):
            mask = (pert["delta"].abs() == d).to_numpy()
            out[name][f"{d:.0%}"] = {**_q(dp[mask]), "share_unchanged": float((dp[mask] == 0).mean()),
                                     "n": int(mask.sum())}
        single = pert["kind"] == "single"
        out[name]["by_feature_5pct"] = {f: _q(dp[(single & (pert["feature"] == f) & (pert["delta"].abs() == 0.05)).to_numpy()])
                                        for f in PERTURB_FEATURES}
    return out


def explanation_similarity(phi0: np.ndarray, phi: np.ndarray) -> dict:
    """Spearman rank correlation of |SHAP|, top-k overlap (|top-k ∩ top-k'| / k), and normalised L1 distance
    sum|phi - phi'| / sum|phi|. Rows are model features."""
    a, b = np.abs(phi0), np.abs(phi)
    rho = stats.spearmanr(a, b).statistic if np.ptp(a) and np.ptp(b) else float(np.array_equal(a, b))
    top = lambda v: set(np.argsort(-v)[:TOP_K])
    return {"spearman": float(rho), "top_k_overlap": len(top(a) & top(b)) / TOP_K,
            "l1": float(np.abs(phi0 - phi).sum() / max(a.sum(), 1e-12)), "same_top1": bool(a.argmax() == b.argmax())}


def explanation_stability(explainers: dict, base: pd.DataFrame, pert: pd.DataFrame) -> dict:
    out = {}
    for name, ex in explainers.items():
        phi0, phi = ex.grouped_shap(base), ex.grouped_shap(pert)
        sims = pd.DataFrame([explanation_similarity(phi0[p], phi[i]) for i, p in enumerate(pert["profile"])])
        out[name] = {}
        for d in sorted(pert["delta"].abs().unique()):
            s = sims[(pert["delta"].abs() == d).to_numpy()]
            out[name][f"{d:.0%}"] = {"spearman": _q(s["spearman"]), "top_k_overlap": _q(s["top_k_overlap"]),
                                     "l1": _q(s["l1"]), "share_same_top1": float(s["same_top1"].mean()),
                                     "share_full_top_k_overlap": float((s["top_k_overlap"] == 1).mean()),
                                     "n": len(s)}
    return out


def local_stability(patient: PatientInput, explainer: ModelExplainer, pipeline) -> dict:
    """Per-assessment version (UI): every numeric input alone by +-1, 2, 5%."""
    base = patient.to_frame().astype(float)
    pert = perturbations(base)
    pert = pert[pert["kind"] == "single"]
    phi0, phi = explainer.grouped_shap(base)[0], explainer.grouped_shap(pert)
    sims = [explanation_similarity(phi0, row) for row in phi]
    dp = np.abs(pipeline.predict_proba(pert[FEATURES])[:, 1] - pipeline.predict_proba(base)[0, 1]) * 100
    return {"n": len(pert), "min_top_k_overlap": min(s["top_k_overlap"] for s in sims),
            "share_full_top_k_overlap": float(np.mean([s["top_k_overlap"] == 1 for s in sims])),
            "min_spearman": min(s["spearman"] for s in sims), "max_change_pp": float(dp.max())}


# ---- monotonicity / controlled sensitivity ----------------------------------------------------
def response(model, row: pd.Series, feature: str) -> tuple[np.ndarray, np.ndarray]:
    low, high = PLAUSIBLE[feature]
    grid = np.arange(low, high + CURVE[feature] / 2, CURVE[feature])
    X = pd.DataFrame([row.to_dict()] * len(grid)).assign(**{feature: grid})
    ok = plausible_mask(X).to_numpy()
    return grid[ok], model.predict_proba(X[ok][FEATURES])[:, 1]


def shape(p: np.ndarray) -> str:
    d = np.diff(p)
    return "non-decreasing" if (d >= -1e-12).all() else "non-increasing" if (d <= 1e-12).all() else "non-monotone"


def monotonicity(candidates: dict, X_test) -> dict:
    """Controlled grid of one input, all others fixed, for the demo inputs and N_CURVE_PROFILES test records.
    Describes the direction of the model's response; imposes nothing."""
    profiles = pd.concat([baseline_profiles(X_test).iloc[:3],
                          X_test.sample(N_CURVE_PROFILES, random_state=SEED + 1)[FEATURES].astype(float)],
                         ignore_index=True)
    out = {"profiles": len(profiles), "features": {}}
    for f in CURVE:
        out["features"][f] = {"models": {}}
        for name, model in candidates.items():
            shapes, drops, curves = [], [], []
            for i, row in profiles.iterrows():
                x, p = response(model, row, f)
                shapes.append(shape(p))
                drops.append(float(max(0.0, -(np.diff(p).min()))) if len(p) > 1 else 0.0)
                if i < 3:
                    curves.append({"profile": list(DEMO_INPUTS)[i], "x": x.round(1).tolist(), "p": p.round(4).tolist()})
            out["features"][f]["models"][name] = {
                "counts": {s: shapes.count(s) for s in ("non-decreasing", "non-increasing", "non-monotone")},
                "largest_single_step_decrease_pp": float(max(drops) * 100), "demo_curves": curves}
    return out


# ---- model disagreement ----------------------------------------------------------------------
def disagreement(candidates: dict, final: str, X_test) -> dict:
    """Prediction variation across the three candidate models on the test set (descriptive)."""
    P = pd.DataFrame({n: m.predict_proba(X_test)[:, 1] for n, m in candidates.items()}, index=X_test.index)
    spread = (P.max(axis=1) - P.min(axis=1)) * 100
    classes = (P >= 0.5).astype(int)
    demos = pd.concat([p.to_frame() for p in DEMO_INPUTS.values()], ignore_index=True)
    demo_p = {n: m.predict_proba(demos)[:, 1] for n, m in candidates.items()}
    cases = [{"case": label, **{n: float(demo_p[n][i]) for n in candidates}} for i, label in enumerate(DEMO_INPUTS)]
    for q in (50, 90, 99):
        idx = (spread - spread.quantile(q / 100)).abs().idxmin()
        cases.append({"case": f"Test record at the {q}th spread percentile", **P.loc[idx].to_dict()})
    for c in cases:
        values = [c[n] for n in candidates]
        c.update(min=min(values), max=max(values), spread_pp=(max(values) - min(values)) * 100)
    return {"final_model": final, "rows": len(P),
            "spread_pp": {**_q(spread), "p25": float(spread.quantile(.25)), "p75": float(spread.quantile(.75))},
            "share_spread_at_least_pp": {str(t): float((spread >= t).mean()) for t in (5, 10, 20)},
            "spread_percentiles": np.percentile(spread, np.arange(101)).round(3).tolist(),
            "share_same_class_at_050": float((classes.nunique(axis=1) == 1).mean()),
            "correlation": P.corr().round(4).to_dict(), "cases": cases}


def model_disagreement(patient: PatientInput, candidates: dict, spread_percentiles: list) -> dict:
    X = patient.to_frame()
    probs = {n: float(m.predict_proba(X)[0, 1]) for n, m in candidates.items()}
    spread = (max(probs.values()) - min(probs.values())) * 100
    return {"probabilities": probs, "spread_pp": spread,
            "test_percentile": float(np.searchsorted(spread_percentiles, spread, side="right") - 1)}


# ---- SHAP reconciliation ---------------------------------------------------------------------
def reconciliation(explainers: dict, candidates: dict, X: pd.DataFrame) -> dict:
    """max |base + sum(SHAP) - model score| and max |displayed probability from SHAP - pipeline probability|."""
    out = {}
    for name, ex in explainers.items():
        phi = ex.grouped_shap(X)
        score = ex.model_score(ex.pre.transform(X))
        recon = ex.base_value + phi.sum(axis=1)
        p_shap = np.array([ex.score_to_probability(s) for s in recon])
        out[name] = {"rows": len(X), "max_score_error": float(np.abs(recon - score).max()),
                     "max_probability_error": float(np.abs(p_shap - candidates[name].predict_proba(X)[:, 1]).max())}
    return out


def main() -> dict:
    df = load_dataset()
    X_train, X_test, y_train, y_test = split(df)
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    final = load_model()["model_name"]
    candidates = load_candidates()
    explainers = {n: ModelExplainer({"pipeline": m}) for n, m in candidates.items()}
    oof_file = np.load(OOF_PATH)
    oof = {k: oof_file[k] for k in oof_file.files}
    if not np.array_equal(oof["index"], X_train.index.to_numpy()):
        raise RuntimeError("OOF file does not match the training split; rerun python -m src.train_models.")
    y = oof["y"]

    novelty, detector = novelty_detection(X_train, X_test)
    joblib.dump(detector, DETECTOR_PATH)
    base = baseline_profiles(X_test)
    pert = perturbations(base)
    print("perturbation rows", len(pert), flush=True)
    report = {"provenance": provenance(),
        "final_model": final, "seed": SEED,
        "calibration": {**calibration_deep_dive(oof, y, metrics, final),
                        "test": calibration_on_test(X_train, y_train, X_test, y_test, candidates)},
        "thresholds_oof": oof_thresholds(oof, y, final, metrics["comparison"][final]["selected"]),
        "novelty": novelty,
        "perturbation": {"profiles": len(base), "rows": len(pert), "features": PERTURB_FEATURES,
                         "deltas": DELTAS, "joint_per_delta": N_JOINT,
                         "prediction": prediction_stability(candidates, base, pert)},
        "explanation_stability": {"top_k": TOP_K, "models": explanation_stability(explainers, base, pert)},
        "reconciliation": reconciliation(explainers, candidates, X_test.sample(1000, random_state=SEED)),
        "monotonicity": monotonicity(candidates, X_test),
        "disagreement": disagreement(candidates, final, X_test),
    }
    ARTIFACT.write_text(json.dumps(report, indent=1, allow_nan=False, default=float))
    return report


if __name__ == "__main__":
    r = main()
    print(f"Wrote {ARTIFACT} ({ARTIFACT.stat().st_size / 1e6:.2f} MB)")
    for n, m in r["calibration"]["models"].items():
        for v, d in m["variants"].items():
            print("calib", n, v, {k: f"{s['mean']:.4f}±{s['std']:.4f}" for k, s in d["stats"].items()})
    print("calib test", {k: {m: round(x, 4) for m, x in v.items()} for k, v in r["calibration"]["test"].items()})
    print("novelty", r["novelty"]["chosen"], r["novelty"]["methods"])
    for n, d in r["perturbation"]["prediction"].items():
        print("pert", n, {k: (round(v["median"], 2), round(v["p95"], 2), round(v["max"], 2)) for k, v in d.items() if "%" in k})
    for n, d in r["explanation_stability"]["models"].items():
        print("expl", n, {k: (round(v["spearman"]["mean"], 3), round(v["top_k_overlap"]["mean"], 3), round(v["share_full_top_k_overlap"], 3)) for k, v in d.items()})
    print("recon", r["reconciliation"])
    for f, d in r["monotonicity"]["features"].items():
        print("mono", f, {n: (m["counts"], round(m["largest_single_step_decrease_pp"], 2)) for n, m in d["models"].items()})
    print("disagree", r["disagreement"]["spread_pp"], r["disagreement"]["share_spread_at_least_pp"], r["disagreement"]["share_same_class_at_050"])
