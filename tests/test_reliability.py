"""Phase 6: repeated CV, OOF integrity, test-set integrity, calibration, thresholds, novelty, stability,
disagreement, shift experiments and SHAP reconciliation."""
import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from src import reliability as rel
from src.data_loader import ROOT, SEED, load_dataset, split
from src.evaluate_models import (bootstrap_ci, calibration_stats, decision_curve, fold_assignment, oof_cross_validate,
                                 subgroup_metrics, threshold_metrics)
from src.explainability import ModelExplainer
from src.prediction import DEMO_INPUTS, load_model, predict
from src.shift_analysis import population_mix
from src.train_models import N_FOLDS, N_REPEATS, build_models, build_variant

rng = np.random.default_rng(1)
B = DEMO_INPUTS["Example Patient B"]


@pytest.fixture(scope="module")
def data():
    return split(load_dataset())


@pytest.fixture(scope="module")
def report():
    return json.loads((ROOT / "artifacts" / "metrics.json").read_text())


@pytest.fixture(scope="module")
def R():
    return json.loads(rel.ARTIFACT.read_text())


@pytest.fixture(scope="module")
def oof():
    f = np.load(rel.OOF_PATH)
    return {k: f[k] for k in f.files}


@pytest.fixture(scope="module")
def bundle():
    return load_model()


@pytest.fixture(scope="module")
def detector():
    return joblib.load(rel.DETECTOR_PATH)


# ---- repeated CV ------------------------------------------------------------------------------
def test_repeated_cv_splits_are_reproducible_complete_and_disjoint(data):
    X_train, X_test, y_train, _ = data
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(cv.split(X_train, y_train))
    again = list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED).split(X_train, y_train))
    assert len(splits) == 25
    for (tr, va), (tr2, va2) in zip(splits, again):
        assert np.array_equal(tr, tr2) and np.array_equal(va, va2)
        assert not set(tr) & set(va) and len(tr) + len(va) == len(X_train)
    for r in range(N_REPEATS):  # each repeat validates every row exactly once
        assert np.array_equal(np.sort(np.concatenate([va for _, va in splits[r * 5:(r + 1) * 5]])), np.arange(len(X_train)))
    assert not set(X_train.index) & set(X_test.index)  # CV runs on training rows only


# ---- OOF --------------------------------------------------------------------------------------
def test_oof_file_covers_every_training_row_with_valid_probabilities(oof, data, report):
    X_train, X_test, y_train, _ = data
    assert np.array_equal(oof["index"], X_train.index) and not set(oof["index"]) & set(X_test.index)
    assert np.array_equal(oof["y"], y_train)
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    assert np.array_equal(oof["fold"], fold_assignment(cv, X_train, y_train))
    for name in report["comparison"]:
        for variant in ("raw", "sigmoid", "isotonic"):
            p = oof[f"{name}__{variant}"]
            assert p.shape == (N_REPEATS, len(X_train)) and not np.isnan(p).any() and ((p >= 0) & (p <= 1)).all()


def test_repeated_oof_never_predicts_a_row_with_a_model_that_saw_it():
    """1-nearest-neighbour memorises its training rows: on random labels any leak would give ~100% accuracy."""
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    y = pd.Series(rng.integers(0, 2, 500))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    folds, p = oof_cross_validate(KNeighborsClassifier(1), X, y, cv)
    assert p.shape == (3, 500) and not np.isnan(p).any() and len(folds) == 15
    assert {(f["repeat"], f["fold"]) for f in folds} == {(r, k) for r in range(3) for k in range(5)}
    for r in range(3):
        assert 0.4 < np.mean((p[r] >= 0.5) == y) < 0.6


# ---- final test-set integrity -----------------------------------------------------------------
def test_final_model_is_fit_on_training_rows_only(bundle, report, data):
    """Refitting the selected variant on the training split alone reproduces the persisted final model, so
    neither the model nor any calibrator inside it saw test rows or test labels."""
    X_train, X_test, y_train, _ = data
    refit = build_variant(build_models()[bundle["model_name"]], bundle["calibration"]).fit(X_train, y_train)
    np.testing.assert_allclose(refit.predict_proba(X_test)[:, 1], bundle["pipeline"].predict_proba(X_test)[:, 1],
                               atol=1e-6)


def test_lr_calibrator_is_fit_on_training_rows_only(data):
    X_train, X_test, y_train, _ = data
    saved = joblib.load(ROOT / "models" / "logistic_regression.joblib")
    refit = build_variant(build_models()["logistic_regression"], "isotonic").fit(X_train, y_train)
    np.testing.assert_allclose(refit.predict_proba(X_test)[:, 1], saved.predict_proba(X_test)[:, 1], atol=1e-12)


def test_no_threshold_is_tuned(R):
    """No artifact selects a threshold; the class output stays at the fixed 0.50."""
    for path in ("analysis.json", "reliability.json", "metrics.json"):
        text = (ROOT / "artifacts" / path).read_text().lower()
        assert "best_threshold" not in text and "optimal" not in text
    r = predict(B)
    assert r.predicted_class == int(r.probability_positive >= 0.5)


# ---- calibration ------------------------------------------------------------------------------
def test_calibration_stats_on_known_cases():
    p = rng.uniform(0.05, 0.95, 20000)
    y = rng.uniform(size=p.size) < p                    # perfectly calibrated by construction
    s = calibration_stats(y, p)
    assert s["slope"] == pytest.approx(1, abs=0.1) and s["intercept"] == pytest.approx(0, abs=0.05) and s["ece"] < 0.02
    too_extreme = np.clip(0.5 + (p - 0.5) * 1.6, 0.001, 0.999)
    assert calibration_stats(y, too_extreme)["slope"] < 0.8


def test_calibration_artifact_is_reproducible(R, oof, report):
    final = R["final_model"]
    again = rel.calibration_deep_dive(oof, oof["y"], report, final)
    assert again == {k: R["calibration"][k] for k in again}
    for m in R["calibration"]["models"].values():
        for v in m["variants"].values():
            assert sum(b["count"] for b in v["bins"]) == len(oof["y"])
            assert all(0 <= b["mean_predicted"] <= 1 and 0 <= b["observed_rate"] <= 1 for b in v["bins"])


# ---- thresholds and decision curve ------------------------------------------------------------
def test_threshold_grid_is_sorted_and_consistent(R):
    for grid in (R["thresholds_oof"]["grid"], json.loads((ROOT / "artifacts" / "analysis.json").read_text())["thresholds"]["grid"]):
        ts = [r["threshold"] for r in grid]
        assert ts == sorted(ts) and ts[0] == 0.05 and ts[-1] == 0.95
        for r in grid:
            n = r["tp"] + r["fp"] + r["tn"] + r["fn"]
            assert r["accuracy"] == pytest.approx((r["tp"] + r["tn"]) / n)
            assert r["specificity"] + r["fpr"] == pytest.approx(1) and r["recall"] + r["fnr"] == pytest.approx(1)


def test_decision_curve_formula():
    y, p = np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.6, 0.1])
    row = decision_curve(y, p, [0.5])[0]
    assert row["model"] == pytest.approx(1 / 4 - 1 / 4 * 1)    # TP=1, FP=1, weight 0.5/0.5
    assert row["flag_all"] == pytest.approx(0.5 - 0.5 * 1)
    assert threshold_metrics(y, p, [0.5])[0]["accuracy"] == 0.5


# ---- bootstrap --------------------------------------------------------------------------------
def test_bootstrap_intervals_are_ordered_and_in_range():
    y = rng.integers(0, 2, 800)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 800), 0, 1)
    out = bootstrap_ci(y, {"m": p}, n_resamples=100, seed=3)
    assert out == bootstrap_ci(y, {"m": p}, n_resamples=100, seed=3)
    for metric, c in out["models"]["m"].items():
        assert 0 <= c["lower"] <= c["upper"] <= 1, metric
        if metric != "ece":  # binned ECE is biased upward under resampling (documented)
            assert c["lower"] <= c["estimate"] <= c["upper"]


# ---- subgroups --------------------------------------------------------------------------------
def test_subgroup_counts_and_unavailable_metrics():
    y = np.r_[np.ones(50), rng.integers(0, 2, 400)]
    p = rng.uniform(size=450)
    groups = np.r_[["single"] * 50, ["mixed"] * 400]
    rows = {r["group"]: r for r in subgroup_metrics(y, p, groups, n_resamples=50)}
    assert rows["single"]["n"] == 50 and rows["single"]["prevalence"] == 1.0
    assert rows["single"]["unavailable"] == "Metric unavailable for this subgroup: only one outcome class."
    assert "roc_auc" not in rows["single"] and "pr_auc" not in rows["single"]
    m = rows["mixed"]
    assert m["n"] == 400 and m["positives"] == int(y[50:].sum())
    for k, (lo, hi) in m["ci"].items():
        assert lo <= hi


# ---- novelty / input conformity ---------------------------------------------------------------
def test_detector_is_fit_on_training_rows_only(detector, data):
    X_train, X_test, _, _ = data
    assert set(detector["fit_index"]) <= set(X_train.index)
    assert not set(detector["fit_index"]) & set(X_test.index)


def test_detector_choice_does_not_depend_on_the_test_set(R, data):
    """Detector comparison and threshold use training rows only: any other test set gives the same choice."""
    X_train, X_test, _, _ = data
    summary, _ = rel.novelty_detection(X_train, X_test.iloc[:500])
    assert summary["chosen"] == R["novelty"]["chosen"]
    for name, m in summary["methods"].items():
        for k in ("threshold", "flag_rate_reference", "flag_rate_atypical", "auc_reference_vs_atypical"):
            assert m[k] == pytest.approx(R["novelty"]["methods"][name][k], abs=1e-9)


def test_conformity_flags_unusual_inputs_without_blocking(detector):
    normal = rel.input_conformity(B, detector)
    assert not normal["unusual"] and normal["unusual_features"] == []
    extreme = replace(B, ap_hi=240.0, ap_lo=60.0, weight=140.0)  # valid, but far from the training data
    flagged = rel.input_conformity(extreme, detector)
    assert flagged["unusual"] and "ap_hi" in flagged["unusual_features"]
    assert 0 <= predict(extreme).probability_positive <= 1        # still predicted
    ref = detector["reference_scores"].copy()
    rel.input_conformity(extreme, detector)
    assert np.array_equal(ref, detector["reference_scores"])       # assessing never refits


# ---- stability --------------------------------------------------------------------------------
def test_perturbations_are_deterministic_and_valid():
    base = pd.concat([p.to_frame() for p in DEMO_INPUTS.values()], ignore_index=True).astype(float)
    a, b = rel.perturbations(base), rel.perturbations(base)
    pd.testing.assert_frame_equal(a, b)
    assert rel.plausible_mask(a).all() and set(a["profile"]) == {0, 1, 2}
    p = load_model()["pipeline"].predict_proba(a[rel.FEATURES])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_explanation_similarity_statistic():
    phi = np.array([0.5, -0.3, 0.2, 0.1, -0.05, 0.01])
    same = rel.explanation_similarity(phi, phi)
    assert same == {"spearman": 1.0, "top_k_overlap": 1.0, "l1": 0.0, "same_top1": True}
    flipped = rel.explanation_similarity(phi, phi[::-1])
    assert -1 <= flipped["spearman"] < 1 and 0 <= flipped["top_k_overlap"] <= 1 and flipped["l1"] > 0


def test_local_stability_is_deterministic(bundle):
    ex = ModelExplainer(bundle)
    a, b = rel.local_stability(B, ex, bundle["pipeline"]), rel.local_stability(B, ex, bundle["pipeline"])
    assert a == b and 0 <= a["share_full_top_k_overlap"] <= 1 and -1 <= a["min_spearman"] <= 1 and a["n"] > 0


def test_stability_artifact_ranges(R):
    for m in R["perturbation"]["prediction"].values():
        for d, v in m.items():
            if d.endswith("%"):
                assert 0 <= v["median"] <= v["p95"] <= v["max"] <= 100
    for m in R["explanation_stability"]["models"].values():
        for v in m.values():
            assert -1 <= v["spearman"]["min"] <= v["spearman"]["mean"] <= 1 and 0 <= v["top_k_overlap"]["mean"] <= 1


def test_monotonicity_shape_and_linear_candidate(R):
    assert rel.shape(np.array([0.1, 0.2, 0.2, 0.3])) == "non-decreasing"
    assert rel.shape(np.array([0.3, 0.2])) == "non-increasing"
    assert rel.shape(np.array([0.1, 0.3, 0.2])) == "non-monotone"
    lr = R["monotonicity"]["features"]["ap_hi"]["models"]["logistic_regression"]["counts"]
    assert lr["non-decreasing"] == R["monotonicity"]["profiles"]  # linear score + monotone calibration


# ---- disagreement ------------------------------------------------------------------------------
def test_candidate_models_return_valid_probabilities_and_spread(R):
    candidates = rel.load_candidates()
    d = rel.model_disagreement(B, candidates, R["disagreement"]["spread_percentiles"])
    assert set(d["probabilities"]) == set(candidates) and all(0 <= p <= 1 for p in d["probabilities"].values())
    assert d["spread_pp"] == pytest.approx((max(d["probabilities"].values()) - min(d["probabilities"].values())) * 100)
    assert 0 <= d["test_percentile"] <= 100
    assert d["probabilities"][R["final_model"]] == pytest.approx(predict(B).probability_positive)


# ---- SHAP reconciliation (regression guard) -----------------------------------------------------
def test_final_shap_reconciles_with_score_and_probability(bundle, data):
    X = data[1].sample(200, random_state=0)
    ex = ModelExplainer(bundle)
    recon = ex.base_value + ex.grouped_shap(X).sum(axis=1)
    np.testing.assert_allclose(recon, ex.model_score(ex.pre.transform(X)), atol=1e-5)
    np.testing.assert_allclose([ex.score_to_probability(s) for s in recon], bundle["pipeline"].predict_proba(X)[:, 1],
                               atol=1e-5)


def test_reconciliation_artifact_within_tolerance(R):
    for v in R["reconciliation"].values():
        assert v["max_score_error"] < 1e-5 and v["max_probability_error"] < 1e-5


# ---- shift ------------------------------------------------------------------------------------
def test_shift_experiment_is_reproducible_and_separate(data):
    X_test = data[1]
    assert np.array_equal(population_mix(X_test, "age", 1.0, 7), population_mix(X_test, "age", 1.0, 7))
    older = X_test.iloc[population_mix(X_test, "age", 1.0, 7)]["age"].mean()
    assert older > X_test["age"].mean()
    shift = json.loads((ROOT / "artifacts" / "shift_analysis.json").read_text())
    assert shift["synthetic"] is True and "Not a real-world" in shift["note"]
    analysis = (ROOT / "artifacts" / "analysis.json").read_text()
    assert "population_mix" not in analysis and "measurement_offset" not in analysis
