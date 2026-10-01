"""Model selection, calibration and the precomputed analysis artifact (real data, real models)."""
import json
import math

import pytest

from src.analysis import AGE_LABELS, ARTIFACT, MODEL_NAMES, TABLE_THRESHOLDS
from src.data_loader import ROOT, load_dataset, split
from src.evaluate_models import full_metrics
from src.prediction import FINAL_MODEL_PATH, load_model
from src.preprocessing import FEATURES, MODEL_FEATURES
from src.train_models import N_FOLDS, N_REPEATS, choose_variant, select_final


@pytest.fixture(scope="module")
def report():
    return json.loads((ROOT / "artifacts" / "metrics.json").read_text())


@pytest.fixture(scope="module")
def analysis():
    if not ARTIFACT.exists():
        from src.analysis import main
        main()
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def bundle():
    if not FINAL_MODEL_PATH.exists():
        from src.train_models import main
        main()
    return load_model()


def test_comparison_covers_all_models_variants_and_metrics(report):
    assert report["cv"] == {"scheme": "RepeatedStratifiedKFold", "n_splits": N_FOLDS, "n_repeats": N_REPEATS,
                            "random_state": 42}
    for name in MODEL_NAMES:
        for variant in ("raw", "sigmoid", "isotonic"):
            entry = report["comparison"][name][variant]
            assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss", "brier"} <= set(entry["cv"])
            assert len(entry["cv"]["roc_auc"]["folds"]) == N_FOLDS * N_REPEATS == 25
            assert all(0 <= b["mean_predicted"] <= 1 for b in entry["oof_calibration"])
            assert "test" not in entry  # variants are never scored on the test set
        assert set(report["comparison"][name]["test"]) >= {"roc_auc", "brier"}  # selected variant only


def test_selection_protocol_reproduces_recorded_decision(report):
    comparison = report["comparison"]
    for name in MODEL_NAMES:
        assert choose_variant(comparison[name])[0] == comparison[name]["selected"]
    selection = select_final(comparison)
    assert {k: selection[k] for k in ("model", "variant", "steps")} == \
        {k: report["selection"][k] for k in ("model", "variant", "steps")}
    assert report["final_model"] == selection["model"]


def test_selection_ignores_test_set(report):
    """Removing or scrambling every test-set number must not change any selection decision."""
    import copy
    scrambled = copy.deepcopy(report["comparison"])
    for name in MODEL_NAMES:
        scrambled[name]["test"] = {k: 0.0 for k in scrambled[name]["test"]}
    assert select_final(scrambled)["model"] == report["selection"]["model"]
    for name in MODEL_NAMES:
        del scrambled[name]["test"]
    assert select_final(scrambled)["model"] == report["selection"]["model"]


def test_selection_rules_on_synthetic_inputs():
    rng = __import__("numpy").random.default_rng(0)
    noise = lambda: rng.normal(0, 0.002, 25)

    def variants(raw, iso):
        mk = lambda f: {"cv": {"brier": {"mean": float(f.mean()), "folds": list(f)}}}
        return {"raw": mk(raw), "sigmoid": mk(raw), "isotonic": mk(iso)}
    raw = 0.2 + noise()
    assert choose_variant(variants(raw, raw - 0.003))[0] == "isotonic"          # consistent improvement
    assert choose_variant(variants(raw, raw - 0.003 + rng.normal(0, 0.02, 25)))[0] == "raw"  # noisy: not significant

    def entry(auc, brier):
        return {"selected": "raw", "raw": {"cv": {"roc_auc": {"folds": list(auc)}, "brier": {"folds": list(brier)}}}}
    auc, brier = 0.79 + noise(), 0.19 + noise()
    same = {"logistic_regression": entry(auc, brier), "random_forest": entry(auc + noise() / 10, brier),
            "xgboost": entry(auc + noise() / 10, brier)}
    assert select_final(same)["model"] == "logistic_regression"                 # no significant gain: keep
    only_auc = {**same, "xgboost": entry(auc + 0.01, brier + 0.01)}
    assert select_final(only_auc)["model"] == "logistic_regression"             # better ranking, worse Brier
    both = {**same, "xgboost": entry(auc + 0.01, brier - 0.005)}
    assert select_final(both)["model"] == "xgboost"


def test_final_model_matches_report(bundle, report):
    assert (bundle["model_name"], bundle["calibration"]) == (report["selection"]["model"], report["selection"]["variant"])
    _, X_test, _, y_test = split(load_dataset())
    p = bundle["pipeline"].predict_proba(X_test)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert full_metrics(y_test, p)["brier"] == pytest.approx(report["comparison"][bundle["model_name"]]["test"]["brier"])


def test_analysis_artifact_structure(analysis, report):
    assert (analysis["final_model"], analysis["calibration"]) == (report["selection"]["model"], report["selection"]["variant"])
    assert set(analysis["models"]) == set(MODEL_NAMES)
    assert [r["threshold"] for r in analysis["thresholds"]["table"]] == TABLE_THRESHOLDS
    assert {r["group"] for r in analysis["subgroups"]["age_group"]} == set(AGE_LABELS)
    assert sum(r["n"] for r in analysis["subgroups"]["gender"]) == analysis["test_rows"]
    assert {r["feature"] for r in analysis["shap_global"]} == set(MODEL_FEATURES)
    assert {r["feature"] for r in analysis["permutation_importance"]} == set(FEATURES)
    assert analysis["bootstrap"]["seed"] == analysis["seed"] and analysis["bootstrap"]["n_resamples"] == 1000


def test_analysis_numbers_are_consistent(analysis, bundle):
    _, X_test, _, y_test = split(load_dataset())
    assert analysis["test_rows"] == len(X_test)
    # bootstrap point estimates are the plain test metrics of the saved models
    final = analysis["final_model"]
    final_test = analysis["models"][final]["test"]
    assert analysis["bootstrap"]["models"][final]["roc_auc"]["estimate"] == pytest.approx(final_test["roc_auc"])
    cm = analysis["models"][final]["confusion_matrix"]
    half = next(r for r in analysis["thresholds"]["table"] if r["threshold"] == 0.5)
    assert cm == [[half["tn"], half["fp"]], [half["fn"], half["tp"]]]
    for r in analysis["thresholds"]["grid"]:
        assert r["tp"] + r["fp"] + r["tn"] + r["fn"] == len(X_test)
    dq = analysis["data_quality"]
    assert dq["raw_rows"] - dq["clean_rows"] == dq["removed_rows"] == sum(dq["removed_by_rule"].values())
    for f, resp in analysis["feature_response"].items():
        for m in resp["models"].values():
            assert len(m["average"]) == len(resp["grid"])
            assert all(0 <= v <= 1 for v in m["average"] if not math.isnan(v))


def test_permutation_importance_uses_held_out_rows_only(analysis):
    # Computed on the test split with ROC-AUC scoring; the top feature must also lead global SHAP.
    assert analysis["permutation_importance"][0]["feature"] == analysis["shap_global"][0]["feature"] == "ap_hi"
