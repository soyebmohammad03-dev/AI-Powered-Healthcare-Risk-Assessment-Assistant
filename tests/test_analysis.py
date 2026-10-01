"""Model selection, calibration and the precomputed analysis artifact (real data, real models)."""
import json
import math

import pytest

from src.analysis import AGE_LABELS, ARTIFACT, MODEL_NAMES, TABLE_THRESHOLDS
from src.data_loader import ROOT, load_dataset, split
from src.evaluate_models import full_metrics
from src.prediction import FINAL_MODEL_PATH, load_model
from src.preprocessing import FEATURES, MODEL_FEATURES
from src.train_models import AUC_MARGIN, BRIER_MARGIN, choose_variant, select_final


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
    for name in MODEL_NAMES:
        for variant in ("raw", "sigmoid", "isotonic"):
            entry = report["comparison"][name][variant]
            assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss", "brier"} <= set(entry["cv"])
            assert len(entry["cv"]["roc_auc"]["folds"]) == 5
            assert all(0 <= b["mean_predicted"] <= 1 for b in entry["oof_calibration"])


def test_selection_framework_reproduces_recorded_decision(report):
    comparison = report["comparison"]
    for name in MODEL_NAMES:
        assert choose_variant(comparison[name])[0] == comparison[name]["selected"]
    selection = select_final(comparison)
    assert (selection["model"], selection["variant"]) == (report["selection"]["model"], report["selection"]["variant"])
    assert selection == {k: report["selection"][k] for k in ("model", "variant", "steps")}


def test_selection_rules_on_synthetic_inputs():
    def variants(raw_folds, iso_folds):
        mk = lambda f: {"cv": {"brier": {"mean": sum(f) / len(f), "folds": f}}}
        return {"raw": mk(raw_folds), "sigmoid": mk(raw_folds), "isotonic": mk(iso_folds)}
    better = [x - 2 * BRIER_MARGIN for x in [0.2] * 5]
    assert choose_variant(variants([0.2] * 5, better))[0] == "isotonic"
    mixed = better[:4] + [0.21]  # better on average but not in every fold
    assert choose_variant(variants([0.2] * 5, mixed))[0] == "raw"

    def entry(auc, brier):
        return {"selected": "raw", "raw": {"cv": {"roc_auc": {"mean": auc}, "brier": {"mean": brier}}}}
    small_gain = {"logistic_regression": entry(0.79, 0.19), "random_forest": entry(0.79 + AUC_MARGIN / 2, 0.18),
                  "xgboost": entry(0.79 + AUC_MARGIN / 2, 0.18)}
    assert select_final(small_gain)["model"] == "logistic_regression"
    big_gain = {**small_gain, "xgboost": entry(0.79 + AUC_MARGIN * 2, 0.18)}
    assert select_final(big_gain)["model"] == "xgboost"


def test_final_model_is_calibrated_and_matches_report(bundle, report):
    assert bundle["calibration"] == report["selection"]["variant"] == "isotonic"
    _, X_test, _, y_test = split(load_dataset())
    p = bundle["pipeline"].predict_proba(X_test)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert full_metrics(y_test, p)["brier"] == pytest.approx(
        report["comparison"]["logistic_regression"]["isotonic"]["test"]["brier"])


def test_analysis_artifact_structure(analysis):
    assert analysis["final_model"] == "logistic_regression" and analysis["calibration"] == "isotonic"
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
    lr_test = analysis["models"]["logistic_regression"]["test"]
    assert analysis["bootstrap"]["models"]["logistic_regression"]["roc_auc"]["estimate"] == pytest.approx(lr_test["roc_auc"])
    cm = analysis["models"]["logistic_regression"]["confusion_matrix"]
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
