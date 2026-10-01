"""Train and compare Logistic Regression, Random Forest and XGBoost, each raw and calibrated, with
5x5 repeated stratified CV, and choose the final model with the pre-declared protocol below.

Run: python -m src.train_models   (several minutes; then src.analysis, src.reliability, src.shift_analysis)
"""
import json

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import ROOT, SEED, SOURCE, TARGET, clean, load_raw, provenance, split
from src.evaluate_models import (CV_SCORING, calibration_data, calibration_stats, corrected_resampled_ttest,
                                 evaluate, fold_assignment, full_metrics, oof_cross_validate, summarize_folds)
from src.prediction import FINAL_MODEL_PATH
from src.preprocessing import CATEGORICAL, CATEGORIES, FEATURES, MODEL_FEATURES, NUMERIC, build_preprocessor

MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
OOF_PATH = ARTIFACTS_DIR / "oof_predictions.npz"  # generated, not committed (about 10 MB)
VARIANTS = ["raw", "sigmoid", "isotonic"]  # raw probabilities, Platt scaling, isotonic regression
N_FOLDS, N_REPEATS = 5, 5

# ---- Model-selection protocol (declared in Phase 6 BEFORE the repeated-CV comparison was run) -----------
# It replaces the previous margins (0.02 ROC-AUC, 0.001 Brier; commit da15801), which were set after the
# first comparison was known. There are no tunable margins: every comparison is a corrected resampled t-test on the SAME 25
# repeated-CV splits of the training data, at the conventional ALPHA. The test set is never used.
#   1. Calibration (per model): keep raw probabilities unless a calibrator lowers CV Brier significantly;
#      if both do, take the lower mean Brier.
#   2. Explanation-quality gate: a model is eligible only if an exact, additive SHAP explanation of its
#      internal score exists and the score maps to the displayed probability monotonically. (All three pass.)
#   3. Model: start from the most interpretable model and move to the next one only if it is significantly
#      better on BOTH discrimination (ROC-AUC, higher) AND probability quality (Brier, lower). The test
#      accounts for split-to-split variability, so a model that wins only in some splits does not qualify;
#      split stability (std, min, max) is reported alongside.
# The author had seen the earlier single 5-fold results; the protocol is therefore not blind, but it
# contains no number that can be set to favour a particular model.
ALPHA = 0.05
INTERPRETABILITY_ORDER = ["logistic_regression", "random_forest", "xgboost"]  # most to least transparent
EXPLAINABLE = {"logistic_regression": "exact linear SHAP on log-odds",
               "random_forest": "exact TreeSHAP on probability",
               "xgboost": "exact TreeSHAP on log-odds"}
TEST_FRACTION = 1 / N_FOLDS


def build_models() -> dict:
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20,
                                                random_state=SEED, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                 eval_metric="logloss"),
    }
    return {name: Pipeline([("pre", build_preprocessor()), ("clf", clf)]) for name, clf in classifiers.items()}


def build_variant(pipeline: Pipeline, variant: str) -> Pipeline:
    """raw, or Pipeline[pre -> CalibratedClassifierCV(classifier)].

    ensemble=False: the classifier is refit on all rows it is given, and ONE calibrator is learned from
    3-fold out-of-fold scores of those rows. Inside oof_cross_validate that means only the outer fold's
    training rows, so the calibrator never sees validation (or test) labels.
    """
    if variant == "raw":
        return clone(pipeline)
    return Pipeline([("pre", clone(pipeline.named_steps["pre"])),
                     ("clf", CalibratedClassifierCV(clone(pipeline.named_steps["clf"]), method=variant,
                                                    cv=3, ensemble=False))])


def choose_variant(variants: dict) -> tuple[str, str, dict]:
    """Protocol step 1. Returns (variant, reason, tests)."""
    raw = variants["raw"]["cv"]["brier"]["folds"]
    tests = {v: corrected_resampled_ttest(variants[v]["cv"]["brier"]["folds"], raw, TEST_FRACTION)
             for v in ("sigmoid", "isotonic")}
    better = [v for v, t in tests.items() if t["mean_difference"] < 0 and t["p_value"] < ALPHA]
    describe = "; ".join(f"{v} {t['mean_difference']:+.4f} (p={t['p_value']:.2g})" for v, t in tests.items())
    if better:
        best = min(better, key=lambda v: variants[v]["cv"]["brier"]["mean"])
        return best, f"{best} calibration significantly lowered CV Brier vs raw [{describe}].", tests
    return "raw", f"No calibrator significantly lowered CV Brier vs raw [{describe}]; raw kept.", tests


def select_final(comparison: dict) -> dict:
    """Protocol steps 2-3: walk from most to least interpretable; switch only for a significant gain in both
    CV ROC-AUC and CV Brier over the current choice (each model in its selected variant)."""
    eligible = [m for m in INTERPRETABILITY_ORDER if m in EXPLAINABLE]
    chosen, steps, tests = eligible[0], [], {}
    for other in eligible[1:]:
        a, b = comparison[chosen][comparison[chosen]["selected"]]["cv"], comparison[other][comparison[other]["selected"]]["cv"]
        auc = corrected_resampled_ttest(b["roc_auc"]["folds"], a["roc_auc"]["folds"], TEST_FRACTION)
        brier = corrected_resampled_ttest(b["brier"]["folds"], a["brier"]["folds"], TEST_FRACTION)
        switch = (auc["mean_difference"] > 0 and auc["p_value"] < ALPHA
                  and brier["mean_difference"] < 0 and brier["p_value"] < ALPHA)
        tests[f"{other}_vs_{chosen}"] = {"roc_auc": auc, "brier": brier}
        steps.append(f"{other} vs {chosen}: CV ROC-AUC {auc['mean_difference']:+.4f} (p={auc['p_value']:.2g}), "
                     f"CV Brier {brier['mean_difference']:+.4f} (p={brier['p_value']:.2g}) -> "
                     f"{'switch to ' + other if switch else 'keep ' + chosen}.")
        chosen = other if switch else chosen
    return {"model": chosen, "variant": comparison[chosen]["selected"], "steps": steps, "tests": tests}


def dataset_summary(raw, df, removed) -> dict:
    return {
        "source": SOURCE,
        "raw_records": len(raw),
        "clean_records": len(df),
        "removed_by_rule": removed,
        "input_features": FEATURES,
        "model_features": MODEL_FEATURES,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "target": f"{TARGET}: 1 = cardiovascular disease present, 0 = absent",
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
    }


def main() -> dict:
    raw = load_raw()
    df, removed = clean(raw)
    X_train, X_test, y_train, y_test = split(df)  # X_test is touched only after selection, at the end
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)

    MODELS_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    models = build_models()
    comparison, oof_store = {}, {}
    for name, pipeline in models.items():
        comparison[name] = {}
        for variant in VARIANTS:
            folds, oof = oof_cross_validate(build_variant(pipeline, variant), X_train, y_train, cv)
            per_repeat = [calibration_stats(y_train, o) for o in oof]
            comparison[name][variant] = {
                "cv": summarize_folds(folds),
                "oof_calibration": calibration_data(np.tile(y_train, N_REPEATS), oof.ravel()),
                "oof_calibration_stats": {k: {"mean": float(np.mean([r[k] for r in per_repeat])),
                                              "std": float(np.std([r[k] for r in per_repeat]))}
                                          for k in per_repeat[0]},
            }
            oof_store[f"{name}__{variant}"] = oof.astype(np.float32)
            print(f"  {name:<20} {variant:<9} CV ROC-AUC {comparison[name][variant]['cv']['roc_auc']['mean']:.4f} "
                  f"Brier {comparison[name][variant]['cv']['brier']['mean']:.4f}", flush=True)
        selected, reason, tests = choose_variant(comparison[name])
        comparison[name].update(selected=selected, calibration_reason=reason, calibration_tests=tests)
    np.savez_compressed(OOF_PATH, index=X_train.index.to_numpy(), y=y_train.to_numpy(),
                        fold=fold_assignment(cv, X_train, y_train), **oof_store)

    selection = select_final(comparison)
    final = selection["model"]

    # Selection is complete. Only now: fit each model's selected variant on the whole training split and
    # score it once on the held-out test set (reported, never used to choose anything).
    results = {}
    for name, pipeline in models.items():
        fitted = build_variant(pipeline, comparison[name]["selected"]).fit(X_train, y_train)
        models[name] = fitted
        comparison[name]["test"] = full_metrics(y_test, fitted.predict_proba(X_test)[:, 1])
        cv_sel = comparison[name][comparison[name]["selected"]]["cv"]
        results[name] = {"cv": {m: {k: cv_sel[m][k] for k in ("mean", "std", "min", "max")} for m in cv_sel},
                         "test": evaluate(fitted, X_test, y_test)}
        joblib.dump(fitted, MODELS_DIR / f"{name}.joblib")

    # The final artifact is the full fitted Pipeline (preprocessing + classifier), so inference
    # cannot apply different preprocessing than training. It is the exact model scored on the test set.
    joblib.dump({
        "pipeline": models[final],
        "model_name": final,
        "features": FEATURES,
        "model_features": MODEL_FEATURES,
        "categories": CATEGORIES,
        "seed": SEED,
        "test_metrics": results[final]["test"],
        "calibration": selection["variant"],
        "calibration_reason": comparison[final]["calibration_reason"],
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "provenance": provenance(),
    }, FINAL_MODEL_PATH)

    report = {"provenance": provenance(), "seed": SEED, "split": {"train": len(X_train), "test": len(X_test)},
              "cv": {"scheme": "RepeatedStratifiedKFold", "n_splits": N_FOLDS, "n_repeats": N_REPEATS,
                     "random_state": SEED},
              "dataset": dataset_summary(raw, df, removed), "final_model": final,
              "final_model_reason": " ".join(selection["steps"]) + " " + comparison[final]["calibration_reason"],
              "selection": {**selection, "alpha": ALPHA, "test": "corrected resampled t-test (Nadeau & Bengio)",
                            "interpretability_order": INTERPRETABILITY_ORDER, "explanation_gate": EXPLAINABLE},
              "models": results, "comparison": comparison}
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = main()
    print(f"Final model: {report['final_model']} -> {FINAL_MODEL_PATH}")
    print(f"Train {report['split']['train']} / Test {report['split']['test']}")
    print(f"{'model':<22}{'split':<7}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'auc':>7}")
    for step in report["selection"]["steps"]:
        print(step)
    for name, r in report["models"].items():
        cv = {m: v["mean"] for m, v in r["cv"].items()}
        for split, m in (("cv", cv), ("test", r["test"])):
            print(f"{name:<22}{split:<7}" + "".join(f"{m[k]:>7.3f}" for k in CV_SCORING))
