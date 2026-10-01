"""Train and compare Logistic Regression, Random Forest and XGBoost, each raw and calibrated.

Run: python -m src.train_models   (then python -m src.analysis for the post-hoc analyses)
"""
import json

import joblib
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import ROOT, SEED, SOURCE, TARGET, clean, load_raw, split
from src.evaluate_models import (CV_SCORING, calibration_data, evaluate, full_metrics, oof_cross_validate,
                                 summarize_folds)
from src.prediction import FINAL_MODEL_PATH
from src.preprocessing import CATEGORICAL, CATEGORIES, FEATURES, MODEL_FEATURES, NUMERIC, build_preprocessor

MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
VARIANTS = ["raw", "sigmoid", "isotonic"]  # raw probabilities, Platt scaling, isotonic regression
FINAL_MODEL = "logistic_regression"  # what the explainer supports; asserted against the framework below
FINAL_VARIANT = "isotonic"
# Selection framework (decided on 5-fold CV of the training split; the test set is never used to choose).
# Margins were fixed in the upgrade phase, after the Phase 1 comparison was known; they are judgement
# calls about practical relevance, stated so they can be challenged, not statistically derived.
INTERPRETABILITY_ORDER = ["logistic_regression", "random_forest", "xgboost"]  # most to least transparent
AUC_MARGIN = 0.02     # a less interpretable model must beat the more interpretable one by this CV ROC-AUC
BRIER_MARGIN = 0.001  # a calibrator must lower mean CV Brier by this much, and in every fold


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


def choose_variant(variants: dict) -> tuple[str, str]:
    raw = variants["raw"]["cv"]["brier"]
    for name in ("sigmoid", "isotonic"):
        cal = variants[name]["cv"]["brier"]
        gain = raw["mean"] - cal["mean"]
        if gain >= BRIER_MARGIN and all(c < r for c, r in zip(cal["folds"], raw["folds"])):
            return name, f"{name} calibration lowered mean CV Brier by {gain:.4f} and in every fold."
    best = min(("sigmoid", "isotonic"), key=lambda v: variants[v]["cv"]["brier"]["mean"])
    change = variants[best]["cv"]["brier"]["mean"] - raw["mean"]
    return "raw", (f"Calibration not material: best calibrator ({best}) changed mean CV Brier by {change:+.4f} "
                   f"(needs <= -{BRIER_MARGIN} and lower in every fold); raw probabilities kept.")


def select_final(comparison: dict) -> dict:
    """Walk from most to least interpretable; move on only for a material CV discrimination gain."""
    chosen = INTERPRETABILITY_ORDER[0]
    steps = []
    for other in INTERPRETABILITY_ORDER[1:]:
        a = comparison[chosen]["selected"]
        b = comparison[other]["selected"]
        auc_gain = comparison[other][b]["cv"]["roc_auc"]["mean"] - comparison[chosen][a]["cv"]["roc_auc"]["mean"]
        brier_change = comparison[other][b]["cv"]["brier"]["mean"] - comparison[chosen][a]["cv"]["brier"]["mean"]
        switch = auc_gain >= AUC_MARGIN and brier_change <= 0
        steps.append(f"{other} vs {chosen}: CV ROC-AUC {auc_gain:+.4f}, CV Brier {brier_change:+.4f} -> "
                     f"{'switch' if switch else 'keep ' + chosen} (margin {AUC_MARGIN}).")
        chosen = other if switch else chosen
    return {"model": chosen, "variant": comparison[chosen]["selected"], "steps": steps}


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
    X_train, X_test, y_train, y_test = split(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    MODELS_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    models = build_models()
    results, comparison = {}, {}
    for name, pipeline in models.items():
        comparison[name] = {}
        for variant in VARIANTS:
            model = build_variant(pipeline, variant)
            folds, oof = oof_cross_validate(model, X_train, y_train, cv)
            fitted = clone(model).fit(X_train, y_train)
            comparison[name][variant] = {
                "cv": summarize_folds(folds),
                "test": full_metrics(y_test, fitted.predict_proba(X_test)[:, 1]),
                "oof_calibration": calibration_data(y_train, oof),
            }
            comparison[name][variant]["_fitted"] = fitted
        selected, comparison[name]["calibration_reason"] = choose_variant(comparison[name])
        comparison[name]["selected"] = selected
        # Each model is kept and reported in its selected variant.
        models[name] = comparison[name][selected]["_fitted"]
        cv_sel = comparison[name][selected]["cv"]
        results[name] = {"cv_train_5fold": {m: {k: cv_sel[m][k] for k in ("mean", "std")} for m in CV_SCORING},
                         "test": evaluate(models[name], X_test, y_test)}
        joblib.dump(models[name], MODELS_DIR / f"{name}.joblib")
        for variant in VARIANTS:
            del comparison[name][variant]["_fitted"]

    selection = select_final(comparison)
    if (selection["model"], selection["variant"]) != (FINAL_MODEL, FINAL_VARIANT):
        # The explainer (exact LinearExplainer on the logistic regression score) assumes this choice.
        raise RuntimeError(f"Selection framework chose {selection}; update FINAL_MODEL and the explainer.")

    # The final artifact is the full fitted Pipeline (preprocessing + classifier), so inference
    # cannot apply different preprocessing than training. It is the exact model scored on the test set.
    joblib.dump({
        "pipeline": models[FINAL_MODEL],
        "model_name": FINAL_MODEL,
        "features": FEATURES,
        "model_features": MODEL_FEATURES,
        "categories": CATEGORIES,
        "seed": SEED,
        "test_metrics": results[FINAL_MODEL]["test"],
        "calibration": FINAL_VARIANT,
        "calibration_reason": comparison[FINAL_MODEL]["calibration_reason"],
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }, FINAL_MODEL_PATH)

    report = {"seed": SEED, "split": {"train": len(X_train), "test": len(X_test)},
              "dataset": dataset_summary(raw, df, removed), "final_model": FINAL_MODEL,
              "final_model_reason": " ".join(selection["steps"]) + " " +
              comparison[FINAL_MODEL]["calibration_reason"],
              "selection": {**selection, "auc_margin": AUC_MARGIN, "brier_margin": BRIER_MARGIN,
                            "interpretability_order": INTERPRETABILITY_ORDER},
              "models": results, "comparison": comparison}
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = main()
    print(f"Final model: {report['final_model']} -> {FINAL_MODEL_PATH}")
    print(f"Train {report['split']['train']} / Test {report['split']['test']}")
    print(f"{'model':<22}{'split':<7}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'auc':>7}")
    for name, r in report["models"].items():
        cv = {m: v["mean"] for m, v in r["cv_train_5fold"].items()}
        for split, m in (("cv", cv), ("test", r["test"])):
            print(f"{name:<22}{split:<7}" + "".join(f"{m[k]:>7.3f}" for k in CV_SCORING))
