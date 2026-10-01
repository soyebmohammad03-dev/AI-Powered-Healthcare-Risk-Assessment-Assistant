"""Train and evaluate Logistic Regression, Random Forest and XGBoost.

Run: python -m src.train_models
"""
import json

import joblib
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import ROOT, SEED, SOURCE, TARGET, clean, load_raw, split
from src.evaluate_models import CV_SCORING, evaluate
from src.prediction import FINAL_MODEL_PATH
from src.preprocessing import CATEGORICAL, CATEGORIES, FEATURES, MODEL_FEATURES, NUMERIC, build_preprocessor

MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
# Selection decision, made on 5-fold CV first (test set only as a consistency check); full table in README.
FINAL_MODEL = "logistic_regression"
SELECTION_REASON = (
    "XGBoost and Random Forest lead by about 0.01 ROC-AUC (CV 0.801 / 0.800 vs 0.791) and under 1 point "
    "of accuracy. All three are equally stable (CV std <= 0.01) and match their test scores. Logistic "
    "Regression is kept: its effects are monotone and global, its SHAP values are exact "
    "(coef x (x - mean)) and easy to explain, which outweighs the small performance gap for an "
    "explainable decision-support prototype."
)


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
    results = {}
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=CV_SCORING)
        cv_metrics = {m: {"mean": scores[f"test_{m}"].mean(), "std": scores[f"test_{m}"].std()} for m in CV_SCORING}
        model.fit(X_train, y_train)
        results[name] = {"cv_train_5fold": cv_metrics, "test": evaluate(model, X_test, y_test)}
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

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
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }, FINAL_MODEL_PATH)

    report = {"seed": SEED, "split": {"train": len(X_train), "test": len(X_test)},
              "dataset": dataset_summary(raw, df, removed), "final_model": FINAL_MODEL,
              "final_model_reason": SELECTION_REASON, "models": results}
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
