"""Train and evaluate Logistic Regression, Random Forest and XGBoost.

Run: python -m src.train_models
"""
import json

import joblib
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import ROOT, TARGET, URL, load_dataset
from src.evaluate_models import CV_SCORING, evaluate
from src.prediction import FINAL_MODEL_PATH
from src.preprocessing import CATEGORICAL, CATEGORIES, FEATURES, NUMERIC, build_preprocessor

SEED = 42
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
# Fixed decision (see docs/methodology.md): best 5-fold CV ROC-AUC, accuracy, precision and F1,
# and directly interpretable. Not chosen from the 61-record test set.
FINAL_MODEL = "logistic_regression"


def build_models() -> dict:
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                 eval_metric="logloss"),
    }
    return {name: Pipeline([("pre", build_preprocessor()), ("clf", clf)]) for name, clf in classifiers.items()}


def dataset_summary(df) -> dict:
    return {
        "source": URL,
        "records": len(df),
        "features": FEATURES,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "target": "target = 1 if original 'num' > 0 (disease present), else 0",
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v},
    }


def main() -> dict:
    df = load_dataset()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
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
        "categories": CATEGORIES,
        "seed": SEED,
        "test_metrics": results[FINAL_MODEL]["test"],
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }, FINAL_MODEL_PATH)

    report = {"seed": SEED, "split": {"train": len(X_train), "test": len(X_test)},
              "dataset": dataset_summary(df), "final_model": FINAL_MODEL, "models": results}
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
