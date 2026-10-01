"""SHAP explanations for the persisted final pipeline, whichever candidate the selection protocol chose.

Run `python -m src.explainability` to explain the demo inputs and print global importance.

Two spaces, kept separate on purpose:
  * SCORE: the classifier's internal output. Logistic Regression and XGBoost: log-odds; Random Forest:
    probability (mean of the trees). SHAP is exact and additive here:
        base_value + sum(contribution.shap_value) == model_output
    (LinearExplainer for Logistic Regression; TreeSHAP, tree_path_dependent, for the tree models.)
  * PROBABILITY shown to users = link(score) (sigmoid for log-odds), then the calibrator if the pipeline
    has one. Every step is monotone non-decreasing. It equals PredictionResult.probability_positive.
    After a non-linear link or calibration, contributions are NOT additive in probability space; they keep
    their direction, not their size.
base_value: Logistic Regression, the score at the average (preprocessed) training record; tree models, the
average score over the training data. base_probability is its displayed probability.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data_loader import load_dataset, split
from src.prediction import ModelArtifactError, PatientInput, default_model
from src.preprocessing import CATEGORICAL, CATEGORY_LABELS, FEATURE_LABELS, FEATURES, MODEL_FEATURES, NUMERIC

POSITIVE = "toward_positive"   # pushes toward target 1 (dataset label: disease present)
NEGATIVE = "toward_negative"
NEUTRAL = "neutral"            # exactly zero: the input equals the background mean


@dataclass(frozen=True)
class FeatureContribution:
    feature: str                # internal name, e.g. "ap_hi" (one of MODEL_FEATURES)
    label: str                  # readable name, e.g. "Systolic blood pressure"
    value: float                # the patient's untransformed value (BMI: derived from height/weight)
    display_value: str          # e.g. "140", "27.8" or "Above normal"
    shap_value: float           # contribution to the score; sum of the feature's encoded columns
    direction: str              # POSITIVE, NEGATIVE or NEUTRAL
    relative_importance: float  # |shap_value| / sum of all |shap_value|; sums to 1 across features
    text: str                   # careful, non-causal wording


@dataclass(frozen=True)
class LocalExplanation:
    base_value: float           # reference score (see module docstring)
    model_output: float         # score for this input = base_value + sum of contributions
    base_probability: float     # displayed probability at base_value (the reference shown to users)
    probability_positive: float # displayed probability for this input (== prediction service)
    uncalibrated_probability: float  # link(model_output), before calibration; for transparency
    contributions: list[FeatureContribution]  # sorted by |shap_value|, largest first


@dataclass(frozen=True)
class GlobalImportance:
    feature: str
    label: str
    mean_abs_shap: float        # mean |score contribution| over the dataset
    relative_importance: float  # share of the total; sums to 1 across features


def model_parts(pipeline) -> tuple[object, object | None]:
    """(classifier, calibrator or None) inside the persisted pipeline."""
    clf = pipeline.named_steps["clf"]
    calibrator = None
    if isinstance(clf, CalibratedClassifierCV):
        if len(clf.calibrated_classifiers_) != 1:
            raise ModelArtifactError("Expected a single calibrated classifier (ensemble=False).")
        inner = clf.calibrated_classifiers_[0]
        clf, calibrator = inner.estimator, inner.calibrators[0]
    if not isinstance(clf, (LogisticRegression, RandomForestClassifier, XGBClassifier)):
        raise ModelArtifactError(f"No exact SHAP explainer for {type(clf).__name__}; supported: "
                                 "LogisticRegression, RandomForestClassifier, XGBClassifier.")
    return clf, calibrator


def _column_owners(pipeline) -> list[str]:
    """Original feature that owns each column after preprocessing (in output order).

    Numeric features map 1:1. A categorical feature owns one column per one-hot category,
    minus the dropped column for binary features (drop="if_binary").
    """
    encoder = pipeline.named_steps["pre"][-1]  # the ColumnTransformer after BMI derivation
    ohe = encoder.named_transformers_["cat"]
    owners = list(NUMERIC)
    for feature, cats, dropped in zip(CATEGORICAL, ohe.categories_, ohe.drop_idx_):
        owners += [feature] * (len(cats) - (dropped is not None))
    names = encoder.get_feature_names_out()
    if len(owners) != len(names) or not all(
            n == f"num__{o}" or n.startswith(f"cat__{o}_") for n, o in zip(names, owners)):
        raise ModelArtifactError("Preprocessed columns do not match the expected feature layout.")
    return owners


def _display(feature: str, value: float) -> str:
    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature][int(value)]
    return f"{value:.1f}" if feature == "bmi" else f"{value:g}"


class ModelExplainer:
    """Exact SHAP for the classifier inside the final pipeline.

    The explainer works on the preprocessed columns (the classifier's real inputs: BMI derived,
    numerics scaled, categoricals one-hot encoded; height and weight enter only through BMI).
    Logistic Regression: LinearExplainer with the preprocessed training split as background (Independent
    masker), so SHAP_j = coef_j * (x_j - mean_j). Tree models: TreeSHAP (tree_path_dependent), exact for
    the trees' own output. Contributions of a categorical feature's one-hot columns are then summed back to
    that one feature (valid because SHAP values are additive).
    """

    def __init__(self, bundle: dict | None = None):
        self.pipeline = (bundle or default_model())["pipeline"]
        self.pre = self.pipeline.named_steps["pre"]
        self.clf, self.calibrator = model_parts(self.pipeline)
        self.owners = _column_owners(self.pipeline)
        self.linear = isinstance(self.clf, LogisticRegression)
        self.log_odds = not isinstance(self.clf, RandomForestClassifier)  # RF trees average probabilities
        if self.linear:
            background = self.pre.transform(split(load_dataset())[0])
            # max_samples=all rows: the default (100) would subsample the background.
            self._shap = shap.LinearExplainer(self.clf, shap.maskers.Independent(background, max_samples=len(background)))
        else:
            self._shap = shap.TreeExplainer(self.clf)
        self.base_value = float(np.ravel(self._shap.expected_value)[-1])  # RF: one per class; class 1 last
        if not self.linear:
            # shap 0.52 reports XGBoost's expected value about 6e-4 log-odds away from the model's own base
            # margin (a constant, same for every row). Re-anchor it so base + contributions == score exactly.
            Z = np.vstack([np.zeros(len(self.owners)), np.ones(len(self.owners))])
            gap = self.model_score(Z) - self._raw_shap(Z).sum(axis=1) - self.base_value
            if np.ptp(gap) > 1e-5:
                raise ModelArtifactError("TreeSHAP offset is not constant; explanations would not reconcile.")
            self.base_value += float(gap.mean())
        self.base_probability = self.score_to_probability(self.base_value)

    def link(self, score: float) -> float:
        return 1 / (1 + math.exp(-score)) if self.log_odds else score

    def model_score(self, Z: np.ndarray) -> np.ndarray:
        """The classifier's own score for preprocessed rows Z (what SHAP explains), for reconciliation."""
        if self.linear:
            return self.clf.decision_function(Z)
        if isinstance(self.clf, XGBClassifier):
            return self.clf.predict(Z, output_margin=True)
        return self.clf.predict_proba(Z)[:, 1]

    def _raw_shap(self, Z: np.ndarray) -> np.ndarray:
        values = np.asarray(self._shap.shap_values(Z), dtype=float)
        return values[..., 1] if values.ndim == 3 else values  # RF: (rows, columns, classes)

    def score_to_probability(self, score: float) -> float:
        if self.calibrator is None:
            return float(self.link(score))
        # CalibratedClassifierCV calibrates decision_function when the classifier has one, else predict_proba.
        x = score if hasattr(self.clf, "decision_function") else self.link(score)
        return float(np.clip(self.calibrator.predict(np.array([x]))[0], 0.0, 1.0))

    def grouped_shap(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values per model feature: shape (rows, len(MODEL_FEATURES)), in MODEL_FEATURES order.
        X holds raw FEATURES; the persisted preprocessing derives BMI and encodes."""
        values = self._raw_shap(self.pre.transform(X[FEATURES]))
        grouped = np.zeros((values.shape[0], len(MODEL_FEATURES)))
        for col, owner in enumerate(self.owners):
            grouped[:, MODEL_FEATURES.index(owner)] += values[:, col]
        return grouped

    def explain(self, patient: PatientInput) -> LocalExplanation:
        if not isinstance(patient, PatientInput):
            raise TypeError("explain() expects a validated PatientInput.")
        X = patient.to_frame()
        shap_row = self.grouped_shap(X)[0]
        total = float(np.abs(shap_row).sum()) or 1.0
        contributions = []
        for feature, s in zip(MODEL_FEATURES, shap_row):
            label, value = FEATURE_LABELS[feature], float(getattr(patient, feature))  # bmi: PatientInput.bmi
            direction = POSITIVE if s > 0 else NEGATIVE if s < 0 else NEUTRAL
            display = _display(feature, value)
            shown = f"{label} ({display})"
            text = (f"{shown} did not shift the model's score for this input." if s == 0 else
                    f"{shown} moved the model estimate {'higher' if s > 0 else 'lower'} for this input.")
            contributions.append(FeatureContribution(
                feature=feature, label=label, value=value, display_value=display,
                shap_value=float(s), direction=direction, relative_importance=abs(float(s)) / total,
                text=text,
            ))
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
        output = self.base_value + float(shap_row.sum())
        return LocalExplanation(
            base_value=self.base_value, model_output=output, base_probability=self.base_probability,
            # taken from the pipeline itself, so it is identical to the prediction service by construction
            probability_positive=float(self.pipeline.predict_proba(X)[0, 1]),
            uncalibrated_probability=self.link(output),
            contributions=contributions)

    def global_importance(self, X: pd.DataFrame | None = None) -> list[GlobalImportance]:
        """Mean |SHAP| per model feature, ranked. Defaults to every record of the cleaned dataset."""
        X = load_dataset() if X is None else X
        mean_abs = np.abs(self.grouped_shap(X)).mean(axis=0)
        ranked = sorted(zip(MODEL_FEATURES, mean_abs), key=lambda t: t[1], reverse=True)
        return [GlobalImportance(f, FEATURE_LABELS[f], float(v), float(v / mean_abs.sum())) for f, v in ranked]


@lru_cache(maxsize=1)
def default_explainer() -> ModelExplainer:
    return ModelExplainer()


def explain(patient: PatientInput) -> LocalExplanation:
    return default_explainer().explain(patient)


if __name__ == "__main__":
    from src.prediction import DEMO_INPUTS, predict

    explainer = default_explainer()
    print(f"Base score: {explainer.base_value:+.4f}  (displayed probability {explainer.base_probability:.4f})")
    for name, patient in DEMO_INPUTS.items():
        e, p = explainer.explain(patient), predict(patient)
        print(f"\n{name}: P(pos) predict={p.probability_positive:.4f} explain={e.probability_positive:.4f} "
              f"uncalibrated={e.uncalibrated_probability:.4f} score={e.model_output:+.3f}")
        for c in e.contributions[:5]:
            print(f"  {c.label:<32}{c.display_value:>18}  {c.shap_value:+.3f}  {c.relative_importance:5.1%}")
    print("\nGlobal importance (mean |SHAP|, score units, all cleaned records):")
    for i, g in enumerate(explainer.global_importance(), 1):
        print(f"  {i:>2}. {g.label:<32}{g.mean_abs_shap:.3f}  {g.relative_importance:5.1%}")
