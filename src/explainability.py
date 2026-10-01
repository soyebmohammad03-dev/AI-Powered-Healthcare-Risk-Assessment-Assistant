"""SHAP explanations for the persisted Logistic Regression pipeline.

Run `python -m src.explainability` to explain the demo inputs and print global importance.

Explanation space: SHAP values are in LOG-ODDS (the classifier's decision_function), not probability.
    base_value + sum(contribution.shap_value) == model_output (log-odds)
    sigmoid(model_output) == PredictionResult.probability_positive
base_value is the model's log-odds at the average (preprocessed) training record, the reference
point every contribution is measured from. It is not the average predicted probability.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_dataset, split
from src.prediction import ModelArtifactError, PatientInput, default_model
from src.preprocessing import CATEGORICAL, CATEGORY_LABELS, FEATURE_LABELS, FEATURES, NUMERIC

POSITIVE = "toward_positive"   # pushes toward target 1 (dataset label: disease present)
NEGATIVE = "toward_negative"
NEUTRAL = "neutral"            # exactly zero: the input equals the background mean


@dataclass(frozen=True)
class FeatureContribution:
    feature: str                # internal name, e.g. "chol"
    label: str                  # readable name, e.g. "Cholesterol"
    value: float                # the patient's original (untransformed) input
    display_value: str          # e.g. "245" or "Asymptomatic"
    shap_value: float           # log-odds contribution; sum of the feature's encoded columns
    direction: str              # POSITIVE, NEGATIVE or NEUTRAL
    relative_importance: float  # |shap_value| / sum of all |shap_value|; sums to 1 across features
    text: str                   # careful, non-causal wording


@dataclass(frozen=True)
class LocalExplanation:
    base_value: float           # log-odds reference point (see module docstring)
    model_output: float         # log-odds for this patient
    probability_positive: float
    contributions: list[FeatureContribution]  # sorted by |shap_value|, largest first


@dataclass(frozen=True)
class GlobalImportance:
    feature: str
    label: str
    mean_abs_shap: float        # mean |log-odds contribution| over the dataset
    relative_importance: float  # share of the total; sums to 1 across features


def _column_owners(pipeline) -> list[str]:
    """Original feature that owns each column after preprocessing (in output order).

    Numeric features map 1:1. A categorical feature owns one column per one-hot category,
    minus the dropped column for binary features (drop="if_binary").
    """
    pre = pipeline.named_steps["pre"]
    ohe = pre.named_transformers_["cat"][-1]
    owners = list(NUMERIC)
    for feature, cats, dropped in zip(CATEGORICAL, ohe.categories_, ohe.drop_idx_):
        owners += [feature] * (len(cats) - (dropped is not None))
    names = pre.get_feature_names_out()
    if len(owners) != len(names) or not all(
            n == f"num__{o}" or n.startswith(f"cat__{o}_") for n, o in zip(names, owners)):
        raise ModelArtifactError("Preprocessed columns do not match the expected feature layout.")
    return owners


def _display(feature: str, value: float) -> str:
    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature][int(value)]
    return f"{value:g}"


class ModelExplainer:
    """Wraps shap.LinearExplainer for the final pipeline's classifier.

    The explainer works on the preprocessed columns (the classifier's real inputs), using the
    preprocessed training split as background with an Independent masker. For a linear model this
    gives exact SHAP values: coef_j * (x_j - mean_j). Contributions of a categorical feature's
    one-hot columns are then summed back to that one feature (valid because SHAP values are additive).
    """

    def __init__(self, bundle: dict | None = None):
        self.pipeline = (bundle or default_model())["pipeline"]
        self.pre = self.pipeline.named_steps["pre"]
        clf = self.pipeline.named_steps["clf"]
        if not isinstance(clf, LogisticRegression):
            raise ModelArtifactError(f"LinearExplainer needs LogisticRegression, got {type(clf).__name__}.")
        self.owners = _column_owners(self.pipeline)
        X_train = split(load_dataset())[0]
        background = self.pre.transform(X_train)
        # max_samples=all rows: the default (100) would subsample the background.
        self._shap = shap.LinearExplainer(clf, shap.maskers.Independent(background, max_samples=len(background)))
        self.base_value = float(np.ravel(self._shap.expected_value)[0])

    def _grouped_shap(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values per original feature: shape (rows, len(FEATURES)), columns in FEATURES order."""
        values = np.asarray(self._shap.shap_values(self.pre.transform(X)))
        grouped = np.zeros((values.shape[0], len(FEATURES)))
        for col, owner in enumerate(self.owners):
            grouped[:, FEATURES.index(owner)] += values[:, col]
        return grouped

    def explain(self, patient: PatientInput) -> LocalExplanation:
        if not isinstance(patient, PatientInput):
            raise TypeError("explain() expects a validated PatientInput.")
        X = patient.to_frame()
        shap_row = self._grouped_shap(X)[0]
        total = float(np.abs(shap_row).sum()) or 1.0
        contributions = []
        for feature, s in zip(FEATURES, shap_row):
            label, value = FEATURE_LABELS[feature], float(getattr(patient, feature))
            direction = POSITIVE if s > 0 else NEGATIVE if s < 0 else NEUTRAL
            display = _display(feature, value)
            shown = f"{label} ({display})"
            text = (f"{shown} did not shift the model's output for this input." if s == 0 else
                    f"{shown} contributed toward a {'higher' if s > 0 else 'lower'} model-estimated "
                    f"probability for this input.")
            contributions.append(FeatureContribution(
                feature=feature, label=label, value=value, display_value=display,
                shap_value=float(s), direction=direction, relative_importance=abs(float(s)) / total,
                text=text,
            ))
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
        output = self.base_value + float(shap_row.sum())
        return LocalExplanation(base_value=self.base_value, model_output=output,
                                probability_positive=1 / (1 + math.exp(-output)),
                                contributions=contributions)

    def global_importance(self, X: pd.DataFrame | None = None) -> list[GlobalImportance]:
        """Mean |SHAP| per feature, ranked. Defaults to all 303 dataset records."""
        X = load_dataset()[FEATURES] if X is None else X
        mean_abs = np.abs(self._grouped_shap(X)).mean(axis=0)
        ranked = sorted(zip(FEATURES, mean_abs), key=lambda t: t[1], reverse=True)
        return [GlobalImportance(f, FEATURE_LABELS[f], float(v), float(v / mean_abs.sum())) for f, v in ranked]


@lru_cache(maxsize=1)
def default_explainer() -> ModelExplainer:
    return ModelExplainer()


def explain(patient: PatientInput) -> LocalExplanation:
    return default_explainer().explain(patient)


if __name__ == "__main__":
    from src.prediction import DEMO_INPUTS, predict

    explainer = default_explainer()
    print(f"Base value (log-odds): {explainer.base_value:+.4f}  "
          f"(probability {1 / (1 + math.exp(-explainer.base_value)):.4f})")
    for name, patient in DEMO_INPUTS.items():
        e, p = explainer.explain(patient), predict(patient)
        print(f"\n{name}: P(pos) predict={p.probability_positive:.4f} shap={e.probability_positive:.4f} "
              f"log-odds={e.model_output:+.3f}")
        for c in e.contributions[:5]:
            print(f"  {c.label:<40}{c.display_value:>24}  {c.shap_value:+.3f}  {c.relative_importance:5.1%}")
    print("\nGlobal importance (mean |SHAP|, log-odds, all 303 records):")
    for i, g in enumerate(explainer.global_importance(), 1):
        print(f"  {i:>2}. {g.label:<40}{g.mean_abs_shap:.3f}  {g.relative_importance:5.1%}")
