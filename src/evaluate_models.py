"""Evaluation layer: metrics, out-of-fold CV, calibration, thresholds, bootstrap, subgroups.

Pure functions of (y_true, probability) wherever possible, so every analysis is testable on small data.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)

CV_SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
              "f1": "f1", "roc_auc": "roc_auc"}
FULL_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss", "brier"]
LOWER_IS_BETTER = {"log_loss", "brier"}


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),  # [[TN, FP], [FN, TP]]
    }


def full_metrics(y, proba, threshold: float = 0.5) -> dict:
    """Threshold metrics at `threshold` plus threshold-free ranking and probability-quality metrics."""
    y, proba = np.asarray(y), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "log_loss": log_loss(y, proba, labels=[0, 1]),
        "brier": brier_score_loss(y, proba),
    }


def oof_cross_validate(model, X: pd.DataFrame, y: pd.Series, cv) -> tuple[list[dict], np.ndarray]:
    """Per-fold full metrics and out-of-fold probabilities.

    Each fold's model (including any calibrator it contains) is fit on that fold's training rows only,
    so every out-of-fold probability comes from a model that never saw the row.
    """
    oof = np.full(len(y), np.nan)
    folds = []
    for train_idx, val_idx in cv.split(X, y):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = fitted.predict_proba(X.iloc[val_idx])[:, 1]
        folds.append(full_metrics(y.iloc[val_idx], oof[val_idx]))
    return folds, oof


def summarize_folds(folds: list[dict]) -> dict:
    return {m: {"mean": float(np.mean([f[m] for f in folds])), "std": float(np.std([f[m] for f in folds])),
                "folds": [float(f[m]) for f in folds]} for m in FULL_METRICS}


def calibration_data(y, proba, bins: int = 10) -> list[dict]:
    """Reliability diagram data on equal-width probability bins (empty bins omitted)."""
    y, proba = np.asarray(y), np.asarray(proba)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    return [{"bin_low": b / bins, "bin_high": (b + 1) / bins, "count": int((idx == b).sum()),
             "mean_predicted": float(proba[idx == b].mean()), "observed_rate": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def threshold_metrics(y, proba, thresholds) -> list[dict]:
    """Confusion counts and rates at each threshold (positive = probability >= threshold)."""
    y, proba = np.asarray(y).astype(bool), np.asarray(proba)
    rows = []
    for t in thresholds:
        pred = proba >= t
        tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
        fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")  # undefined when nothing is predicted positive
        recall = tp / (tp + fn)
        rows.append({
            "threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "specificity": tn / (tn + fp),
            "f1": 2 * tp / (2 * tp + fp + fn), "fpr": fp / (fp + tn), "fnr": fn / (fn + tp),
            "predicted_positive_rate": (tp + fp) / len(y),
        })
    return rows


BOOTSTRAP_METRICS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "f1": lambda y, p: f1_score(y, p >= 0.5),
    "brier": brier_score_loss,
}


def bootstrap_ci(y, probas: dict, n_resamples: int = 1000, seed: int = 42, level: float = 0.95,
                 reference: str | None = None) -> dict:
    """Percentile bootstrap over test rows. All models share the same resamples (paired), so differences
    against `reference` get their own interval. Describes uncertainty from the finite evaluation sample only."""
    y = np.asarray(y)
    probas = {k: np.asarray(v) for k, v in probas.items()}
    rng = np.random.default_rng(seed)
    draws = {k: {m: [] for m in BOOTSTRAP_METRICS} for k in probas}
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        for name, p in probas.items():
            for m, fn in BOOTSTRAP_METRICS.items():
                draws[name][m].append(fn(y[idx], p[idx]))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100

    def interval(point, samples):
        return {"estimate": float(point), "lower": float(np.percentile(samples, lo)),
                "upper": float(np.percentile(samples, hi))}

    out = {"n_resamples": n_resamples, "seed": seed, "level": level, "models": {}, "differences": {}}
    for name, p in probas.items():
        out["models"][name] = {m: interval(fn(y, p), draws[name][m]) for m, fn in BOOTSTRAP_METRICS.items()}
    if reference:
        for name in probas:
            if name != reference:
                out["differences"][name] = {m: interval(
                    BOOTSTRAP_METRICS[m](y, probas[name]) - BOOTSTRAP_METRICS[m](y, probas[reference]),
                    np.subtract(draws[name][m], draws[reference][m])) for m in BOOTSTRAP_METRICS}
    return out


def subgroup_metrics(y, proba, groups, min_per_class: int = 30) -> list[dict]:
    """Metrics per subgroup. Groups with fewer than `min_per_class` rows of either class are reported
    with counts only (`reliable: False`) rather than over-interpreted."""
    y, proba, groups = pd.Series(np.asarray(y)), pd.Series(np.asarray(proba)), pd.Series(np.asarray(groups))
    rows = []
    for g in sorted(groups.unique(), key=str):
        mask = (groups == g).to_numpy()
        yg, pg = y[mask].to_numpy(), proba[mask].to_numpy()
        row = {"group": str(g), "n": int(mask.sum()), "positives": int(yg.sum()),
               "observed_rate": float(yg.mean()), "mean_predicted": float(pg.mean())}
        row["reliable"] = bool(min(yg.sum(), len(yg) - yg.sum()) >= min_per_class)
        if row["reliable"]:
            m = full_metrics(yg, pg)
            row.update({k: m[k] for k in ("roc_auc", "precision", "recall", "f1", "brier")})
        rows.append(row)
    return rows
