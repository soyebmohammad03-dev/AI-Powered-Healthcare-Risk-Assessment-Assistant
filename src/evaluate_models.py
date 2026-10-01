"""Evaluation layer: metrics, (repeated) out-of-fold CV, model comparison test, calibration, thresholds,
decision curves, bootstrap, subgroups.

Pure functions of (y_true, probability) wherever possible, so every analysis is testable on small data.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
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
    """Per-split full metrics and out-of-fold probabilities, shape (n_repeats, n_rows).

    Works for StratifiedKFold (1 repeat) and RepeatedStratifiedKFold. Each split's model (including any
    calibrator it contains) is fit on that split's training rows only, so every out-of-fold probability
    comes from a model that never saw the row. Within one repeat every row is predicted exactly once.
    """
    n_repeats = getattr(cv, "n_repeats", 1)
    n_folds = cv.get_n_splits() // n_repeats
    oof = np.full((n_repeats, len(y)), np.nan)
    folds = []
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        repeat = i // n_folds
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[repeat, val_idx] = fitted.predict_proba(X.iloc[val_idx])[:, 1]
        folds.append({"repeat": repeat, "fold": i % n_folds, **full_metrics(y.iloc[val_idx], oof[repeat, val_idx])})
    return folds, oof


def fold_assignment(cv, X, y) -> np.ndarray:
    """(n_repeats, n_rows) validation-fold index of every row in every repeat (metadata for the OOF file)."""
    n_repeats = getattr(cv, "n_repeats", 1)
    n_folds = cv.get_n_splits() // n_repeats
    ids = np.full((n_repeats, len(y)), -1, dtype=np.int8)
    for i, (_, val_idx) in enumerate(cv.split(X, y)):
        ids[i // n_folds, val_idx] = i % n_folds
    return ids


def summarize_folds(folds: list[dict]) -> dict:
    return {m: {"mean": float(np.mean([f[m] for f in folds])), "std": float(np.std([f[m] for f in folds])),
                "min": float(np.min([f[m] for f in folds])), "max": float(np.max([f[m] for f in folds])),
                "folds": [float(f[m]) for f in folds]} for m in FULL_METRICS}


def corrected_resampled_ttest(a, b, test_fraction: float) -> dict:
    """Paired comparison of two models scored on the SAME repeated-CV splits (Nadeau & Bengio 2003;
    Bouckaert & Frank 2004). Splits overlap, so the naive paired t-test is over-confident; the variance is
    inflated by (1/k + n_test/n_train). Returns the mean difference a - b, t, two-sided p (df = k - 1)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"mean_difference": float(d.mean()), "t": float("nan"), "p_value": 0.0 if d.mean() else 1.0, "k": k}
    t = d.mean() / np.sqrt((1 / k + test_fraction / (1 - test_fraction)) * var)
    return {"mean_difference": float(d.mean()), "t": float(t), "p_value": float(2 * stats.t.sf(abs(t), k - 1)),
            "k": k}


def calibration_data(y, proba, bins: int = 10) -> list[dict]:
    """Reliability diagram data on equal-width probability bins (empty bins omitted)."""
    y, proba = np.asarray(y), np.asarray(proba)
    idx = np.minimum((proba * bins).astype(int), bins - 1)
    return [{"bin_low": b / bins, "bin_high": (b + 1) / bins, "count": int((idx == b).sum()),
             "mean_predicted": float(proba[idx == b].mean()), "observed_rate": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def expected_calibration_error(y, proba, bins: int = 10) -> float:
    """Count-weighted mean |observed rate - mean predicted| over 10 equal-width probability bins."""
    rows = calibration_data(y, proba, bins)
    return float(sum(r["count"] * abs(r["observed_rate"] - r["mean_predicted"]) for r in rows) / len(np.asarray(y)))


def calibration_stats(y, proba) -> dict:
    """Brier, log loss, ECE, and logistic recalibration of the outcome on logit(p):
    slope (1 = ideal; < 1 means predictions are too extreme) and calibration-in-the-large intercept
    (slope fixed at 1; 0 = ideal; > 0 means predictions are too low on average). Probabilities are clipped
    to [1e-6, 1 - 1e-6] first, because isotonic calibration can output exactly 0 or 1."""
    y, proba = np.asarray(y), np.asarray(proba)
    logit = np.log(np.clip(proba, 1e-6, 1 - 1e-6) / (1 - np.clip(proba, 1e-6, 1 - 1e-6)))
    slope = LogisticRegression(C=np.inf).fit(logit[:, None], y).coef_[0, 0]
    intercept = optimize.brentq(lambda a: (1 / (1 + np.exp(-(a + logit)))).mean() - y.mean(), -10, 10)
    return {"brier": float(brier_score_loss(y, proba)), "log_loss": float(log_loss(y, proba, labels=[0, 1])),
            "ece": expected_calibration_error(y, proba), "slope": float(slope), "intercept": float(intercept)}


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
            "accuracy": (tp + tn) / len(y),
            "predicted_positive_rate": (tp + fp) / len(y),
        })
    return rows


def decision_curve(y, proba, thresholds) -> list[dict]:
    """Net benefit = TP/n - FP/n * t / (1 - t) (Vickers & Elkin 2006), for the model and for flagging
    everyone; flagging no one is 0. Exploratory and specific to this dataset's prevalence: it shows how the
    trade-off moves with t, it does not establish clinical utility."""
    y, proba = np.asarray(y).astype(bool), np.asarray(proba)
    n, prevalence = len(y), y.mean()
    out = []
    for t in thresholds:
        pred = proba >= t
        w = t / (1 - t)
        out.append({"threshold": float(t),
                    "model": float((pred & y).sum() / n - (pred & ~y).sum() / n * w),
                    "flag_all": float(prevalence - (1 - prevalence) * w)})
    return out


BOOTSTRAP_METRICS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "f1": lambda y, p: f1_score(y, p >= 0.5),
    "brier": brier_score_loss,
    "ece": expected_calibration_error,
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


def subgroup_metrics(y, proba, groups, min_per_class: int = 30, n_resamples: int = 500, seed: int = 42) -> list[dict]:
    """Subgroup Performance Analysis. Groups with fewer than `min_per_class` rows of either class (including
    groups with only one class) get counts only, `reliable: False` and an `unavailable` reason; no metric is
    computed for them. Reliable groups get percentile-bootstrap 95% intervals (within-group resampling)."""
    y, proba, groups = pd.Series(np.asarray(y)), pd.Series(np.asarray(proba)), pd.Series(np.asarray(groups))
    rng = np.random.default_rng(seed)
    rows = []
    for g in sorted(groups.unique(), key=str):
        mask = (groups == g).to_numpy()
        yg, pg = y[mask].to_numpy(), proba[mask].to_numpy()
        row = {"group": str(g), "n": int(mask.sum()), "positives": int(yg.sum()), "prevalence": float(yg.mean()),
               "observed_rate": float(yg.mean()), "mean_predicted": float(pg.mean())}
        minority = min(yg.sum(), len(yg) - yg.sum())
        row["reliable"] = bool(minority >= min_per_class)
        if not row["reliable"]:
            row["unavailable"] = ("Metric unavailable for this subgroup: only one outcome class." if minority == 0 else
                                  f"Metric unavailable for this subgroup: fewer than {min_per_class} records of one class.")
            rows.append(row)
            continue
        m = full_metrics(yg, pg)
        row.update({k: m[k] for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "brier")})
        draws = {k: [] for k in ("roc_auc", "pr_auc", "recall", "brier")}
        for _ in range(n_resamples):
            idx = rng.integers(0, len(yg), len(yg))
            if 0 < yg[idx].sum() < len(idx):
                d = full_metrics(yg[idx], pg[idx])
                for k in draws:
                    draws[k].append(d[k])
        row["ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}
        rows.append(row)
    return rows
