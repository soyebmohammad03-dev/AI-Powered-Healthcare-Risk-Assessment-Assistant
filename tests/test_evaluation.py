"""Evaluation layer: metric definitions, invariants, reproducibility and leakage."""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from src.evaluate_models import (FULL_METRICS, bootstrap_ci, calibration_data, full_metrics, oof_cross_validate,
                                 subgroup_metrics, summarize_folds, threshold_metrics)

rng = np.random.default_rng(0)
Y = rng.integers(0, 2, 2000)
P = np.clip(Y * 0.3 + rng.uniform(0, 0.7, 2000), 0, 1)  # informative but imperfect scores


def test_full_metrics_definitions():
    m = full_metrics(Y, P)
    assert set(m) == set(FULL_METRICS)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(Y, P))
    assert m["brier"] == pytest.approx(np.mean((P - Y) ** 2)) == pytest.approx(brier_score_loss(Y, P))
    perfect = full_metrics([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0])
    assert perfect["roc_auc"] == 1 and perfect["brier"] == 0 and perfect["f1"] == 1


def test_threshold_metrics_are_consistent():
    rows = threshold_metrics(Y, P, np.linspace(0.05, 0.95, 19))
    for r in rows:
        assert r["tp"] + r["fp"] + r["tn"] + r["fn"] == len(Y)
        assert r["recall"] + r["fnr"] == pytest.approx(1)
        assert r["specificity"] + r["fpr"] == pytest.approx(1)
        if not math.isnan(r["precision"]):
            assert r["f1"] == pytest.approx(2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"]))
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)  # raising the threshold never increases recall
    fprs = [r["fpr"] for r in rows]
    assert fprs == sorted(fprs, reverse=True)


def test_threshold_metrics_match_full_metrics_at_half():
    row = threshold_metrics(Y, P, [0.5])[0]
    m = full_metrics(Y, P, threshold=0.5)
    assert row["precision"] == pytest.approx(m["precision"]) and row["recall"] == pytest.approx(m["recall"])
    assert row["f1"] == pytest.approx(m["f1"])


def test_calibration_data_counts_and_means():
    bins = calibration_data(Y, P, bins=10)
    assert sum(b["count"] for b in bins) == len(Y)
    for b in bins:
        assert b["bin_low"] <= b["mean_predicted"] <= b["bin_high"]
        assert 0 <= b["observed_rate"] <= 1


def test_bootstrap_is_reproducible_and_brackets_estimate():
    probas = {"a": P, "b": np.clip(P + rng.normal(0, 0.05, len(P)), 0, 1)}
    first = bootstrap_ci(Y, probas, n_resamples=200, seed=7, reference="a")
    second = bootstrap_ci(Y, probas, n_resamples=200, seed=7, reference="a")
    assert first == second
    assert first != bootstrap_ci(Y, probas, n_resamples=200, seed=8, reference="a")
    for metrics in first["models"].values():
        for ci in metrics.values():
            assert ci["lower"] <= ci["estimate"] <= ci["upper"]
    assert set(first["differences"]) == {"b"}


def test_subgroup_metrics_flags_small_groups():
    groups = np.where(np.arange(len(Y)) < 40, "tiny", "large")
    rows = {r["group"]: r for r in subgroup_metrics(Y, P, groups, min_per_class=30)}
    assert rows["tiny"]["n"] == 40 and rows["tiny"]["reliable"] is False and "roc_auc" not in rows["tiny"]
    assert rows["large"]["reliable"] and 0 <= rows["large"]["roc_auc"] <= 1
    assert sum(r["n"] for r in rows.values()) == len(Y)


def test_oof_predictions_do_not_leak():
    """A 1-nearest-neighbour model memorises its training rows. On random labels, out-of-fold
    predictions must be at chance; any leakage of a row into its own fold's model would give ~100%."""
    X = pd.DataFrame(rng.normal(size=(600, 3)), columns=list("abc"))
    y = pd.Series(rng.integers(0, 2, 600))
    folds, oof = oof_cross_validate(KNeighborsClassifier(1), X, y, StratifiedKFold(5, shuffle=True, random_state=0))
    assert oof.shape == (1, len(y)) and not np.isnan(oof).any()
    assert 0.4 < np.mean((oof[0] >= 0.5) == y) < 0.6
    summary = summarize_folds(folds)
    assert len(summary["roc_auc"]["folds"]) == 5
    assert summary["roc_auc"]["mean"] == pytest.approx(np.mean(summary["roc_auc"]["folds"]))
