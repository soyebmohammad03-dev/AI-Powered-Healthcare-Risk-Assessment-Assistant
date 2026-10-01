import numpy as np
import pandas as pd
import pytest

from src.data_loader import TARGET, load_dataset
from src.evaluate_models import evaluate
from src.preprocessing import CATEGORIES, FEATURES, build_preprocessor
from src.train_models import build_models


@pytest.fixture(scope="module")
def df():
    return load_dataset()


def test_dataset_structure(df):
    assert df.shape == (303, 14)
    assert set(FEATURES + [TARGET]) == set(df.columns)
    assert set(df[TARGET]) == {0, 1}
    assert df[TARGET].sum() == 139  # num 1-4 in the Cleveland data
    assert df.isna().sum().to_dict() == {**{c: 0 for c in df.columns}, "ca": 4, "thal": 2}
    for col, codes in CATEGORIES.items():
        assert set(df[col].dropna()) == set(codes)


def test_preprocessing_imputes_and_encodes(df):
    out = build_preprocessor().fit_transform(df[FEATURES])
    assert out.shape[0] == len(df)
    assert not np.isnan(out).any()


def test_preprocessing_rejects_undocumented_category(df):
    pre = build_preprocessor().fit(df[FEATURES])
    bad = df[FEATURES].iloc[[0]].copy()
    bad["cp"] = 9.0
    with pytest.raises(ValueError):
        pre.transform(bad)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost"])
def test_model_trains_and_outputs_probabilities(df, name):
    model = build_models()[name].fit(df[FEATURES], df[TARGET])
    proba = model.predict_proba(df[FEATURES])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    metrics = evaluate(model, df[FEATURES], df[TARGET])
    assert metrics["roc_auc"] > 0.8  # training-set sanity check, not a reported metric
    assert np.sum(metrics["confusion_matrix"]) == len(df)
