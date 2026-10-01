import json

import numpy as np
import pandas as pd
import pytest

from src.data_loader import RAW_COLUMNS, ROOT, TARGET, clean, load_raw, split
from src.evaluate_models import evaluate
from src.preprocessing import (CATEGORIES, FEATURES, MODEL_FEATURES, NUMERIC, PLAUSIBLE, add_bmi,
                               build_preprocessor)
from src.train_models import build_models

GOOD_ROW = dict(id=1, age=50 * 365, gender=1, height=165, weight=65.0, ap_hi=120, ap_lo=80,
                cholesterol=1, gluc=1, smoke=0, alco=0, active=1, cardio=0)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean(raw)


@pytest.fixture(scope="module")
def df(cleaned):
    return cleaned[0]


# ---- data -------------------------------------------------------------------------------------

def test_raw_dataset_structure(raw):
    assert list(raw.columns) == RAW_COLUMNS
    assert raw.shape == (70000, 13)
    assert set(raw[TARGET]) == {0, 1}
    assert raw.isna().sum().sum() == 0


def test_cleaned_dataset(df, cleaned):
    removed = cleaned[1]
    assert len(df) == 68573 and sum(removed.values()) == 70000 - 68573
    assert removed["duplicate_record"] == 24
    assert list(df.columns) == FEATURES + [TARGET]
    assert set(df[TARGET]) == {0, 1}
    assert not df.duplicated().any()
    for col in ["age", "height", "weight", "ap_hi", "ap_lo"]:
        assert df[col].between(*PLAUSIBLE[col]).all()
    assert (df["ap_hi"] > df["ap_lo"]).all()
    assert df["age"].between(29, 65).all()  # converted from days to years
    for col, codes in CATEGORIES.items():
        assert set(df[col]) == set(codes)


def test_cleaning_rules_on_dirty_rows():
    rows = [GOOD_ROW,
            {**GOOD_ROW, "id": 2},                       # duplicate apart from id
            {**GOOD_ROW, "id": 3, "cardio": 2},          # invalid target
            {**GOOD_ROW, "id": 4, "gluc": 7},            # invalid category code
            {**GOOD_ROW, "id": 5, "ap_hi": 16020},       # recording error
            {**GOOD_ROW, "id": 6, "ap_hi": 80, "ap_lo": 90},  # systolic not above diastolic
            {**GOOD_ROW, "id": 7, "height": 120, "weight": 200},  # BMI 139
            {**GOOD_ROW, "id": 8, "weight": "abc"}]      # malformed
    out, removed = clean(pd.DataFrame(rows)[RAW_COLUMNS])
    assert len(out) == 1
    assert removed == {"malformed_or_missing": 1, "invalid_target": 1, "invalid_category_code": 1,
                       "duplicate_record": 1, "age_out_of_range": 0, "height_out_of_range": 0,
                       "weight_out_of_range": 0, "ap_hi_out_of_range": 1, "ap_lo_out_of_range": 0,
                       "ap_hi_not_above_ap_lo": 1, "bmi_out_of_range": 1}
    assert out["age"].iloc[0] == pytest.approx(50 * 365 / 365.25)


def test_no_id_leakage(df):
    assert "id" not in df.columns and "id" not in FEATURES and "id" not in MODEL_FEATURES
    assert TARGET not in FEATURES


# ---- preprocessing ----------------------------------------------------------------------------

def test_bmi_derivation():
    out = add_bmi(pd.DataFrame({"height": [180.0], "weight": [81.0]}))
    assert out["bmi"].iloc[0] == pytest.approx(25.0)


def test_split_is_stratified_disjoint_and_reproducible(df):
    X_train, X_test, y_train, y_test = split(df)
    assert (len(X_train), len(X_test)) == (54858, 13715)
    assert not set(X_train.index) & set(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.001
    assert X_train.index.equals(split(df)[0].index)


def test_preprocessing_columns_and_no_leakage(df):
    X_train, X_test, _, _ = split(df)
    pre = build_preprocessor().fit(X_train)
    encoder = pre[-1]
    names = list(encoder.get_feature_names_out())
    assert names[:len(NUMERIC)] == [f"num__{n}" for n in NUMERIC]
    assert not any("height" in n or "weight" in n for n in names)   # only through BMI
    assert len(names) == 14 and not np.isnan(pre.transform(X_test)).any()
    # scaler learned from the training split only, never the full dataset
    scaler = encoder.named_transformers_["num"]
    np.testing.assert_allclose(scaler.mean_, add_bmi(X_train)[NUMERIC].mean().to_numpy())
    assert not np.allclose(scaler.mean_, add_bmi(df[FEATURES])[NUMERIC].mean().to_numpy())


def test_preprocessing_rejects_unknown_category(df):
    pre = build_preprocessor().fit(df[FEATURES])
    bad = df[FEATURES].iloc[[0]].copy()
    bad["cholesterol"] = 9
    with pytest.raises(ValueError):
        pre.transform(bad)


# ---- models -----------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost"])
def test_models_train_and_produce_metrics(df, name):
    sample = df.sample(4000, random_state=0)  # small sample keeps the test fast
    X_train, X_test, y_train, y_test = split(sample)
    model = build_models()[name].fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    metrics = evaluate(model, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}
    assert metrics["roc_auc"] > 0.7


def test_metrics_report():
    report = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    assert set(report["models"]) == {"logistic_regression", "random_forest", "xgboost"}
    assert report["final_model"] in report["models"] and report["final_model_reason"]
    assert report["dataset"]["raw_records"] == 70000 and report["dataset"]["clean_records"] == 68573
    for r in report["models"].values():
        assert set(r) == {"cv_train_5fold", "test"}
        assert all(0 <= r["cv_train_5fold"][m]["mean"] <= 1 for m in r["cv_train_5fold"])
