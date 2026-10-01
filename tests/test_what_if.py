from dataclasses import asdict

import joblib
import pytest

from src.data_loader import ROOT
from src.prediction import DEMO_INPUTS, FINAL_MODEL_PATH, InvalidInputError, predict
from src.preprocessing import PLAUSIBLE
from src.what_if import ADJUSTABLE, build_scenario, changes_from, compare, response_curve

B = DEMO_INPUTS["Example Patient B"]


@pytest.fixture(scope="module", autouse=True)
def artifact():
    if not FINAL_MODEL_PATH.exists():
        from src.train_models import main
        main()


def test_no_change_reproduces_baseline():
    r = compare(B, {})
    assert r.baseline == r.scenario == predict(B)
    assert r.delta_pp == 0 and r.changes == {} and r.single_effects == {}


def test_delta_matches_two_predictions():
    r = compare(B, {"ap_hi": 120, "ap_lo": 80, "smoke": 0})
    assert r.delta_pp == pytest.approx((r.scenario.probability_positive - r.baseline.probability_positive) * 100)
    assert set(r.changes) == {"ap_hi", "ap_lo", "smoke"}
    assert r.changes["ap_hi"] == (B.ap_hi, 120)
    assert 0 <= r.scenario.probability_positive <= 1


def test_scenario_does_not_modify_baseline():
    before = asdict(B)
    compare(B, {"weight": 70.0})
    assert asdict(B) == before


def test_deterministic():
    assert compare(B, {"cholesterol": 3}) == compare(B, {"cholesterol": 3})


@pytest.mark.parametrize("changes, message", [
    ({"ap_hi": 300}, "outside the accepted range"),
    ({"ap_lo": 140}, "Systolic blood pressure must be higher"),   # 130/140 is impossible
    ({"weight": 250.0}, "BMI"),                                      # 175 cm, 250 kg -> BMI 81.6
    ({"cholesterol": 5}, "not a valid code"),
    ({"age": 40}, "cannot be changed"),
    ({"gender": 1}, "cannot be changed"),
])
def test_guardrails(changes, message):
    with pytest.raises(InvalidInputError, match=message):
        compare(B, changes)


def test_single_effect_reported_as_none_when_change_is_only_valid_combined():
    r = compare(B, {"ap_hi": 160, "ap_lo": 140})  # diastolic 140 alone would exceed systolic 130
    assert r.single_effects["ap_lo"] is None and r.single_effects["ap_hi"] is not None


def test_response_curve_stays_in_valid_domain_and_contains_baseline():
    curve = response_curve(B, "ap_hi")
    assert curve["value"].min() > B.ap_lo and curve["value"].max() <= PLAUSIBLE["ap_hi"][1]
    assert curve["probability"].between(0, 1).all()
    at_baseline = curve.loc[curve["value"] == B.ap_hi, "probability"].iloc[0]
    assert at_baseline == pytest.approx(predict(B).probability_positive)
    # Monotonicity is a property of the linear candidate (linear score + monotone calibration), not of
    # every model; the deployed tree model's non-monotone response is reported in reliability.json.
    lr = {"pipeline": joblib.load(ROOT / "models" / "logistic_regression.joblib")}
    assert response_curve(B, "ap_hi", lr)["probability"].is_monotonic_increasing


def test_response_curve_rejects_other_features():
    with pytest.raises(ValueError):
        response_curve(B, "age")


def test_changes_from_keeps_only_adjustable_differences():
    values = {**asdict(B), "ap_hi": 140, "age": 99}
    assert changes_from(B, values) == {"ap_hi": 140}
    assert set(ADJUSTABLE) <= set(asdict(B))
    assert build_scenario(B, {}) == B
