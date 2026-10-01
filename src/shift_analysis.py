"""Controlled distribution-shift experiment -> artifacts/shift_analysis.json.

Run: python -m src.shift_analysis   (after python -m src.train_models)

SYNTHETIC. This is not a validation on a real future or external population. It takes the held-out test set
and changes it in two controlled ways, then re-measures each candidate model:
  * population-mix shift: test rows are resampled with weights exp(beta * z), z = the standardised feature.
    Each row keeps its own label, so the outcome-given-inputs relationship is unchanged; only the mix of
    people changes (e.g. more older people).
  * measurement offset: an input is recorded systematically higher or lower (e.g. every blood pressure
    +10 mmHg) while labels stay the same. Records that become invalid are dropped and counted.
It shows how sensitive the reported metrics are to distribution change; nothing is tuned on it.
"""
import json

import numpy as np
import pandas as pd

from src.data_loader import ROOT, SEED, load_dataset, provenance, split
from src.evaluate_models import calibration_stats, full_metrics
from src.preprocessing import add_bmi, plausible_mask
from src.reliability import load_candidates

ARTIFACT = ROOT / "artifacts" / "shift_analysis.json"
MIX_FEATURES = {"age": "Age", "ap_hi": "Systolic blood pressure", "bmi": "BMI"}
BETAS = [-1.0, -0.5, 0.5, 1.0]
OFFSETS = [
    ("Blood pressure recorded 10 mmHg lower", {"ap_hi": -10, "ap_lo": -10}, "add"),
    ("Blood pressure recorded 5 mmHg higher", {"ap_hi": 5, "ap_lo": 5}, "add"),
    ("Blood pressure recorded 10 mmHg higher", {"ap_hi": 10, "ap_lo": 10}, "add"),
    ("Weight recorded 5% lower", {"weight": 0.95}, "multiply"),
    ("Weight recorded 5% higher", {"weight": 1.05}, "multiply"),
    ("Age recorded 3 years higher", {"age": 3}, "add"),
]


def score(candidates: dict, X: pd.DataFrame, y: pd.Series) -> dict:
    out = {}
    for name, model in candidates.items():
        p = model.predict_proba(X)[:, 1]
        m, c = full_metrics(y, p), calibration_stats(y, p)
        out[name] = {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier": m["brier"], "ece": c["ece"],
                     "mean_predicted": float(p.mean()), "observed_rate": float(np.mean(y))}
    return out


def population_mix(X: pd.DataFrame, feature: str, beta: float, seed: int) -> np.ndarray:
    values = add_bmi(X)[feature]
    z = ((values - values.mean()) / values.std()).to_numpy()
    w = np.exp(beta * z)
    return np.random.default_rng(seed).choice(len(X), len(X), replace=True, p=w / w.sum())


def offset(X: pd.DataFrame, change: dict, how: str) -> pd.DataFrame:
    out = X.copy()
    for f, v in change.items():
        out[f] = out[f] + v if how == "add" else out[f] * v
    return out


def main() -> dict:
    _, X_test, _, y_test = split(load_dataset())
    candidates = load_candidates()
    report = {"provenance": provenance(),"synthetic": True, "kind": "Controlled distribution-shift experiment",
              "note": "Synthetic perturbations of the held-out test set. Not a real-world or external validation.",
              "rows": len(X_test), "seed": SEED, "baseline": score(candidates, X_test, y_test),
              "population_mix": [], "measurement_offset": []}
    for i, (feature, label) in enumerate(MIX_FEATURES.items()):
        for j, beta in enumerate(BETAS):
            idx = population_mix(X_test, feature, beta, SEED + 10 * i + j)
            Xs, ys = X_test.iloc[idx], y_test.iloc[idx]
            report["population_mix"].append({
                "feature": feature, "label": label, "beta": beta,
                "feature_mean": float(add_bmi(Xs)[feature].mean()),
                "feature_mean_baseline": float(add_bmi(X_test)[feature].mean()),
                "metrics": score(candidates, Xs, ys)})
    for label, change, how in OFFSETS:
        Xs = offset(X_test, change, how)
        ok = plausible_mask(Xs).to_numpy()
        report["measurement_offset"].append({"label": label, "dropped_invalid": int((~ok).sum()),
                                             "metrics": score(candidates, Xs[ok], y_test[ok])})
    ARTIFACT.write_text(json.dumps(report, indent=1, allow_nan=False))
    return report


if __name__ == "__main__":
    r = main()
    print(f"Wrote {ARTIFACT}")
    fmt = lambda m: {n: f"AUC {v['roc_auc']:.3f} Brier {v['brier']:.3f} pred {v['mean_predicted']:.3f} obs {v['observed_rate']:.3f}" for n, v in m.items()}
    print("baseline", fmt(r["baseline"]))
    for s in r["population_mix"]:
        print(s["feature"], s["beta"], round(s["feature_mean"], 1), fmt(s["metrics"])["xgboost"])
    for s in r["measurement_offset"]:
        print(s["label"], s["dropped_invalid"], fmt(s["metrics"])["xgboost"])
