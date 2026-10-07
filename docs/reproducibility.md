# Reproducibility guide

Everything in this repository can be rebuilt from a fresh clone with one script. This page explains what that script does, where the data comes from, which outputs are committed, and how far results repeat.

## Requirements

| | |
|---|---|
| Python | **3.12** (`.python-version`) |
| Platform | verified on **macOS, Apple Silicon**; the CI runs there too. Linux should work; Windows is untested. |
| System library | XGBoost needs OpenMP. On macOS: `brew install libomp` |
| Disk / network | about 3 MB of data downloaded once; no account or credentials |
| Time | first setup about 10 minutes on an Apple Silicon laptop, mostly model training |

## One-command setup

```bash
./scripts/setup.sh                 # venv, dependencies, dataset, any missing model files
./scripts/setup.sh --regenerate    # also retrain and rebuild every artifact
```

Choose an interpreter with `PYTHON=/path/to/python3.12 ./scripts/setup.sh`. The script is safe to rerun and only creates what is missing:

1. checks for Python 3.12;
2. creates `.venv` and installs the pinned dependencies (`requirements.txt` for runtime; `requirements-dev.txt` adds pytest);
3. checks that XGBoost can load OpenMP, and says how to install it if not;
4. downloads and validates the dataset, and checks the cleaned row count (68,573);
5. trains the models if `models/` is missing them or they were written by other library versions, then builds the novelty detector;
6. loads the saved model and scores the three synthetic demo inputs.

If a pre-existing `.venv` is broken (for example, copied from another machine), delete it and rerun the script.

| Task | Command |
|---|---|
| Run the app | `.venv/bin/streamlit run app.py` |
| Run the tests | `.venv/bin/python -m pytest -q` |
| Retrain the models only | `.venv/bin/python -m src.train_models` |
| Regenerate every model and artifact | `./scripts/setup.sh --regenerate` |
| Check the saved model | `.venv/bin/python -m src.prediction` |

The full pipeline order is `src.train_models` → `src.analysis` → `src.reliability` → `src.shift_analysis`; see [architecture.md](architecture.md) for what each step writes.

## Dataset

The dataset is not committed. `src/data_loader.py` downloads it from Kaggle's public API on first use, with no account: `https://www.kaggle.com/api/v1/datasets/download/sulianova/cardiovascular-disease-dataset`. It then validates the file:

- **Checksum:** `cardio_train.csv` must have SHA-256 `21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c`, which pins all 70,000 rows exactly.
- **Format:** semicolon-separated, columns `id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio` in that order.
- **Cleaning:** documented rules leave 68,573 rows; `setup.sh` checks this count.

**If the download fails** (offline, or Kaggle changes its API), download the dataset manually from the [Kaggle page](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset), unzip it, and place `cardio_train.csv` at `data/cardio_train.csv`. The checksum is still verified.

**Raw schema:** `age` in days (converted to years); `gender` 1 = female, 2 = male; `height` in cm and `weight` in kg (combined into BMI); `ap_hi`/`ap_lo` systolic and diastolic blood pressure in mm Hg; `cholesterol` and `gluc` 1 = normal, 2 = above normal, 3 = well above normal; `smoke`, `alco`, `active` 0/1; target `cardio` 1 = disease present. `id` is never used.

**Split:** stratified 80/20 with seed 42, giving 54,858 training and 13,715 test rows. The test set is used only after model selection.

## What is committed, and what is rebuilt

| Committed | Rebuilt by `setup.sh` |
|---|---|
| `artifacts/metrics.json`, `analysis.json`, `reliability.json`, `shift_analysis.json`, each with a `provenance` block (generation time, Python and library versions, dataset SHA-256, seed) | `models/*.joblib` (final model, candidates, novelty detector) |
| the code, tests, docs and pinned requirements | `artifacts/oof_predictions.npz`, `data/cardio_train.csv`, `.venv/` |

Because the artifacts are committed, the Model and Methodology pages can be read without retraining.

## Determinism

- **Seeds:** every random step uses seed 42: the split, the CV folds, the models, the bootstraps and the perturbations.
- **Repeatability:** regenerating gives identical selection decisions. Logistic Regression reproduces exactly; the tree models reproduce to floating-point precision (multithreaded training), so regenerated artifacts differ from the committed ones only in far decimal places and in their provenance timestamps.
- **Pins:** all runtime dependencies are pinned in `requirements.txt`. Pickled scikit-learn and XGBoost models are only reliable on the versions that wrote them, so the model loader refuses a file built with other versions, or with a feature contract that does not match the code, and names the command that rebuilds it. Change pins only together with a full regeneration.

## Continuous integration

`.github/workflows/tests.yml` runs on every push and pull request, on a macOS Apple Silicon runner with Python 3.12:

1. installs `libomp`;
2. restores the dataset and trained models from the Actions cache, keyed on `requirements.txt` and `src/`;
3. runs `./scripts/setup.sh`, which downloads the pinned dataset and retrains only when the cache missed;
4. runs the full test suite.

The suite is dataset-backed by design (it checks the real pipeline, test-set integrity and SHAP reconciliation), so CI builds the models rather than mocking them. It does not regenerate the committed artifacts.
