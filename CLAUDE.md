# CLAUDE.md — project constitution

**AI-Powered Healthcare Risk Assessment Assistant.** This is an educational, explainable decision-support prototype built for the B.Tech course *Design of Artificial Intelligence Products*. Sole author: Soyeb Mohammad. MIT licensed (`LICENSE`).

It estimates the probability of the dataset label `cardio` from 11 self-reportable inputs. Around that estimate it shows exact SHAP explanations, what-if analysis, model analytics and per-assessment reliability signals.

**Scope: NOT clinical.** It does not diagnose, screen, triage or recommend treatment, and it is not clinically validated. Probabilities agree with one public dataset's labels; they are not clinical risks. Never write anything that implies otherwise.

## Before changing anything

- **Read first.** Inspect the existing implementation before you change architecture. It is deliberate and was reviewed phase by phase. Read the module, its tests and `docs/methodology.md` first.
- **Selection protocol.** Do not change the model-selection protocol (`src/train_models.py`, the comment block above `ALPHA`) or the final model without an explicit request. If you do change it, record why in `docs/methodology.md`.
- **No invented numbers.** Every number in the docs or UI must come from a generated artifact. Never fabricate metrics, clinical claims, validation or "reliability scores". Reliability signals are shown separately on purpose; there is no combined score.
- **Tests.** Never delete or weaken a test to make the suite pass. Fix the cause.
- **Git.** The author is Soyeb Mohammad, using the GitHub noreply email from the global git config. Do **not** add `Co-authored-by` lines or any other author or AI attribution to commits or PRs.
- **Secrets.** No secrets, credentials or machine-specific paths go in the repo; `tests/test_portability.py` checks for paths. The dataset needs no credentials.

## Commands (Python 3.12, `.venv`)

```bash
./scripts/setup.sh                 # fresh clone -> venv, deps, dataset, missing models (~10 min first time)
./scripts/setup.sh --regenerate    # retrain + rebuild all artifacts
.venv/bin/python -m pytest -q      # full suite (162 tests, ~35 s once models exist)
.venv/bin/streamlit run app.py     # the app
```

The pipeline order is `src.train_models` → `src.analysis` → `src.reliability` → `src.shift_analysis`; each is run with `python -m`.

## Dependencies

- `requirements.txt` holds the exact runtime pins and `requirements-dev.txt` adds pytest. They are plain pip files in a venv; do not add other tooling.
- Pickled models are tied to the scikit-learn/XGBoost versions that wrote them. `src/prediction.load_model` rejects a mismatch, so a version bump means a full regeneration.
- On macOS, XGBoost needs `brew install libomp`.

## Data

- **Source:** Kaggle `sulianova/cardiovascular-disease-dataset`. It is downloaded by `src/data_loader.py` from the public API with no account, and pinned by SHA-256 (70,000 raw rows). It is not committed; manual fallback is `data/cardio_train.csv`.
- **Cleaning** (`clean_with_exclusions`, in order): malformed or missing values, invalid target, invalid category code, duplicates, implausible ranges (`preprocessing.PLAUSIBLE`), systolic ≤ diastolic, implausible BMI. This leaves **68,573 rows**. `age` is converted from days to years and `id` is dropped.
- **Features:** age, gender (1 = female, 2 = male), height, weight, ap_hi, ap_lo, cholesterol and gluc (1–3), smoke, alco, active (0/1), plus engineered BMI.
- **Target:** `cardio`, where 1 = disease present.
- **Split:** stratified 80/20 with seed 42, giving 54,858 / 13,715 rows. The test set is used only after selection.

## ML pipeline and final model

- **Candidates:** Logistic Regression, Random Forest and XGBoost, each raw, sigmoid- or isotonic-calibrated (9 variants). They are compared on 5×5 repeated stratified CV of the training split. Preprocessing and calibration are fitted inside each fold.
- **Selection protocol** (pre-declared, no tunable margins): corrected resampled t-tests (Nadeau–Bengio) at α = 0.05.
  - Keep raw probabilities unless a calibrator significantly lowers Brier.
  - A model must have an exact SHAP explanation to be eligible.
  - Walk the interpretability order LR → RF → XGB, moving on only for a significant gain in both ROC-AUC and Brier.
- **Final model: XGBoost, raw probabilities.** Test ROC-AUC 0.804 [0.797, 0.812], Brier 0.180. `models/final_model.joblib` is a bundle: the full Pipeline plus its feature contract, versions and provenance.
- **Explainability:** exact TreeSHAP or linear SHAP on the model score, mapped to probability (`src/explainability.py`), plus permutation importance and PDP/ICE (`src/analysis.py`). Contributions describe the model, not medical causation.
- **Reliability** (`src/reliability.py`, `src/shift_analysis.py`):
  - calibration deep dive, OOF thresholds;
  - Mahalanobis input-conformity detector, fitted on training rows only;
  - prediction and explanation stability under ±1% perturbations;
  - monotonicity checks and candidate-model disagreement;
  - a synthetic distribution-shift experiment, which is *not* external validation.

## Layout

- **`src/`** holds all logic: `data_loader`, `preprocessing`, `evaluate_models`, `train_models`, `analysis`, `reliability`, `shift_analysis`, `prediction`, `explainability`, `what_if` and `recommendations` (rule-based, no LLM).
- **`app.py` and `ui/`** are the Streamlit app. `app.py` handles navigation and `ui/core.py` holds shared components and cached resources. The five pages are Assess, Explain, Explore, Model and Methodology.
- **`artifacts/*.json`** files are generated and committed; each has a `provenance` block. `oof_predictions.npz` is generated and ignored.
- **`models/*.joblib`** files are generated and ignored. `data/` is downloaded and ignored.
- **`docs/`** holds `README.md` (index), `architecture.md`, `reproducibility.md`, `RESEARCH_VALIDATION.md` (evidence chain, claim → artifact traceability, Phase 9 audit), `methodology.md` (how), `evaluation.md` (every measured number) and `model_card.md` (intended use and limitations). Update them whenever methodology or numbers change. `docs/assets/` holds the banner, the architecture diagram (an SVG) and screenshots of the real app taken with the synthetic examples; retake screenshots when the UI changes.
- **`.github/workflows/tests.yml`** runs `scripts/setup.sh` and the full suite on macOS (Apple Silicon) for every push and pull request, caching the dataset and models.
- **`tests/`** contains the pytest suite. Fixtures train the models if they are missing.

## Known limitations

- XGBoost responds in steps and is non-monotone in blood pressure, age and weight. Its top-5 explanation features stay identical in 74% of ±1% perturbations, against 95% for Logistic Regression.
- Weak ranking at ages 60–65, with ROC-AUC 0.70.
- The dataset has limited provenance and self-reported lifestyle inputs, and the model has learned confounded patterns (smoking, alcohol).
- Setup is only tested on macOS arm64 with Python 3.12.

## Status and roadmap

- Phases 1–9 are complete (Phase 8: product UX polish, MIT license; Phase 9: scientific validation and evidence audit, which moved the novelty-detector choice off the test set). They covered data, models, SHAP, guidance, UI, rigorous evaluation and reliability, then reproducibility and public release.
- `v1.0.0` (on `dc03018`) is the frozen release; later commits on `main` are UI polish, documentation and CI only.
- Candidate future phases include external validation and a monotone-constrained XGBoost evaluated under the same protocol. **Do not start a phase unless asked.**
