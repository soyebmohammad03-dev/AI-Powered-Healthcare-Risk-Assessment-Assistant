#!/usr/bin/env bash
# One-command setup from a fresh clone. Safe to rerun: it only creates what is missing.
#   ./scripts/setup.sh                 venv + dependencies + dataset + any missing model files
#   ./scripts/setup.sh --regenerate    also retrain and rebuild every artifact (about 10 min)
# Override the interpreter with PYTHON=/path/to/python3.12 ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || PY=python3
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null || {
  echo "Python 3.12 is required (found: $("$PY" --version 2>&1))."
  echo "Install it (macOS: brew install python@3.12) or set PYTHON=/path/to/python3.12."; exit 1; }

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements-dev.txt

.venv/bin/python -c 'import xgboost' 2>/dev/null || {
  echo "XGBoost cannot load its OpenMP runtime."
  [ "$(uname)" = Darwin ] && echo "Install it with: brew install libomp   then rerun this script."; exit 1; }

echo "Dataset…"
.venv/bin/python -c 'from src.data_loader import load_dataset, CLEAN_ROWS; n = len(load_dataset()); assert n == CLEAN_ROWS, n; print(f"  {n:,} rows after cleaning")'

run() { echo "Running python -m $1…"; .venv/bin/python -m "$1"; }
if [ "${1:-}" = --regenerate ]; then
  run src.train_models; run src.analysis; run src.reliability; run src.shift_analysis
else
  # Model files are not committed: build them if missing or written by other library versions.
  if ! .venv/bin/python -m src.prediction >/dev/null 2>&1; then
    run src.train_models; run src.reliability  # reliability also writes models/novelty_detector.joblib
  elif [ ! -f models/novelty_detector.joblib ]; then
    run src.reliability
  fi
fi

.venv/bin/python -m src.prediction
echo "Done. Test: .venv/bin/python -m pytest    Run: .venv/bin/streamlit run app.py"
