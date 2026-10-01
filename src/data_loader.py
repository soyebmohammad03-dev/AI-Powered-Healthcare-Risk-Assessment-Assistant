"""Download, clean and load the Cardiovascular Disease dataset (Kaggle: sulianova)."""
import hashlib
import io
import platform
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import CATEGORIES, FEATURES, PLAUSIBLE, bmi

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cardio_train.csv"
SOURCE = "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset"
URL = "https://www.kaggle.com/api/v1/datasets/download/sulianova/cardiovascular-disease-dataset"
SHA256 = "21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c"  # cardio_train.csv

RAW_COLUMNS = ["id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
               "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
TARGET = "cardio"  # 1 = cardiovascular disease present, 0 = absent
SEED = 42
RAW_ROWS, CLEAN_ROWS = 70_000, 68_573  # pinned by SHA256; documented here for readers


def download(path: Path = DATA_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(URL, timeout=60) as resp, zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                path.write_bytes(zf.read("cardio_train.csv"))
        except Exception as exc:
            raise RuntimeError(f"Could not download the dataset ({exc}). Download it manually from {SOURCE}, "
                               f"unzip it and place cardio_train.csv at {path}.") from exc
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Checksum mismatch for {path}: {digest}. Expected the original cardio_train.csv "
                         f"from {SOURCE} (sha256 {SHA256}). Delete the file and rerun to download it again.")
    return path


def provenance() -> dict:
    """Recorded in every generated artifact: what produced it."""
    import datetime
    from importlib.metadata import version
    return {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "dataset_sha256": SHA256, "seed": SEED,
            "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost", "shap")}}


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(download(path), sep=";")
    if list(df.columns) != RAW_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the documented data-quality rules in order. Returns (clean data, rows removed per rule).

    Output columns: FEATURES + TARGET, age in years. `id` is dropped: an identifier, never a feature.
    """
    df, removed, _ = clean_with_exclusions(raw)
    return df, removed


def clean_with_exclusions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """As clean(), plus the excluded rows themselves per rule (for the data-quality report)."""
    df = raw.drop(columns="id").apply(pd.to_numeric, errors="coerce")
    df["age"] = df["age"] / 365.25  # days -> years
    removed, excluded = {}, {}

    def drop(rule: str, bad: pd.Series):
        nonlocal df
        removed[rule] = int(bad.sum())
        excluded[rule] = df[bad]
        df = df[~bad]

    drop("malformed_or_missing", df.isna().any(axis=1))
    drop("invalid_target", ~df[TARGET].isin([0, 1]))
    drop("invalid_category_code", ~pd.concat([df[c].isin(v) for c, v in CATEGORIES.items()], axis=1).all(axis=1))
    drop("duplicate_record", df.duplicated())
    for col in ["age", "height", "weight", "ap_hi", "ap_lo"]:
        drop(f"{col}_out_of_range", ~df[col].between(*PLAUSIBLE[col]))
    drop("ap_hi_not_above_ap_lo", df["ap_hi"] <= df["ap_lo"])
    drop("bmi_out_of_range", ~bmi(df["height"], df["weight"]).between(*PLAUSIBLE["bmi"]))

    df = df.astype({c: int for c in [*CATEGORIES, TARGET]}).reset_index(drop=True)
    return df[FEATURES + [TARGET]], removed, excluded


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))[0]


def split(df: pd.DataFrame):
    """The project's one train/test split: stratified 80/20, seed 42. Returns X_train, X_test, y_train, y_test."""
    return train_test_split(df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=SEED)
