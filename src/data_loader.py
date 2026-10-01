"""Download and load the UCI Heart Disease (Cleveland) dataset."""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed.cleveland.data"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

# Column order from the dataset's documentation (heart-disease.names, attributes 3,4,9,10,12,16,19,32,38,40,41,44,51,58).
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
TARGET = "target"


def download(path: Path = DATA_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Checksum mismatch for {path}: {digest}")
    return path


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return features plus a binary `target` (1 = disease present, i.e. num 1-4; 0 = absent)."""
    df = pd.read_csv(download(path), header=None, names=COLUMNS, na_values="?")
    df[TARGET] = (df.pop("num") > 0).astype(int)
    return df
