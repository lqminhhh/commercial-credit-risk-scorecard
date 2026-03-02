from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def get_data_path(filename: str = "credit_risk_dataset.csv") -> Path:
    return DATA_DIR / filename


def load_dataset(filename: str = "credit_risk_dataset.csv") -> pd.DataFrame:
    data_path = get_data_path(filename)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    return pd.read_csv(data_path)