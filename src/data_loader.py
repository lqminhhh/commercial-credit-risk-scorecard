"""
Data and model loading utilities for the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


def get_data_path(filename: str = "credit_risk_dataset.csv") -> Path:
    """
    Return full path to a file in the data directory.
    """
    return DATA_DIR / filename


def get_model_path(filename: str = "pd_model.pkl") -> Path:
    """
    Return full path to a file in the models directory.
    """
    return MODEL_DIR / filename


def load_dataset(filename: str = "credit_risk_dataset.csv") -> pd.DataFrame:
    """
    Load the raw portfolio dataset.
    """
    data_path = get_data_path(filename)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    return pd.read_csv(data_path)


def load_model(filename: str = "pd_model.pkl"):
    """
    Load the trained PD model (e.g., sklearn pipeline saved with joblib).
    """
    model_path = get_model_path(filename)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)