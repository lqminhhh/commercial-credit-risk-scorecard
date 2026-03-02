"""
Model loading and prediction helpers for PD scoring.
"""

from __future__ import annotations

import pandas as pd


def predict_pd(model, borrower_df: pd.DataFrame) -> float:
    """
    Predict borrower Probability of Default (PD) using a trained sklearn pipeline.

    Parameters
    ----------
    model :
        Trained sklearn model or pipeline with predict_proba().
    borrower_df : pd.DataFrame
        Single-row DataFrame containing model-ready borrower features.

    Returns
    -------
    float
        Predicted probability of default.
    """
    if borrower_df.empty:
        raise ValueError("borrower_df is empty. Cannot generate prediction.")

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba().")

    pd_score = model.predict_proba(borrower_df)[0, 1]
    return float(pd_score)


def score_borrower(model, borrower_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return borrower DataFrame with predicted PD appended.

    Parameters
    ----------
    model :
        Trained sklearn model or pipeline.
    borrower_df : pd.DataFrame
        Single-row borrower feature DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of borrower_df with predicted_pd column added.
    """
    scored_df = borrower_df.copy()
    scored_df["predicted_pd"] = predict_pd(model, borrower_df)
    return scored_df