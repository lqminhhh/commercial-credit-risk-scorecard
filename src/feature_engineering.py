"""
Feature engineering functions for borrower-level underwriting analysis.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import (
    LOW_RISK_LTI_CUTOFF,
    HIGH_RISK_LTI_CUTOFF,
    LOAN_GRADE_MAP,
)


def compute_loan_to_income(person_income: float, loan_amnt: float) -> float:
    """
    Compute Loan-to-Income (LTI) ratio.

    Parameters
    ----------
    person_income : float
        Borrower's annual income.
    loan_amnt : float
        Requested loan amount.

    Returns
    -------
    float
        Loan-to-income ratio.
    """
    if person_income is None or person_income <= 0:
        return np.nan
    return loan_amnt / person_income


def compute_interest_burden_pct_income(
    person_income: float,
    loan_amnt: float,
    loan_int_rate: float,
) -> float:
    """
    Approximate annual interest burden as a percentage of income.

    Formula
    -------
    (loan_amnt * loan_int_rate) / person_income

    Notes
    -----
    loan_int_rate is expected as a percentage (e.g., 12.5, not 0.125).
    """
    if person_income is None or person_income <= 0:
        return np.nan
    if loan_int_rate is None:
        return np.nan
    return (loan_amnt * (loan_int_rate / 100.0)) / person_income


def map_loan_grade(loan_grade: str) -> float:
    """
    Map loan grade from A-G to ordered numeric scale.
    A = 1 (best), G = 7 (worst)
    """
    if loan_grade is None:
        return np.nan
    return LOAN_GRADE_MAP.get(str(loan_grade).upper(), np.nan)


def map_prior_default_flag(cb_person_default_on_file: str) -> int:
    """
    Map prior default flag:
    Y -> 1
    N -> 0
    """
    if str(cb_person_default_on_file).upper() == "Y":
        return 1
    return 0


def assign_risk_tier(loan_to_income: float) -> str:
    """
    Assign underwriting risk tier based on LTI thresholds.
    """
    if pd.isna(loan_to_income):
        return "Unknown"
    if loan_to_income < LOW_RISK_LTI_CUTOFF:
        return "Low Risk"
    if loan_to_income < HIGH_RISK_LTI_CUTOFF:
        return "Medium Risk"
    return "High Risk"


def prepare_borrower_features(input_data: dict) -> pd.DataFrame:
    """
    Convert a single borrower input dictionary into a model-ready DataFrame
    with engineered underwriting features.

    Expected input keys
    -------------------
    person_income
    person_emp_length
    cb_person_cred_hist_length
    loan_amnt
    loan_int_rate
    person_home_ownership
    loan_intent
    loan_grade
    cb_person_default_on_file

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with both raw and engineered features.
    """
    borrower = input_data.copy()

    # Normalize categorical strings
    borrower["person_home_ownership"] = str(
        borrower.get("person_home_ownership", "")
    ).upper()
    borrower["loan_intent"] = str(
        borrower.get("loan_intent", "")
    ).upper()
    borrower["loan_grade"] = str(
        borrower.get("loan_grade", "")
    ).upper()
    borrower["cb_person_default_on_file"] = str(
        borrower.get("cb_person_default_on_file", "")
    ).upper()

    # Engineered features
    borrower["loan_to_income"] = compute_loan_to_income(
        person_income=borrower.get("person_income"),
        loan_amnt=borrower.get("loan_amnt"),
    )

    borrower["interest_burden_pct_income"] = compute_interest_burden_pct_income(
        person_income=borrower.get("person_income"),
        loan_amnt=borrower.get("loan_amnt"),
        loan_int_rate=borrower.get("loan_int_rate"),
    )

    borrower["loan_grade_num"] = map_loan_grade(
        borrower.get("loan_grade")
    )

    borrower["prior_default_flag"] = map_prior_default_flag(
        borrower.get("cb_person_default_on_file")
    )

    borrower["risk_tier"] = assign_risk_tier(
        borrower.get("loan_to_income")
    )

    return pd.DataFrame([borrower])


def get_model_feature_columns() -> list[str]:
    """
    Return the exact feature columns expected by the trained PD model.
    """
    return [
        "person_income",
        "person_emp_length",
        "cb_person_cred_hist_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_to_income",
        "interest_burden_pct_income",
        "loan_grade_num",
        "prior_default_flag",
        "person_home_ownership",
        "loan_intent",
    ]