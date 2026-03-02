"""
Expected Loss (EL) calculations for borrower-level credit risk assessment.
"""

from __future__ import annotations

from src.config import LGD


def calculate_ead(loan_amnt: float) -> float:
    """
    Approximate Exposure at Default (EAD) using loan amount.
    """
    return float(loan_amnt)


def calculate_expected_loss(
    predicted_pd: float,
    loan_amnt: float,
    lgd: float = LGD,
) -> float:
    """
    Calculate Expected Loss (EL).

    Formula
    -------
    EL = PD * LGD * EAD
    """
    ead = calculate_ead(loan_amnt)
    return float(predicted_pd) * float(lgd) * ead


def build_expected_loss_summary(
    predicted_pd: float,
    loan_amnt: float,
    lgd: float = LGD,
) -> dict:
    """
    Return a simple borrower-level expected loss summary.
    """
    ead = calculate_ead(loan_amnt)
    el = calculate_expected_loss(
        predicted_pd=predicted_pd,
        loan_amnt=loan_amnt,
        lgd=lgd,
    )

    return {
        "pd": float(predicted_pd),
        "lgd": float(lgd),
        "ead": float(ead),
        "expected_loss": float(el),
    }