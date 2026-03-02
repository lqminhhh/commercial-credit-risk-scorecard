"""
Underwriting decision logic for borrower assessment.
"""

from __future__ import annotations

from src.config import APPROVE_PD_CUTOFF, REJECT_PD_CUTOFF


def assign_underwriting_decision(
    risk_tier: str,
    predicted_pd: float,
    prior_default_flag: int,
) -> str:
    """
    Assign underwriting decision based on risk tier and predicted PD.

    Rules
    -----
    - Reject:
        - if High Risk tier, OR
        - PD >= reject threshold
    - Approve:
        - if Low Risk tier AND PD < approve threshold AND no prior default
    - Manual Review:
        - all other cases
    """
    if risk_tier == "High Risk" or predicted_pd >= REJECT_PD_CUTOFF:
        return "Reject"

    if (
        risk_tier == "Low Risk"
        and predicted_pd < APPROVE_PD_CUTOFF
        and prior_default_flag == 0
    ):
        return "Approve"

    return "Manual Review"


def summarize_decision_reason(
    risk_tier: str,
    predicted_pd: float,
    loan_to_income: float,
    prior_default_flag: int,
) -> str:
    """
    Generate a short underwriting explanation in business language.
    """
    reasons = []

    if risk_tier == "High Risk":
        reasons.append("loan-to-income ratio is elevated")
    elif risk_tier == "Medium Risk":
        reasons.append("loan-to-income ratio is moderate")
    elif risk_tier == "Low Risk":
        reasons.append("loan-to-income ratio is relatively conservative")

    if predicted_pd >= REJECT_PD_CUTOFF:
        reasons.append("predicted default probability is above the rejection threshold")
    elif predicted_pd >= APPROVE_PD_CUTOFF:
        reasons.append("predicted default probability is above the low-risk approval band")
    else:
        reasons.append("predicted default probability is within a lower-risk range")

    if prior_default_flag == 1:
        reasons.append("prior default history is present")
    else:
        reasons.append("no prior default history is present")

    return "; ".join(reasons).capitalize() + "."