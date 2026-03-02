"""
Project-wide configuration values for underwriting and risk calculations.
"""

# -----------------------------
# Expected Loss Assumption
# -----------------------------
LGD = 0.45  # Loss Given Default

# -----------------------------
# Loan-to-Income (LTI) thresholds
# -----------------------------
LOW_RISK_LTI_CUTOFF = 0.15
HIGH_RISK_LTI_CUTOFF = 0.35

# -----------------------------
# PD thresholds for decisioning
# -----------------------------
APPROVE_PD_CUTOFF = 0.15
REJECT_PD_CUTOFF = 0.35

# -----------------------------
# Loan grade mapping
# -----------------------------
LOAN_GRADE_MAP = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
}

# -----------------------------
# Supported categorical values
# -----------------------------
HOME_OWNERSHIP_OPTIONS = ["RENT", "OWN", "MORTGAGE", "OTHER"]

LOAN_INTENT_OPTIONS = [
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "PERSONAL",
    "DEBTCONSOLIDATION",
    "HOMEIMPROVEMENT",
]

PRIOR_DEFAULT_OPTIONS = ["Y", "N"]