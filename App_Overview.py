import streamlit as st
import pandas as pd
from pathlib import Path


st.set_page_config(
    page_title="Credit Risk Underwriting App",
    page_icon="🏦",
    layout="wide",
)


def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

st.title("🏦 Credit Risk Underwriting App")
st.markdown(
    """
This application simulates a simplified **commercial loan underwriting workflow** using
borrower-level credit data, underwriting rules, probability-of-default (PD) modeling,
and expected loss estimation.
"""
)
st.info(
    "Disclaimer: This app is an educational underwriting simulation and not a production lending system."
)

with st.expander("About the Dataset"):
    st.markdown(
        """
This project uses a borrower-level credit risk dataset containing applicant financial,
loan, and credit history information commonly used in underwriting analysis.

The target variable is **`loan_status`**:
- **0** = non-default
- **1** = default
"""
    )

    dataset_dict = pd.DataFrame(
        {
            "Feature Name": [
                "person_age",
                "person_income",
                "person_home_ownership",
                "person_emp_length",
                "loan_intent",
                "loan_grade",
                "loan_amnt",
                "loan_int_rate",
                "loan_status",
                "loan_percent_income",
                "cb_person_default_on_file",
                "cb_person_cred_hist_length",
            ],
            "Description": [
                "Borrower age",
                "Annual income",
                "Home ownership status",
                "Employment length (years)",
                "Loan purpose",
                "Loan grade",
                "Loan amount",
                "Interest rate",
                "Loan outcome (0 = non-default, 1 = default)",
                "Loan amount as a share of income",
                "Historical default flag",
                "Credit history length",
            ],
        }
    )

    st.dataframe(dataset_dict, use_container_width=True, hide_index=True)

    st.markdown(
        """
**Source:** https://www.kaggle.com/datasets/laotse/credit-risk-dataset
"""
    )

st.markdown("---")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("What This App Covers")
    st.markdown(
        """
- **Portfolio Overview**  
  Review portfolio-level default risk, risk tier distribution, underwriting decision buckets,
  and expected loss concentration.

- **Borrower Underwriting Tool**  
  Input a borrower profile and generate key underwriting metrics, predicted PD,
  an approval recommendation, and expected loss.

- **Policy Sensitivity**  
  Adjust underwriting thresholds and LGD assumptions to see how decisions and expected loss change.
"""
    )

    st.subheader("Business Objective")
    st.markdown(
        """
The goal of this project is to simulate how a bank or finance team might:
- assess borrower repayment capacity,
- identify higher-risk loan applications,
- support approval / review / rejection decisions,
- and quantify expected loss at both borrower and portfolio levels.
"""
    )

with right:
    st.subheader("Quick Highlights")
    st.markdown(
        """
<div class="highlight-card">
    <p><strong>Risk Metrics</strong><br>
    Loan-to-Income, interest burden, probability-of-default (PD), and Expected Loss</p>
</div>

<div class="highlight-card">
    <p><strong>Decision Logic</strong><br>
    Approve / Manual Review / Reject simulation</p>
</div>

<div class="highlight-card">
    <p><strong>Use Case</strong><br>
    Commercial Credit, Banking Risk, and Underwriting Analytics</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("---")

st.subheader("How To Navigate")
st.markdown(
    """
Use the **sidebar** to open:
1. **Portfolio Overview**
2. **Borrower Underwriting Tool**
3. **Policy Sensitivity**
"""
)