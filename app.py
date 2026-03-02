import streamlit as st
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

It is designed as an educational decision-support tool that demonstrates how
credit analysts can translate borrower information into practical lending decisions.
"""
)

st.markdown("---")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("What this app covers")
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

    st.subheader("Business objective")
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
    Loan-to-Income, interest burden, PD, and Expected Loss</p>
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

st.subheader("How to navigate")
st.markdown(
    """
Use the **sidebar** to open:
1. **Portfolio Overview**
2. **Borrower Underwriting Tool**
3. **Policy Sensitivity**
"""
)

st.info(
    "Disclaimer: This app is an educational underwriting simulation and not a production lending system."
)