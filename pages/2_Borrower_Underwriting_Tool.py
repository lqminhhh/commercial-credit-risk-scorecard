import streamlit as st
import pandas as pd
from pathlib import Path
from src.config import (
    HOME_OWNERSHIP_OPTIONS,
    LOAN_INTENT_OPTIONS,
    PRIOR_DEFAULT_OPTIONS,
)
from src.data_loader import load_model
from src.feature_engineering import (
    prepare_borrower_features,
    get_model_feature_columns,
)
from src.modeling import predict_pd
from src.underwriting import (
    assign_underwriting_decision,
    summarize_decision_reason,
)
from src.expected_loss import build_expected_loss_summary
from src.utils import format_currency, format_pct

def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.set_page_config(page_title="Borrower Underwriting Tool", layout="wide")

st.title("Borrower Underwriting Tool")
st.markdown(
    """
This page simulates a simplified credit underwriting workflow.  
Enter borrower and loan information to generate key underwriting metrics,
estimate probability of default (PD), assign a decision, and calculate expected loss.
"""
)

st.info(
    "This app is an educational underwriting simulation and not a production lending system."
)


@st.cache_resource
def get_model():
    return load_model("pd_model.pkl")


def build_underwriting_summary(
    risk_tier: str,
    predicted_pd: float,
    decision: str,
    loan_to_income: float,
    prior_default_flag: int,
) -> str:
    """
    Build a short, business-style underwriting summary.
    """
    lti_text = (
        "conservative"
        if risk_tier == "Low Risk"
        else "moderate"
        if risk_tier == "Medium Risk"
        else "elevated"
    )

    prior_default_text = (
        "A prior default flag is present."
        if prior_default_flag == 1
        else "No prior default flag is present."
    )

    if decision == "Approve":
        recommendation = (
            "The application appears suitable for approval under the current policy thresholds."
        )
    elif decision == "Manual Review":
        recommendation = (
            "The application should be routed for manual review due to mixed underwriting signals."
        )
    else:
        recommendation = (
            "The application falls outside the current policy tolerance and is recommended for rejection."
        )

    summary = (
        f"This borrower is classified as **{risk_tier}** with a "
        f"loan-to-income ratio of **{loan_to_income:.2f}** and a predicted "
        f"probability of default of **{predicted_pd * 100:.1f}%**. "
        f"Repayment burden appears **{lti_text}** relative to income. "
        f"{prior_default_text} {recommendation}"
    )
    return summary


# -----------------------------
# Load model
# -----------------------------
try:
    model = get_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("Borrower / Loan Inputs")

    with st.form("underwriting_form"):
        st.markdown("### Repayment Capacity")
        person_income = st.number_input(
            "Annual Income ($)",
            min_value=1000,
            max_value=10000000,
            value=65000,
            step=1000,
        )

        person_emp_length = st.number_input(
            "Employment Length (Years)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
        )

        st.markdown("### Loan Structure")
        loan_amnt = st.number_input(
            "Loan Amount ($)",
            min_value=500,
            max_value=1000000,
            value=12000,
            step=500,
        )

        loan_int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.1,
            max_value=50.0,
            value=11.5,
            step=0.1,
        )

        loan_intent = st.selectbox(
            "Loan Intent",
            options=LOAN_INTENT_OPTIONS,
            index=1 if "MEDICAL" in LOAN_INTENT_OPTIONS else 0,
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            options=["A", "B", "C", "D", "E", "F", "G"],
            index=2,
        )

        st.markdown("### Credit Background")
        person_home_ownership = st.selectbox(
            "Home Ownership",
            options=HOME_OWNERSHIP_OPTIONS,
            index=0,
        )

        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (Years)",
            min_value=0,
            max_value=50,
            value=6,
            step=1,
        )

        cb_person_default_on_file = st.selectbox(
            "Prior Default on File",
            options=PRIOR_DEFAULT_OPTIONS,
            index=1,  # default N
        )

        run_assessment = st.form_submit_button("Run Underwriting Assessment")


with right_col:
    st.subheader("Underwriting Results")

    if not run_assessment:
        st.markdown(
            """
Enter borrower information on the left and click **Run Underwriting Assessment**  
to generate:
- underwriting metrics,
- predicted default probability,
- decision recommendation,
- expected loss estimate.
"""
        )
    else:
        try:
            # -----------------------------
            # Build borrower input
            # -----------------------------
            borrower_input = {
                "person_income": float(person_income),
                "person_emp_length": float(person_emp_length),
                "cb_person_cred_hist_length": int(cb_person_cred_hist_length),
                "loan_amnt": float(loan_amnt),
                "loan_int_rate": float(loan_int_rate),
                "person_home_ownership": person_home_ownership,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "cb_person_default_on_file": cb_person_default_on_file,
            }

            borrower_df = prepare_borrower_features(borrower_input)

            model_features = get_model_feature_columns()
            borrower_model_df = borrower_df[model_features]

            # -----------------------------
            # Score borrower
            # -----------------------------
            predicted_pd = predict_pd(model, borrower_model_df)

            risk_tier = borrower_df.loc[0, "risk_tier"]
            prior_default_flag = int(borrower_df.loc[0, "prior_default_flag"])
            loan_to_income = float(borrower_df.loc[0, "loan_to_income"])
            interest_burden = float(borrower_df.loc[0, "interest_burden_pct_income"])

            decision = assign_underwriting_decision(
                risk_tier=risk_tier,
                predicted_pd=predicted_pd,
                prior_default_flag=prior_default_flag,
            )

            decision_reason = summarize_decision_reason(
                risk_tier=risk_tier,
                predicted_pd=predicted_pd,
                loan_to_income=loan_to_income,
                prior_default_flag=prior_default_flag,
            )

            el_summary = build_expected_loss_summary(
                predicted_pd=predicted_pd,
                loan_amnt=float(loan_amnt),
            )

            # -----------------------------
            # Top metrics
            # -----------------------------
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Loan-to-Income", f"{loan_to_income:.2f}")
            m2.metric("Predicted PD", format_pct(predicted_pd))
            m3.metric("Risk Tier", risk_tier)
            m4.metric("Decision", decision)

            st.markdown("---")

            # -----------------------------
            # Supporting metrics
            # -----------------------------
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Interest Burden % Income", format_pct(interest_burden))
            s2.metric("Expected Loss", format_currency(el_summary["expected_loss"]))
            s3.metric("Loan Grade", loan_grade)
            s4.metric("Prior Default Flag", "Yes" if prior_default_flag == 1 else "No")

            st.markdown("---")

            # -----------------------------
            # Underwriting summary
            # -----------------------------
            st.markdown("### Underwriting Summary")
            summary_text = build_underwriting_summary(
                risk_tier=risk_tier,
                predicted_pd=predicted_pd,
                decision=decision,
                loan_to_income=loan_to_income,
                prior_default_flag=prior_default_flag,
            )
            st.markdown(summary_text)

            st.markdown("### Decision Drivers")
            st.markdown(f"- {decision_reason}")
            st.markdown(
                f"- Expected loss under current assumptions is **{format_currency(el_summary['expected_loss'])}**."
            )

            st.markdown("---")

            # -----------------------------
            # Detailed borrower metrics table
            # -----------------------------
            st.markdown("### Borrower Metrics")
            display_df = pd.DataFrame(
                {
                    "Metric": [
                        "Annual Income",
                        "Loan Amount",
                        "Interest Rate",
                        "Employment Length",
                        "Credit History Length",
                        "Loan-to-Income Ratio",
                        "Interest Burden % Income",
                        "Predicted PD",
                        "Risk Tier",
                        "Underwriting Decision",
                        "Expected Loss",
                    ],
                    "Value": [
                        format_currency(float(person_income)),
                        format_currency(float(loan_amnt)),
                        f"{float(loan_int_rate):.2f}%",
                        f"{float(person_emp_length):.1f} years",
                        f"{int(cb_person_cred_hist_length)} years",
                        f"{loan_to_income:.2f}",
                        format_pct(interest_burden),
                        format_pct(predicted_pd),
                        risk_tier,
                        decision,
                        format_currency(el_summary["expected_loss"]),
                    ],
                }
            )

            st.dataframe(display_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Unable to run underwriting assessment: {e}")