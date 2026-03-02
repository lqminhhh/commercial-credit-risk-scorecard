import streamlit as st
from pathlib import Path
from src.feature_engineering import prepare_borrower_features
from src.data_loader import load_dataset
from src.modeling import train_pd_model, predict_pd
from src.expected_loss import build_expected_loss_summary
from src.utils import format_currency, format_pct

def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.set_page_config(page_title="Policy Sensitivity", layout="wide")

st.title("Policy Sensitivity / What-If Analysis")
st.markdown(
    """
This page lets you test how underwriting decisions change when policy thresholds
or loss assumptions are adjusted.
"""
)


@st.cache_resource
def get_trained_model():
    df_raw = load_dataset("credit_risk_dataset.csv")
    model, metrics, _ = train_pd_model(df_raw)
    return model, metrics


def assign_custom_risk_tier(loan_to_income: float, low_cutoff: float, high_cutoff: float) -> str:
    if loan_to_income < low_cutoff:
        return "Low Risk"
    if loan_to_income < high_cutoff:
        return "Medium Risk"
    return "High Risk"


def assign_custom_decision(
    risk_tier: str,
    predicted_pd: float,
    prior_default_flag: int,
    approve_pd_cutoff: float,
    reject_pd_cutoff: float,
) -> str:
    if risk_tier == "High Risk" or predicted_pd >= reject_pd_cutoff:
        return "Reject"
    if risk_tier == "Low Risk" and predicted_pd < approve_pd_cutoff and prior_default_flag == 0:
        return "Approve"
    return "Manual Review"


try:
    model, model_metrics = get_trained_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


left, right = st.columns([1, 1.2])

with left:
    st.subheader("Borrower Inputs")

    person_income = st.number_input("Annual Income ($)", min_value=1000, value=65000, step=1000)
    person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, value=5.0, step=1.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, value=6, step=1)
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=12000, step=500)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.1, value=11.5, step=0.1)

    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_intent = st.selectbox(
        "Loan Intent",
        ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
    )
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)
    cb_person_default_on_file = st.selectbox("Prior Default on File", ["N", "Y"], index=0)

    st.markdown("### Policy Controls")
    low_risk_lti_cutoff = st.slider("Low-Risk LTI Cutoff", 0.05, 0.30, 0.15, 0.01)
    high_risk_lti_cutoff = st.slider("High-Risk LTI Cutoff", 0.20, 0.60, 0.35, 0.01)
    approve_pd_cutoff = st.slider("Approve PD Cutoff", 0.05, 0.30, 0.15, 0.01)
    reject_pd_cutoff = st.slider("Reject PD Cutoff", 0.15, 0.60, 0.35, 0.01)
    lgd = st.slider("LGD Assumption", 0.20, 0.80, 0.45, 0.01)

with right:
    st.subheader("Sensitivity Results")

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

    try:
        borrower_df = prepare_borrower_features(borrower_input)

        model_features = [
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

        borrower_model_df = borrower_df[model_features]
        predicted_pd = predict_pd(model, borrower_model_df)

        loan_to_income = float(borrower_df.loc[0, "loan_to_income"])
        prior_default_flag = int(borrower_df.loc[0, "prior_default_flag"])

        custom_risk_tier = assign_custom_risk_tier(
            loan_to_income,
            low_cutoff=low_risk_lti_cutoff,
            high_cutoff=high_risk_lti_cutoff,
        )

        custom_decision = assign_custom_decision(
            risk_tier=custom_risk_tier,
            predicted_pd=predicted_pd,
            prior_default_flag=prior_default_flag,
            approve_pd_cutoff=approve_pd_cutoff,
            reject_pd_cutoff=reject_pd_cutoff,
        )

        el_summary = build_expected_loss_summary(
            predicted_pd=predicted_pd,
            loan_amnt=float(loan_amnt),
            lgd=float(lgd),
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Loan-to-Income", f"{loan_to_income:.2f}")
        r2.metric("Predicted PD", format_pct(predicted_pd))
        r3.metric("Risk Tier", custom_risk_tier)
        r4.metric("Decision", custom_decision)

        st.markdown("---")

        s1, s2 = st.columns(2)
        s1.metric("Expected Loss", format_currency(el_summary["expected_loss"]))
        s2.metric("LGD Assumption", format_pct(lgd))

        st.markdown("---")

        st.markdown("### Policy Interpretation")
        st.markdown(
            f"""
- Under the current policy settings, this borrower is classified as **{custom_risk_tier}**.
- The predicted default probability is **{format_pct(predicted_pd)}**.
- Based on the selected approval and rejection thresholds, the recommended action is **{custom_decision}**.
- Under the chosen LGD assumption, expected loss is **{format_currency(el_summary['expected_loss'])}**.
"""
        )

    except Exception as e:
        st.error(f"Unable to evaluate policy sensitivity: {e}")