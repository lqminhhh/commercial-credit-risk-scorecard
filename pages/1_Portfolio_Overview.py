import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_dataset, load_model
from src.feature_engineering import (
    map_loan_grade,
    map_prior_default_flag,
    assign_risk_tier,
)
from src.modeling import predict_pd
from src.expected_loss import calculate_expected_loss
from src.utils import format_currency, format_pct

def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.set_page_config(page_title="Portfolio Overview", layout="wide")

st.title("Portfolio Overview")
st.markdown(
    """
This page summarizes the portfolio used in the underwriting project, including
overall credit quality, risk segmentation, model-estimated default risk, and expected loss.
"""
)


@st.cache_data
def get_clean_portfolio():
    df = load_dataset("credit_risk_dataset.csv").copy()

    # Basic cleaning aligned with training script
    df = df[(df["person_age"] >= 18) & (df["person_age"] <= 75)].copy()
    df = df[df["person_income"] > 0].copy()

    income_cap = df["person_income"].quantile(0.99)
    df["person_income"] = df["person_income"].clip(upper=income_cap)

    df["person_emp_length"] = df["person_emp_length"].clip(lower=0, upper=40)
    df.loc[(df["loan_int_rate"] <= 0) | (df["loan_int_rate"] >= 40), "loan_int_rate"] = pd.NA

    # Feature engineering
    df["loan_to_income"] = df["loan_amnt"] / df["person_income"]
    df["interest_burden_pct_income"] = (
        df["loan_amnt"] * (df["loan_int_rate"] / 100.0)
    ) / df["person_income"]
    df["loan_grade_num"] = df["loan_grade"].map(map_loan_grade)
    df["prior_default_flag"] = df["cb_person_default_on_file"].map(map_prior_default_flag)
    df["risk_tier"] = df["loan_to_income"].apply(assign_risk_tier)

    return df


@st.cache_resource
def get_model():
    return load_model("pd_model.pkl")


def score_portfolio(df: pd.DataFrame, model) -> pd.DataFrame:
    scored = df.copy()

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

    clean_for_model = scored[model_features].copy()
    scored["predicted_pd"] = model.predict_proba(clean_for_model)[:, 1]
    scored["expected_loss"] = scored.apply(
        lambda row: calculate_expected_loss(
            predicted_pd=row["predicted_pd"],
            loan_amnt=row["loan_amnt"],
        ),
        axis=1,
    )

    def decision(row):
        if row["risk_tier"] == "High Risk" or row["predicted_pd"] >= 0.35:
            return "Reject"
        if (
            row["risk_tier"] == "Low Risk"
            and row["predicted_pd"] < 0.15
            and row["prior_default_flag"] == 0
        ):
            return "Approve"
        return "Manual Review"

    scored["underwriting_decision"] = scored.apply(decision, axis=1)
    return scored


try:
    df = get_clean_portfolio()
    model = get_model()
    df = score_portfolio(df, model)
except Exception as e:
    st.error(f"Unable to load portfolio data or model: {e}")
    st.stop()


# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Borrowers", f"{len(df):,}")
k2.metric("Portfolio Default Rate", format_pct(df["loan_status"].mean()))
k3.metric("Average Income", format_currency(df["person_income"].mean()))
k4.metric("Average Loan Amount", format_currency(df["loan_amnt"].mean()))
k5.metric("Average PD", format_pct(df["predicted_pd"].mean()))
k6.metric("Total Expected Loss", format_currency(df["expected_loss"].sum()))

st.markdown("---")

# Risk tier summary
risk_summary = df.groupby("risk_tier").agg(
    borrower_count=("risk_tier", "count"),
    default_rate=("loan_status", "mean"),
    avg_pd=("predicted_pd", "mean"),
    total_expected_loss=("expected_loss", "sum"),
).reset_index()

decision_summary = df.groupby("underwriting_decision").agg(
    borrower_count=("underwriting_decision", "count"),
    default_rate=("loan_status", "mean"),
    avg_pd=("predicted_pd", "mean"),
    total_expected_loss=("expected_loss", "sum"),
).reset_index()

left, right = st.columns(2)

with left:
    st.subheader("Risk Tier Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(risk_summary["risk_tier"], risk_summary["borrower_count"])
    ax.set_title("Borrowers by Risk Tier")
    ax.set_xlabel("Risk Tier")
    ax.set_ylabel("Borrower Count")
    st.pyplot(fig)

with right:
    st.subheader("Default Rate by Risk Tier")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(risk_summary["risk_tier"], risk_summary["default_rate"] * 100)
    ax.set_title("Observed Default Rate by Risk Tier")
    ax.set_xlabel("Risk Tier")
    ax.set_ylabel("Default Rate (%)")
    st.pyplot(fig)

st.markdown("---")

left2, right2 = st.columns(2)

with left2:
    st.subheader("Expected Loss by Risk Tier")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(risk_summary["risk_tier"], risk_summary["total_expected_loss"])
    ax.set_title("Total Expected Loss by Risk Tier")
    ax.set_xlabel("Risk Tier")
    ax.set_ylabel("Expected Loss")
    st.pyplot(fig)

with right2:
    st.subheader("Expected Loss by Underwriting Decision")
    decision_order = ["Approve", "Manual Review", "Reject"]
    decision_plot = decision_summary.set_index("underwriting_decision").reindex(decision_order).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(decision_plot["underwriting_decision"], decision_plot["total_expected_loss"])
    ax.set_title("Total Expected Loss by Decision")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Expected Loss")
    st.pyplot(fig)

st.markdown("---")

st.subheader("Risk Tier Summary")
risk_display = risk_summary.copy()
risk_display["default_rate"] = risk_display["default_rate"].apply(format_pct)
risk_display["avg_pd"] = risk_display["avg_pd"].apply(format_pct)
risk_display["total_expected_loss"] = risk_display["total_expected_loss"].apply(format_currency)
st.dataframe(risk_display, use_container_width=True, hide_index=True)

st.subheader("Underwriting Decision Summary")
decision_display = decision_summary.copy()
decision_display["default_rate"] = decision_display["default_rate"].apply(format_pct)
decision_display["avg_pd"] = decision_display["avg_pd"].apply(format_pct)
decision_display["total_expected_loss"] = decision_display["total_expected_loss"].apply(format_currency)
st.dataframe(decision_display, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    """
### Portfolio Takeaways
- Higher-risk borrower segments should show higher observed default rates and higher expected loss.
- Expected loss helps connect borrower-level model outputs to portfolio-level economic risk.
- The underwriting decision buckets provide a practical view of how policy rules separate lower-risk and higher-risk applications.
"""
)