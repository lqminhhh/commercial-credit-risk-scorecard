# Credit Risk Underwriting App

__Author:__ Minh Le

A portfolio project that simulates a **commercial loan underwriting workflow** using borrower-level credit data, **probability of default (PD)** modeling, underwriting decision rules, and **expected loss (EL)** estimation.

The project combines a credit risk analysis notebook with a simple **Streamlit app** that helps evaluate borrower risk and support **Approve / Manual Review / Reject** decisions.

---

## What this project does

- Engineers practical underwriting metrics such as:
  - **Loan-to-Income Ratio (LTI)**
  - **Interest Burden as % of Income**
  - **Risk Tier (Low / Medium / High)**
- Trains a **logistic regression PD model**
- Simulates underwriting decisions using policy-style thresholds
- Estimates **Expected Loss** using:

$$
EL = PD \times LGD \times EAD
$$

- Provides an interactive app with:
  - **Portfolio Overview**
  - **Borrower Underwriting Tool**
  - **Policy Sensitivity**

---

## Repository Structure

```text
├── app.py
├── assets/
├── data/
├── models/
├── notebooks/
├── pages/
├── scripts/
└── src/
```

## Key Results

1. Built an interpretable PD model with approximately:
- Accuracy: 0.86
- ROC AUC: 0.87

2. Created a practical underwriting framework that combines:
- borrower repayment capacity
- default risk
- expected loss

## Run Locally

1. Install dependencies

```{python}
python3 -m pip install -r requirements.txt
```

2. Train and save the model

```{python}
python3 scripts/train_pd_model.py
```

3. Launch the app

```{python}
python3 -m streamlit run app.py
```

## Tech Stack

Python, pandas, numpy, scikit-learn, Streamlit, matplotlib, joblib