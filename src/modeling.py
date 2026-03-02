from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def get_model_feature_columns() -> list[str]:
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


def clean_credit_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Remove rows with unrealistic ages
    df = df[(df['person_age'] >= 18) & (df['person_age'] <= 100)]

    # Winsorize extreme values in income and loan amount
    for col in ['person_income', 'loan_amnt']:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    # Cap employment length at 40 years
    df['person_emp_length'] = df['person_emp_length'].clip(upper=40)

    # Clean impossible interest rates but keep NaN values for imputation
    df.loc[(df["loan_int_rate"] <= 0) | (df["loan_int_rate"] >= 40), "loan_int_rate"] = np.nan

    return df


def engineer_training_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["loan_to_income"] = out["loan_amnt"] / out["person_income"]

    out["prior_default_flag"] = (
        out["cb_person_default_on_file"].astype(str).str.upper() == "Y"
    ).astype(int)

    grade_map = {grade: i for i, grade in enumerate(list("ABCDEFG"), start=1)}
    out["loan_grade_num"] = out["loan_grade"].map(grade_map)

    out["interest_burden_pct_income"] = (
        out["loan_amnt"] * (out["loan_int_rate"] / 100.0)
    ) / out["person_income"]

    # Risk tiers based on Loan-to-Income Ratio
    def assign_risk_tier(lti):
        if pd.isna(lti):
            return "Unknown"
        elif lti < 0.15:
            return "Low Risk"
        elif lti < 0.35:
            return "Medium Risk"
        else:
            return "High Risk"

    out["risk_tier"] = out["loan_to_income"].apply(assign_risk_tier)

    return out


def build_training_pipeline() -> Pipeline:
    feature_num = [
        "person_income",
        "person_emp_length",
        "cb_person_cred_hist_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_to_income",
        "interest_burden_pct_income",
        "loan_grade_num",
        "prior_default_flag",
    ]

    feature_cat = [
        "person_home_ownership",
        "loan_intent",
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_num),
            ("cat", categorical_transformer, feature_cat),
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            solver="liblinear",
            max_iter=500,
        )),
    ])

    return clf


def train_pd_model(df_raw: pd.DataFrame):
    df = clean_credit_data(df_raw)
    df = engineer_training_features(df)

    target = "loan_status"
    feature_cols = get_model_feature_columns()

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_training_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    return model, metrics, df


def predict_pd(model, borrower_df: pd.DataFrame) -> float:
    if borrower_df.empty:
        raise ValueError("borrower_df is empty. Cannot generate prediction.")

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba().")

    pd_score = model.predict_proba(borrower_df)[0, 1]
    return float(pd_score)