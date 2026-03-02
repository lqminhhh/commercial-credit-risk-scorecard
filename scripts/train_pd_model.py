from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix


def main():
    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "credit_risk_dataset.csv"
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pd_model.pkl"

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df_raw = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path}")
    print(f"Raw shape: {df_raw.shape}")

    # --------------------------------------------------
    # Data cleaning
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    df["loan_to_income"] = df["loan_amnt"] / df["person_income"]

    df["prior_default_flag"] = (
        df["cb_person_default_on_file"].astype(str).str.upper() == "Y"
    ).astype(int)

    grade_map = {grade: i for i, grade in enumerate(list("ABCDEFG"), start=1)}
    df["loan_grade_num"] = df["loan_grade"].map(grade_map)

    df["interest_burden_pct_income"] = (
        df["loan_amnt"] * (df["loan_int_rate"] / 100.0)
    ) / df["person_income"]

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

    df["risk_tier"] = df["loan_to_income"].apply(assign_risk_tier)
    
    target = "loan_status"

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

    X = df[feature_num + feature_cat]
    y = df[target]

    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # Preprocessing + model
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    clf.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nModel Evaluation")
    print("-" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    joblib.dump(clf, model_path)
    print(f"\nSaved model to: {model_path}")

    # Optional: save feature reference
    feature_reference = {
        "feature_num": feature_num,
        "feature_cat": feature_cat,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(acc),
        "roc_auc": float(auc),
    }

    feature_ref_path = model_dir / "pd_model_metadata.joblib"
    joblib.dump(feature_reference, feature_ref_path)
    print(f"Saved metadata to: {feature_ref_path}")


if __name__ == "__main__":
    main()