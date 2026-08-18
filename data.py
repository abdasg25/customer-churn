"""Load and clean the churn CSV."""

import pandas as pd

DATA_PATH = "data/Customer-Churn.csv"

# columns where "No internet service" just means the customer has no internet,
# so it's really the same as "No"
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def audit_data(df):
    """Run some structural checks on the raw frame and return a dict of
    findings. Doesn't mutate anything."""
    report = {
        "shape": list(df.shape),
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_ids": int(df["customerID"].nunique()),
        "class_balance": df["Churn"].value_counts(normalize=True).round(4).to_dict(),
        "class_counts": df["Churn"].value_counts().to_dict(),
        "leakage_candidates": [],
        "encoding_traps": {},
        "numeric_ranges": {},
    }

    # total charges comes in as text, not numeric. the blank cells only show up
    # as NaN after we coerce it
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    report["totalcharges_dtype_original"] = str(df["TotalCharges"].dtype)
    report["totalcharges_blank_count"] = int(tc.isna().sum())
    report["totalcharges_blank_tenure"] = sorted(
        df.loc[tc.isna(), "tenure"].astype(int).unique().tolist()
    )

    # near-duplicate rows: same features, different customer id
    feat = df.drop(columns=["customerID"])
    report["near_duplicate_rows"] = int(feat.duplicated().sum())

    # "No internet service" vs "No" trap
    for col in INTERNET_ADDONS:
        report["encoding_traps"][col] = df[col].value_counts().to_dict()
    report["encoding_traps"]["MultipleLines"] = df["MultipleLines"].value_counts().to_dict()

    report["gender_values"] = df["gender"].value_counts().to_dict()
    report["contract_values"] = df["Contract"].value_counts().to_dict()
    report["payment_values"] = df["PaymentMethod"].value_counts().to_dict()

    report["numeric_ranges"]["tenure"] = [int(df["tenure"].min()), int(df["tenure"].max())]
    report["numeric_ranges"]["MonthlyCharges"] = [
        round(float(df["MonthlyCharges"].min()), 2),
        round(float(df["MonthlyCharges"].max()), 2),
    ]
    report["numeric_ranges"]["TotalCharges"] = [
        round(float(tc.min()), 2),
        round(float(tc.max()), 2),
    ]

    # explicit leakage hunt: names that would encode the label after the fact
    leakage_names = ["cancellation_date", "cancel", "days_since_last_login",
                     "account_status", "churn_date"]
    report["leakage_candidates"] = [
        c for c in df.columns if any(k in c.lower() for k in leakage_names)
    ]

    # trailing/leading whitespace in any string column
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    report["stripped_counts"] = {
        c: int((df[c].astype(str).str.strip() != df[c].astype(str)).sum())
        for c in str_cols
        if int((df[c].astype(str).str.strip() != df[c].astype(str)).sum()) > 0
    }

    return report


if __name__ == "__main__":
    import json

    raw = load_data()
    report = audit_data(raw)
    print(json.dumps(report, indent=2, default=str))