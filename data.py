"""Data layer: load and clean the raw churn CSV, expose the cleaned DataFrame
and a documented log of every transformation and why it was applied."""

import pandas as pd

DEFAULT_DATA_PATH = "data/Customer-Churn.csv"

# Columns with a "No internet service" value that semantically means "No"
# when the customer has no internet. These get recoded during cleaning.
INTERNET_ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV with no transformations. Read-only entry point."""
    return pd.read_csv(path)


def audit_data(df: pd.DataFrame) -> dict:
    """Run structural checks against the raw dataframe and return a structured
    report of findings. Pure inspection - mutates nothing."""
    report = {
        "shape": list(df.shape),
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_ids": int(df["customerID"].nunique()),
        "class_balance": df["Churn"].value_counts(normalize=True).round(4).to_dict(),
        "class_counts": df["Churn"].value_counts().to_dict(),
        "nulls_after_coercion": {},
        "encoding_traps": {},
        "leakage_candidates": [],
        "numeric_ranges": {},
    }

    # dtype trap: TotalCharges is read as text; blank cells become real NaNs
    # only after numeric coercion.
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    report["totalcharges_dtype_original"] = str(df["TotalCharges"].dtype)
    report["totalcharges_blank_count"] = int(tc.isna().sum())
    report["totalcharges_blank_tenure"] = sorted(
        df.loc[tc.isna(), "tenure"].astype(int).unique().tolist()
    )
    report["nulls_after_coercion"]["TotalCharges"] = int(tc.isna().sum())

    # encoding trap: "No internet service" vs "No" for internet add-ons
    for col in INTERNET_ADDON_COLS:
        vals = df[col].value_counts().to_dict()
        report["encoding_traps"][col] = vals

    report["encoding_traps"]["MultipleLines"] = df["MultipleLines"].value_counts().to_dict()

    # categorical value sanity
    report["gender_values"] = df["gender"].value_counts().to_dict()
    report["contract_values"] = df["Contract"].value_counts().to_dict()
    report["payment_values"] = df["PaymentMethod"].value_counts().to_dict()

    # numeric ranges
    report["numeric_ranges"]["tenure"] = [int(df["tenure"].min()), int(df["tenure"].max())]
    report["numeric_ranges"]["MonthlyCharges"] = [
        round(float(df["MonthlyCharges"].min()), 2),
        round(float(df["MonthlyCharges"].max()), 2),
    ]
    report["numeric_ranges"]["TotalCharges"] = [
        round(float(tc.min()), 2),
        round(float(tc.max()), 2),
    ]

    # explicit leakage hunt: any column that could encode the label after the fact
    leakage_names = ["cancellation_date", "cancel", "days_since_last_login", "account_status", "churn_date"]
    report["leakage_candidates"] = [
        c for c in df.columns if any(k in c.lower() for k in leakage_names)
    ]

    # whitespace hygiene check
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