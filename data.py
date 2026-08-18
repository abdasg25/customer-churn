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

    # total charges comes in as text, not numeric. the blank cells only show up as NaN after we coerce it
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

    # names that encode the label after the fact
    leakage_names = ["cancellation_date", "cancel", "days_since_last_login",
                     "account_status", "churn_date"]
    report["leakage_candidates"] = [
        c for c in df.columns if any(k in c.lower() for k in leakage_names)
    ]

    # trailing whitespace in any string column
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    report["stripped_counts"] = {
        c: int((df[c].astype(str).str.strip() != df[c].astype(str)).sum())
        for c in str_cols
        if int((df[c].astype(str).str.strip() != df[c].astype(str)).sum()) > 0
    }

    return report


def churn_rate_by(df, col):
    """churn rate + counts per category, sorted by rate."""
    out = (
        df.groupby(col)["Churn"]
        .agg(total="count", churned=lambda s: (s == "Yes").sum())
        .assign(churn_rate=lambda d: d.churned / d.total)
        .sort_values("churn_rate", ascending=False)
    )
    return out


def corr_with_churn(df):
    """correlation of numeric columns vs churn (Yes=1), abs sorted."""
    num = df.select_dtypes(include="number").copy()
    num["churn"] = (df["Churn"] == "Yes").astype(int)
    corr = num.corr()["churn"].drop("churn")
    return corr.abs().sort_values(ascending=False)


# Yes/No columns that become 0/1
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# contract is ordinal: month-to-month is the riskiest, two-year the safest
CONTRACT_ORDER = {"Month-to-month": 0, "One year": 1, "Two year": 2}

# columns that stay categorical and get one-hot encoded in the model layer
CATEGORICAL_COLS = ["InternetService", "MultipleLines", "PaymentMethod"]


def clean_data(df):
    """Apply the documented cleaning decisions. Returns (clean_df, log).

    Every rule here is a hardcoded mapping, no fitting/statistics, so it is
    safe to run on the whole frame before any train/test split (no leakage)."""
    out = df.copy()
    log = []

    # total charges: coerce to numeric, fill the 11 tenure==0 blanks with 0
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    n_blank = int(out["TotalCharges"].isna().sum())
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)
    if n_blank:
        log.append({
            "issue": f"TotalCharges is text with {n_blank} blanks (all tenure==0)",
            "action": "coerce to numeric, fill blanks with 0",
            "reason": "new customers with no completed billing cycle yet",
        })

    # internet add-ons: "No internet service" is really just "No", then 0/1
    n_addons = 0
    for col in INTERNET_ADDONS:
        n_addons += int((out[col] == "No internet service").sum())
        out[col] = out[col].replace("No internet service", "No")
        out[col] = (out[col] == "Yes").astype(int)
    if n_addons:
        log.append({
            "issue": f"{n_addons} 'No internet service' values across 6 add-on columns",
            "action": "recode to 'No', then Yes/No -> 1/0",
            "reason": "semantically identical: customer has no internet",
        })

    # target and the other binary Yes/No columns
    out["Churn"] = (out["Churn"] == "Yes").astype(int)
    for col in BINARY_COLS:
        out[col] = (out[col] == "Yes").astype(int)

    # gender -> 0/1 (which side is 1 is arbitrary, just keep it consistent)
    out["gender"] = (out["gender"] == "Male").astype(int)

    # contract ordinal
    out["Contract"] = out["Contract"].map(CONTRACT_ORDER)
    log.append({
        "issue": "Contract is ordinal (month-to-month < one year < two year)",
        "action": "map to 0/1/2",
        "reason": "ordering correlates with churn risk, a plain dummy would lose that",
    })

    return out, log


def split_data(df, test_size=0.2, seed=42):
    """stratified train/test split on Churn.

    X keeps the categoricals as strings; encoding happens in model.py where
    the one-hot encoder is fit on train only. stratify keeps the 26.5% churn
    ratio in both halves so a small fold can't accidentally drop churners."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import json

    raw = load_data()
    report = audit_data(raw)
    print(json.dumps(report, indent=2, default=str))

    clean, log = clean_data(raw)
    print("\ncleaning log:")
    for entry in log:
        print(" -", entry["issue"], "=>", entry["action"])
    print("\nclean dtypes:")
    print(clean.dtypes.to_string())