"""Train/load the churn model, expose predict_churn_risk()."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from data import CATEGORICAL_COLS

MODEL_PATH = "models/churn_model.joblib"
PREPROC_PATH = "models/preprocessor.joblib"

# numeric columns are everything clean that isn't categorical / id / target
NUMERIC_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges",
]


def build_preprocessor():
    # one-hot the categoricals, pass everything else through untouched.
    # handle_unknown="ignore" so a new category at predict time doesn't crash
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)],
        remainder="passthrough",
    )


def encode(X_train, X_test):
    """fit the one-hot encoder on train only, transform both. no leakage."""
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    return Xtr, Xte, pre


def predict_churn_risk(customer_id_or_features) -> dict:
    raise NotImplementedError