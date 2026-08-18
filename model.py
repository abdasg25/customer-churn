"""Train/load the churn model, expose predict_churn_risk()."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


POS_LABEL = 1  # churn is the positive class


def classification_report_churn(y_true, y_proba, threshold=0.5):
    """metrics for the churn class, the only ones that matter here.

    accuracy is left out on purpose - a majority-class dummy scores 73.5% on
    this data while catching zero churners, so it'd be misleading."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, pos_label=POS_LABEL)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POS_LABEL)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POS_LABEL)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def majority_class_baseline(y_true):
    """what predicting 'no churn' for every customer gives - the null model."""
    return {
        "accuracy": float((y_true == 0).mean()),
        "recall_churn": 0.0,
        "f1_churn": 0.0,
    }


def train_logistic_baseline(X_train_enc, y_train, X_test_enc, y_test, threshold=0.5):
    """logistic regression + scaling as the linear baseline.

    trees don't care about feature scale, logistic does, so the scaler lives
    here rather than in the shared preprocessor."""
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    clf.fit(X_train_enc, y_train)
    proba = clf.predict_proba(X_test_enc)[:, 1]
    return classification_report_churn(y_test, proba, threshold=threshold)


def predict_churn_risk(customer_id_or_features) -> dict:
    raise NotImplementedError