"""Train/load the churn model, expose predict_churn_risk()."""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data import CATEGORICAL_COLS, CONTRACT_ORDER, clean_data, load_data, split_data

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


def train_model(X_train_enc, y_train, X_test_enc, y_test, threshold=0.5):
    """xgboost with scale_pos_weight for the imbalance + a small CV search.

    search is run on the train set only (cv folds inside train), test is held
    out untouched until the end."""
    scale = float((y_train == 0).sum() / (y_train == 1).sum())

    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
    }
    base = XGBClassifier(
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    search = RandomizedSearchCV(
        base, param_grid, n_iter=15, cv=3, scoring="f1", random_state=42,
    )
    search.fit(X_train_enc, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(X_test_enc)[:, 1]
    metrics = classification_report_churn(y_test, proba, threshold=threshold)
    metrics["best_params"] = search.best_params_
    metrics["scale_pos_weight"] = round(scale, 3)
    return best, metrics


def confusion_matrix_churn(y_true, y_proba, threshold=0.5):
    """tn/fp/fn/tp at a threshold - read it as a business table, not a grid."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def best_threshold(y_true, y_proba):
    """the threshold that maximises f1, from the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[: len(thresholds)], recall[: len(thresholds)]
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def feature_importance_global(model, feature_names, top_k=None):
    """gain-based feature importance from xgboost, sorted by importance."""
    pairs = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda t: t[1], reverse=True,
    )
    if top_k:
        pairs = pairs[:top_k]
    return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs]


def _shap_top_factors(explainer, X_row, feature_names, top_k):
    """shap values -> top contributions (log-odds), shared by both explain paths."""
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]
    contribs = sorted(
        zip(feature_names, sv[0]),
        key=lambda t: abs(float(t[1])), reverse=True,
    )[:top_k]
    return [{"feature": f, "contribution": round(float(c), 4)} for f, c in contribs]


def _prettify(feature):
    """turn encoded feature names back into readable ones.

    remainder__Contract -> Contract, cat__InternetService_Fiber optic ->
    InternetService = Fiber optic."""
    if feature.startswith("remainder__"):
        return feature.split("__", 1)[1]
    if feature.startswith("cat__"):
        rest = feature.split("__", 1)[1]
        col, val = rest.split("_", 1)
        return f"{col} = {val}"
    return feature


def explain_prediction(model, X_row, feature_names, top_k=5):
    """standalone shap helper (notebook) - builds its own explainer."""
    import shap  # lazy: only needed when we actually explain something

    return _shap_top_factors(shap.TreeExplainer(model), X_row, feature_names, top_k)


# ---- model-as-tool: predict_churn_risk is the only public entry point ----

_state = {}


def save_artifacts(model, pre):
    import os

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(pre, PREPROC_PATH)


def load_state():
    """lazy-load the trained model + preprocessor + cleaned df, cached once."""
    if not _state:
        _state["model"] = joblib.load(MODEL_PATH)
        _state["pre"] = joblib.load(PREPROC_PATH)
        _state["clean_df"] = clean_data(load_data())[0]
        _state["feature_cols"] = [
            c for c in _state["clean_df"].columns if c not in ("customerID", "Churn")
        ]
        _state["feature_names"] = _state["pre"].get_feature_names_out()
    return _state


def _get_explainer(state):
    # built lazily: bulk prediction doesn't need it, per-row explanation does
    if "explainer" not in state:
        import shap

        state["explainer"] = shap.TreeExplainer(state["model"])
    return state["explainer"]


def bulk_risk_scores():
    """predicted churn probability for every customer (no SHAP, one batched call)."""
    state = load_state()
    X = state["clean_df"][state["feature_cols"]]
    X_enc = state["pre"].transform(X)
    proba = state["model"].predict_proba(X_enc)[:, 1]
    return pd.Series(proba, index=state["clean_df"]["customerID"].values, name="risk_score")


def _row_from_dict(features, feature_cols):
    """build a clean feature row from a hypothetical dict in raw CSV schema."""
    missing = [c for c in feature_cols if c not in features]
    if missing:
        raise ValueError(f"missing features: {missing}")
    if features.get("Contract") not in CONTRACT_ORDER:
        raise ValueError(
            f"Contract must be one of {list(CONTRACT_ORDER)}, got {features.get('Contract')!r}"
        )
    # clean_data wants a Churn column; add a dummy and drop it right after
    row = pd.DataFrame([{**features, "Churn": "No"}])
    clean_row, _ = clean_data(row)
    return clean_row[feature_cols]


def predict_churn_risk(customer_id_or_features, top_k=5) -> dict:
    """predict churn risk for an existing customerID or a hypothetical dict.

    returns {customer_id, risk_score, prediction_class, top_factors}. risk is
    clamped to [0,1] so a wonky model output can't reach the agent as garbage."""
    state = load_state()

    if isinstance(customer_id_or_features, str):
        customer_id = customer_id_or_features
        row = state["clean_df"][state["clean_df"]["customerID"] == customer_id]
        if row.empty:
            raise ValueError(f"customerID {customer_id} not found")
        X = row[state["feature_cols"]]
    elif isinstance(customer_id_or_features, dict):
        customer_id = None
        X = _row_from_dict(customer_id_or_features, state["feature_cols"])
    else:
        raise TypeError("expected a customerID string or a feature dict")

    X_enc = state["pre"].transform(X)
    risk = float(state["model"].predict_proba(X_enc)[:, 1][0])
    risk = max(0.0, min(1.0, risk))

    factors = _shap_top_factors(_get_explainer(state), X_enc, state["feature_names"], top_k)
    for f in factors:
        f["feature"] = _prettify(f["feature"])

    return {
        "customer_id": customer_id,
        "risk_score": round(risk, 4),
        "prediction_class": "Churn" if risk >= 0.5 else "Stay",
        "top_factors": factors,
    }


def train_and_save():
    """one-shot: clean -> split -> encode -> train -> persist. for the app."""
    clean, _ = clean_data(load_data())
    Xtr, Xte, ytr, yte = split_data(clean)
    Xtr_enc, Xte_enc, pre = encode(Xtr, Xte)
    model, metrics = train_model(Xtr_enc, ytr, Xte_enc, yte)
    save_artifacts(model, pre)
    return metrics