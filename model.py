"""Model layer: train/load the churn model and expose a single callable
prediction function. No notebook-only logic lives here."""


def predict_churn_risk(customer_id_or_features) -> dict:
    raise NotImplementedError
