"""Data layer: load and clean the raw churn CSV, expose the cleaned DataFrame
and a documented log of every transformation and why it was applied."""


def load_data(path: str = "data/Customer-Churn.csv"):
    raise NotImplementedError


def clean_data(df):
    raise NotImplementedError
