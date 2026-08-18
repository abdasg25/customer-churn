"""Tool layer: the only interface the agent is allowed to touch.

Wraps the model function and a restricted dataframe-query executor as
agent-callable tools with explicit names, descriptions, and JSON-serializable
outputs."""

import ast
import json

import numpy as np
import pandas as pd

# builtins the agent's generated code is allowed to use. anything not listed
# here raises NameError, so eval/open/__import__ etc. are unavailable by default.
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "isinstance": isinstance,
    "type": type,
}


_df_cache = {}


def _df():
    """lazy-load a readable dataframe for EDA queries.

    raw data + numeric TotalCharges, so categoricals keep their original
    values (e.g. 'Month-to-month' not 0). the model's encoded df stays inside
    model.py; this one is for the agent to describe/query the dataset."""
    if "df" not in _df_cache:
        from data import load_data

        raw = load_data()
        raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0)
        _df_cache["df"] = raw
    return _df_cache["df"]


def _check_ast(code):
    """reject the two obvious escape routes: imports and dunder attribute access.

    this is a take-home-level sandbox, not production hardening - see README."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is not allowed")


def _run(code, ns):
    """execute the snippet; return the value of the last expression if any."""
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        rest = ast.Module(body=tree.body[:-1], type_ignores=[])
        exec(compile(rest, "<query>", "exec"), ns)
        last = ast.Expression(body=tree.body[-1].value)
        return eval(compile(last, "<query>", "eval"), ns)
    exec(code, ns)
    return ns.get("result")


def _format(result):
    """stringify a result, truncating anything large."""
    if isinstance(result, pd.DataFrame):
        n = len(result)
        return f"{result.head(10).to_string()}\n[{n} rows x {result.shape[1]} cols, showing 10]"
    if isinstance(result, pd.Series):
        return result.head(20).to_string()
    if isinstance(result, (dict, list)):
        return json.dumps(result, default=str)[:2000]
    return str(result)[:2000]


def run_data_query(code):
    """execute a pandas snippet against the cleaned df in a locked namespace.

    available names: df (cleaned data), pd, np, and the safe builtins.
    returns {status, output, ...} - errors are captured, never raised."""
    if not isinstance(code, str) or not code.strip():
        return {"status": "error", "error": "empty query"}

    try:
        _check_ast(code)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    ns = {"df": _df(), "pd": pd, "np": np, "__builtins__": SAFE_BUILTINS}
    try:
        result = _run(code, ns)
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    info = {"status": "ok", "output": _format(result)}
    if isinstance(result, pd.DataFrame):
        info["shape"] = list(result.shape)
    elif isinstance(result, pd.Series):
        info["length"] = int(len(result))
    return info


def _call_predict(args):
    """dispatch predict_churn_risk on customer_id or a hypothetical features dict."""
    from model import predict_churn_risk

    if args.get("customer_id"):
        return predict_churn_risk(args["customer_id"])
    if args.get("features"):
        return predict_churn_risk(args["features"])
    raise ValueError("provide either customer_id or features")


def _call_schema(args=None):
    """describe the dataset so the agent doesn't guess column names."""
    df = _df()
    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": [
            {"name": c, "dtype": str(df[c].dtype), "sample": df[c].dropna().unique()[:5].tolist()}
            for c in df.columns
        ],
    }


# name -> python function. the agent only ever sees TOOLS (the schemas) + results.
TOOL_FUNCTIONS = {
    "predict_churn_risk": _call_predict,
    "run_data_query": run_data_query,
    "get_dataset_schema": _call_schema,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_churn_risk",
            "description": (
                "Predict the probability a customer churns. Pass EITHER customer_id "
                "(string, an existing customerID from the dataset) OR features (an object "
                "of the customer's attributes in raw schema). Returns risk_score (0-1), "
                "prediction_class ('Churn'/'Stay'), and top_factors (what drove the score)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "existing customerID to look up"},
                    "features": {
                        "type": "object",
                        "description": (
                            "full set of 19 raw features: gender, SeniorCitizen, Partner, "
                            "Dependents, tenure, PhoneService, MultipleLines, InternetService, "
                            "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, "
                            "StreamingTV, StreamingMovies, Contract, PaperlessBilling, "
                            "PaymentMethod, MonthlyCharges, TotalCharges"
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_data_query",
            "description": (
                "Run a pandas expression against the dataset (variable 'df', plus pd and np). "
                "Use for aggregations, filters, group-bys, correlations, distributions. "
                "The Churn column is 'Yes'/'No'. If unsure of column names, call "
                "get_dataset_schema first. Returns the result string or an error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "pandas expression to evaluate"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_schema",
            "description": "Return the dataset's columns, dtypes, sample values, and row count.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
