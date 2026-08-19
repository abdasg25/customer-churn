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

    raw data + numeric TotalCharges, plus a model-computed risk_score column so
    the agent can filter/aggregate by predicted churn risk. categoricals keep
    their original values (e.g. 'Month-to-month' not 0)."""
    if "df" not in _df_cache:
        from data import load_data
        from model import bulk_risk_scores

        raw = load_data()
        raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0)
        raw["risk_score"] = bulk_risk_scores().reindex(raw["customerID"]).values
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


def run_data_query(args):
    """execute a pandas snippet against the dataset in a locked namespace.

    args = {"code": "..."}. available names: df, pd, np, safe builtins.
    returns {status, output, ...} - errors are captured, never raised."""
    code = (args or {}).get("code", "")
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
    """dispatch predict_churn_risk on customer_id, a features dict, or overrides."""
    from model import predict_churn_risk

    if args.get("customer_id"):
        return predict_churn_risk(args["customer_id"], overrides=args.get("overrides"))
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
                "Predict churn risk for one customer: pass EITHER customer_id (string) "
                "OR features (dict). Optionally pass overrides (dict) with a customer_id "
                "to project that customer forward under different feature values. "
                "Returns risk_score (0-1), prediction_class, top_factors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "existing customerID"},
                    "features": {
                        "type": "object",
                        "description": "all raw feature values (see get_dataset_schema for names)",
                    },
                    "overrides": {
                        "type": "object",
                        "description": "feature changes to project a customer forward, e.g. {\"Contract\": \"Two year\"}",
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
                "Run a pandas expression against 'df' (plus pd, np). For filters, group-bys, "
                "aggregations, correlations. Churn is 'Yes'/'No', risk_score is 0-1. "
                "Call get_dataset_schema if unsure of columns. Returns result or an error."
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
