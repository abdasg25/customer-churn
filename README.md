# Autonomous Data Analyst — Customer Churn

A take-home assessment for Adept Tech Solutions. An agentic chat app over a
customer-churn dataset. The user asks natural-language questions; the agent
plans, computes via tools, and self-checks — **every number in an answer traces
back to a real tool call**, never an LLM guess.

## What it can answer

- **EDA questions** — "what is the churn rate by contract type?", "which features correlate most with churn?"
- **Per-customer risk** — "what's the churn risk for customer 7590-VHVEG?", with the top factors (SHAP) that drove the score.
- **Hypothetical data points** — submit a full feature dict and see how the model scores it against the original data.
- **Projected forward** — take an existing customer and override feature values ("7590-VHVEG if they moved to a two-year contract") to see the counterfactual risk.
- **Aggregate risk across segments** — "average monthly charges for customers with risk > 0.7?" (the dataset exposes a model-computed `risk_score` column).

## Architecture

```
app.py      Streamlit chat UI (renders answers + a collapsible reasoning trace)
  -> agent.py   plan -> act -> observe -> verify loop; anti-hallucination gate
       -> llm.py    Groq client (native tool-calling, 429 retry)
       -> tools.py  the 3 agent-callable tools (see below)
            -> model.py  XGBoost model + predict_churn_risk()
            -> data.py   load/clean/split the CSV
```

Layers depend in one direction only; the UI never touches the model or tools directly.

## The data — what was wrong and how I handled it

1. **`TotalCharges` stored as text** with 11 blank cells. All 11 blanks are the
   `tenure == 0` rows (brand-new customers, no bill yet) and all are `Churn = No`
   — *structurally* missing (MNAR), not random. Fixed by coercing to numeric and
   filling those 11 with `0`.
2. **Encoding trap**: six internet add-on columns (`OnlineSecurity`, …) contain
   a `"No internet service"` value that's semantically just `"No"`; `MultipleLines`
   has `"No phone service"`. Recoded `"No internet service" -> "No"` and then to
   0/1; `MultipleLines` keeps its distinct `"No phone service"` category.
3. **Near-duplicate rows**: 22 rows share identical features across 20 groups
   (distinct `customerID`s). Labels are consistent within every group, so they're
   coincidental duplicates — kept, not dropped.
4. **No label leakage**: searched explicitly for `cancellation_date`,
   `days_since_last_login`, `account_status`, etc. — none present. `customerID`
   is a pure identifier, so it's excluded from features but kept for lookup.

Full rationale lives in `DATA_NOTES.md`.

## Model and metric

**Model**: XGBoost with `scale_pos_weight = 2.77` (the ~74/26 class ratio),
tuned with a small `RandomizedSearchCV` (`f1` scoring, train folds only).
Baselines for reference: a majority-class dummy and a class-weighted logistic
regression.

| Model | F1 (churn) | PR-AUC | Recall |
|---|---|---|---|
| Majority-class dummy | 0.0 | — | 0.0 |
| Logistic regression | 0.613 | 0.632 | 0.781 |
| **XGBoost** | **0.630** | **0.661** | 0.786 |

**Why F1/PR-AUC and not accuracy**: only 26.5% of customers churn, so a model
that always says "no churn" scores 73.5% accuracy while catching zero churners.
A missed churner (false negative) costs a lost customer; a false retention offer
(false positive) costs one cheap email — so recall on the churn class matters
most, but precision still matters. F1 (their harmonic mean) and PR-AUC (which
focuses on the minority class, unlike ROC-AUC) are the honest summaries.

## The agent

A hand-written **plan → act → observe → verify** loop (no framework, so every
line is inspectable).

- **Plan** — the LLM picks which tool(s) to call.
- **Act** — the loop executes them and appends the raw results to the conversation.
- **Observe** — results are captured in an `execution_log`.
- **Verify** — before showing an answer, `validate_answer()` extracts every
  number from it and checks each traces to a tool output (or to the conversation
  context, for thresholds like "0.7"). Ungrounded numbers trigger one corrective
  retry ("recompute or drop"), then a final check. Errors and empty results are
  surfaced, never papered over.

**Multi-turn memory**: `run(question, history=...)` carries prior turns, so
follow-ups like "how many of *those* have month-to-month contracts?" resolve
correctly.

### Tools

- `predict_churn_risk(customer_id | features)` — model-as-a-callable; returns
  `{risk_score, prediction_class, top_factors}`.
- `run_data_query(code)` — a **restricted** `exec()` of pandas against `df` in a
  locked namespace (safe-builtins whitelist + an AST check that rejects imports
  and dunder-attribute access). This is take-home-appropriate sandboxing, **not**
  production hardening for untrusted users — I'd use a real sandbox service there.
- `get_dataset_schema()` — columns/dtypes/sample values, so the agent never
  guesses column names (e.g. it correctly says there is no "region" column).

## Running it

```bash
# local
cp .env.example .env   # add GROQ_API_KEY
streamlit run app.py

# docker
docker build -t churn-agent .
docker run -p 8501:8501 --env-file .env churn-agent
```

Hosted: deployed on Streamlit Community Cloud with `GROQ_API_KEY` in Secrets.

## AI tool use disclosure

Built with an AI coding assistant (opencode / `deepseek-v4-pro`) throughout:
scaffolding, module code, debugging, and the eval test battery. All design
decisions — the metric choice, the sandboxing approach, the verification
mechanism — were reasoned through and are explained in this README and
`DATA_NOTES.md`. I can explain any part of the submission in a follow-up.

## Known limitations

- The numeric validator is a heuristic: it catches numbers with *no* trace, but
  can false-flag presentation numbers (table ranks, `1=Yes/0=No` legends). It
  does not prove arithmetic correctness.
- The model tool doesn't validate feature values itself (e.g. a negative tenure
  is only caught indirectly when the agent asks for a complete valid dict).
- Very open questions can exhaust the 6-iteration tool-call budget and return
  "couldn't finish".
