# Time log

Rough breakdown of the ~10 focused hours. (Spans roughly two calendar days with
breaks, not one continuous sitting.)

| Block | What I did | ~time |
|---|---|---|
| Setup + EDA | Repo scaffold, venv, pinned deps, git init; loaded the CSV and ran a structural audit (dtype traps, class balance, duplicates) | 1.5h |
| Data audit + cleaning | Hunted leakage, characterized the 11 `TotalCharges` blanks (MNAR), recoded the "No internet service" trap, ordinal/binary encodings, stratified split | 1.5h |
| Metric + model | Justified F1/PR-AUC in writing, trained majority + logistic baselines, then XGBoost with a small CV search | 1h |
| Model as a tool | Evaluation (confusion matrix, threshold analysis), SHAP explainability, `predict_churn_risk()` + joblib persistence | 1h |
| Tools + agent loop | Restricted `exec()` tool with a locked namespace; hand-written plan→act→observe loop | 1h |
| Verification | The anti-hallucination numeric validator + corrective retry + multi-step; multi-turn memory | 1.5h |
| Streamlit UI | Chat app, error boundaries, collapsible reasoning trace, example prompts | 1h |
| Docker + deploy + debugging | Dockerfile, docker build/run, Streamlit Cloud; fixed the Groq model migration (404→400), rate-limit retry, and verifier false positives | 1.5h |
| Write-up + eval | README, reflection, this log; ran a 19-question trap-door eval battery against the app | 0.5h |

**Where I stopped:** the "must complete" checklist is done (notebook, live app,
agent with planning/verification, modular code, git, Docker). I did not build the
optional React frontend, charts, critic agent, or a committed eval-set report —
I'd rather spend remaining time making the core honest than adding surface area.

**Biggest time surprise:** Groq's model lineup changed mid-build (Llama 3.3
deprecated → `openai/gpt-oss-20b`), which cost ~1h in migration and tool-call
format debugging.
