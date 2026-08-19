# Reflection

The hardest part wasn't training the model — it was making the agent *honest*.
The brief's core requirement is "never invent a number," and the trap is that
LLMs hallucinate numbers confidently. My first attempt at the verification step
caught obvious fabrications but false-flagged legitimate answers: the model
writes negatives with Unicode dashes (`–0.35`), restates thresholds from the
question ("customers over $10,000"), and adds table-rank digits that aren't data.
Each of those took a real debugging cycle to understand. The lesson: an
anti-hallucination check is a classifier over *your own agent's* output, and it
has precision/recall tradeoffs of its own.

The other surprise was tooling drift. The brief pointed at Groq's Llama 3.3, but
that model ID was deprecated by the time I deployed — the app 404'd, then 400'd
because GPT-OSS-20B (the replacement) required a different tool-call message
shape (`type` + nested `function`) than my first pass emitted. That burned time
I'd planned for polish, and it taught me to isolate the LLM client behind one
module so a provider change is a one-file fix.

What I'd do differently with more time: add multi-turn memory from the start
(rather than retrofitting it after a follow-up question answered the wrong
cohort), build a small eval set with known answers and a hallucination-rate
report before any UI work, and add input validation directly in the model tool
so a negative tenure is rejected at the boundary instead of by the agent's
reasoning. The sandboxing is also deliberately take-home-grade — for anything
untrusted I'd swap the `exec()` for a real sandbox.

I taught myself: Groq/OpenAI native tool-calling message formats, SHAP
TreeExplainer for per-prediction explanations, restricted `exec()` namespaces,
and how to design a rate-limit-tolerant agent loop. The model work itself
(cleaning, leakage hunting, class-imbalance metrics) I was comfortable with —
the genuinely new material was all in the agent layer.
