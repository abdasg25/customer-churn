"""Streamlit chat UI. Calls only the agent layer, never the model or tools."""

import json
import os

import streamlit as st

# on Streamlit Cloud the key comes from the Secrets UI, not a .env file
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

import agent

st.set_page_config(page_title="Customer Churn Analyst", layout="wide")

EXAMPLES = [
    "Which customers are most likely to churn?",
    "What is the churn rate by contract type?",
    "Does churn correlate with tenure?",
    "What's the churn risk for customer 7590-VHVEG?",
    "How many customers use fiber optic internet?",
]

with st.sidebar:
    st.header("Try asking")
    for ex in EXAMPLES:
        if st.button(ex):
            st.session_state.pending = ex
    st.divider()
    st.caption("Each answer has a collapsible trace showing the tool calls behind it.")

st.title("Customer Churn Analyst")
st.caption("Ask about the dataset or query a customer's churn risk. Every number is computed by a tool.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def _trace_parts(result):
    parts = []
    for e in result.get("execution_log", []):
        parts.append({
            "tool": e.get("tool"),
            "args": json.dumps(e.get("arguments"), default=str),
            "result": json.dumps(e.get("result"), default=str)[:1500],
        })
    return parts


def _render_trace(trace, verification):
    with st.expander("Show reasoning trace"):
        for t in trace:
            st.markdown(f"**{t['tool']}**")
            st.code(t["args"], language="json")
            st.code(t["result"], language="text")
        if verification.get("ok"):
            st.markdown(":white_check_mark: numbers verified against tool outputs")
        else:
            st.markdown(":warning: some numbers could not be traced to a tool")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("trace"):
            _render_trace(m["trace"], m.get("verification", {}))

question = st.chat_input("Ask about churn risk, segments, or the dataset...")
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

if question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            try:
                result = agent.run(question, history=history)
                answer = result.get("answer") or "(no answer returned)"
                trace = _trace_parts(result)
                verification = result.get("verification", {})
            except Exception as e:
                if "GROQ_API_KEY" in str(e):
                    answer = (
                        "The LLM API key isn't set. On Streamlit Cloud add GROQ_API_KEY "
                        "under Settings -> Secrets and Reboot; locally add it to a .env file."
                    )
                else:
                    answer = f"Sorry, I hit an error and can't answer that right now ({type(e).__name__})."
                trace, verification = [], {}
        st.markdown(answer)
        if trace:
            _render_trace(trace, verification)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "trace": trace,
        "verification": verification,
    })
