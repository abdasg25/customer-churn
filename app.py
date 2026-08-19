"""Streamlit chat UI. Calls only the agent layer, never the model or tools."""

import streamlit as st

import agent

st.set_page_config(page_title="Customer Churn Analyst", layout="wide")
st.title("Customer Churn Analyst")
st.caption("Ask about the dataset or query a customer's churn risk. Every number is computed by a tool.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("e.g. which customers are most likely to churn?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            try:
                result = agent.run(question)
                answer = result.get("answer", "(no answer)")
            except Exception as e:
                if "GROQ_API_KEY" in str(e):
                    answer = "The LLM API key isn't set. Add GROQ_API_KEY to a .env file and restart."
                else:
                    answer = f"Sorry, I hit an error and can't answer that right now ({type(e).__name__})."
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
