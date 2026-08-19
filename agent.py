"""Agent layer: the plan -> act -> observe loop. Owns conversation state,
tool selection, retries, and (in the next task) the verification gate."""

import json

import llm
from tools import TOOLS, TOOL_FUNCTIONS

MAX_ITERATIONS = 6

SYSTEM_PROMPT = (
    "You are a data analyst working over a customer churn dataset. "
    "You can call tools to compute real numbers. Never invent a number: every "
    "figure in your answer must come from a tool result you just received. "
    "If a tool errors, retry with corrected arguments or say you couldn't "
    "compute it. Be concise and answer in plain English."
)


def run(question, max_iterations=MAX_ITERATIONS):
    """run the plan->act->observe loop for one question.

    returns {"answer": str, "execution_log": [{tool, arguments, result}, ...]}."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    log = []

    for _ in range(max_iterations):
        resp = llm.call(messages, tools=TOOLS)

        # no tool calls -> the model is done reasoning, this is the final answer
        if not resp.get("tool_calls"):
            return {"answer": resp.get("content", ""), "execution_log": log}

        messages.append(resp)
        for tc in resp["tool_calls"]:
            name = tc["name"]
            args = llm.parse_arguments(tc["arguments"])

            if args is None:
                result = {"status": "error", "error": "malformed tool arguments"}
            elif name not in TOOL_FUNCTIONS:
                result = {"status": "error", "error": f"unknown tool: {name}"}
            else:
                try:
                    result = TOOL_FUNCTIONS[name](args)
                except Exception as e:
                    result = {"status": "error", "error": f"{type(e).__name__}: {e}"}

            log.append({"tool": name, "arguments": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, default=str),
            })

    return {
        "answer": "I couldn't finish this within the tool-call limit.",
        "execution_log": log,
    }
