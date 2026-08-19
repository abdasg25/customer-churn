"""Agent layer: plan -> act -> observe -> verify. Owns state, tool selection,
retries, and the anti-hallucination gate that every answer passes through."""

import json
import re

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

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def _extract_numbers(text):
    out = []
    for tok in NUMBER_RE.findall(text or ""):
        t = tok.replace(",", "").rstrip("%")
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _tool_output_text(log):
    return " ".join(json.dumps(e.get("result", ""), default=str) for e in log)


def _close(a, b):
    # ~1.5% relative tolerance so rounding doesn't false-positive, but a number
    # with no nearby counterpart in the tool outputs is treated as ungrounded
    return abs(a - b) <= 0.015 * max(abs(a), abs(b), 1.0)


def _grounded(n, output_nums):
    for m in output_nums:
        # allow the model to restate a proportion as a percentage (and vice versa)
        if _close(n, m) or _close(n, m * 100) or _close(n * 100, m):
            return True
    return False


def validate_answer(answer, log):
    """check every number in the answer traces to a tool output.

    returns {"ok", "violations", "checked_numbers"}. heuristic, not a proof."""
    output_nums = _extract_numbers(_tool_output_text(log))
    nums = _extract_numbers(answer)
    violations = [n for n in nums if not _grounded(n, output_nums)]
    return {"ok": not violations, "violations": violations, "checked_numbers": nums}


def _loop(messages, log, max_iterations):
    """plan->act->observe: drive tool calls until the model returns a final answer."""
    for _ in range(max_iterations):
        resp = llm.call(messages, tools=TOOLS)

        if not resp.get("tool_calls"):
            return resp.get("content", "")

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

    return "I couldn't finish this within the tool-call limit."


def run(question, max_iterations=MAX_ITERATIONS):
    """full run with the verify gate. returns {answer, execution_log, verification}."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    log = []

    answer = _loop(messages, log, max_iterations)
    check = validate_answer(answer, log)

    # one corrective retry if the answer has ungrounded numbers
    if check["violations"] and check["checked_numbers"]:
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": (
            "Some numbers in that answer don't trace to any tool result: "
            + ", ".join(str(v) for v in check["violations"])
            + ". Recompute them with a tool or drop them, then answer again."
        )})
        answer2 = _loop(messages, log, max_iterations)
        check2 = validate_answer(answer2, log)
        if not check2["violations"]:
            answer, check = answer2, check2

    return {"answer": answer, "execution_log": log, "verification": check}
