"""Thin wrapper around the Groq client. Only this module knows the provider,
so swapping to OpenRouter/another host is a one-file change."""

import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("CHURN_MODEL", "openai/gpt-oss-20b")
API_KEY_ENV = "GROQ_API_KEY"

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.getenv(API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"{API_KEY_ENV} is not set. Export it (or add it to .env) before running the agent."
            )
        _client = Groq(api_key=key)
    return _client


def call(messages, tools=None, temperature=0.0):
    """one chat completion. returns a normalized assistant message dict:

    {"role": "assistant", "content": str, "tool_calls": [{id, name, arguments}]}
    where arguments is still a JSON string. temperature 0 keeps numbers stable."""
    kwargs = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    resp = _get_client().chat.completions.create(**kwargs)
    msg = resp.choices[0].message

    out = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def parse_arguments(arguments):
    """json.loads a tool_call's arguments string. returns {} on empty, None on
    malformed json so the caller can decide whether to retry."""
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
