"""Agent layer: the plan -> act -> observe -> verify loop. Owns conversation
state, tool selection, retries, and the anti-hallucination check that gates
every answer before it reaches the user."""


def run(question: str) -> dict:
    raise NotImplementedError
