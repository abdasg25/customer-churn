"""Tool layer: the only interface the agent is allowed to touch. Wraps the
model function and a restricted dataframe-query executor as agent-callable
tools with explicit names, descriptions, and JSON-serializable outputs."""


TOOLS = []
