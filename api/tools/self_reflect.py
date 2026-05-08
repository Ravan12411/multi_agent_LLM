from langchain_core.tools import tool

@tool
def self_reflect(previous_output: str, critique: str) -> str:
    """Allow an agent to re-read its own output along with a critique to self-correct."""
    return f"Reflection applied. I will fix the following based on the critique: {critique}\n\nRe-evaluating previous output: {previous_output}"
