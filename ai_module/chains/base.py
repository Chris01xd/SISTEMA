"""Interfaces the graph depends on. LangChain implementations and deterministic fallbacks
both satisfy them, so the graph never knows (or cares) whether an LLM is configured."""
from __future__ import annotations

from typing import Any, Protocol

from ai_module.schemas.analysis import AnalysisOutput, TwinQueryPlan


class ChainError(RuntimeError):
    """A chain failed (provider error, timeout, unparsable output). Message is safe to expose."""


class Analyst(Protocol):
    name: str
    uses_llm: bool

    def analyze(self, model_input: dict[str, Any]) -> AnalysisOutput:
        """``model_input`` keys: question, context (dict), context_json (str), correction_notes, language."""
        ...


class Planner(Protocol):
    name: str
    uses_llm: bool

    def plan(self, question: str) -> TwinQueryPlan: ...
