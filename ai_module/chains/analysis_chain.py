"""Analysis chain: prompt | chat model with structured output -> ``AnalysisOutput``."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from ai_module.chains.base import ChainError
from ai_module.config.logging import get_logger, timed
from ai_module.prompts import analyst_prompt
from ai_module.schemas.analysis import AnalysisOutput

logger = get_logger(__name__)


def build_analysis_chain(model: BaseChatModel) -> Runnable:
    """Reusable runnable. Input: {question, context_json, correction_notes}. Output: AnalysisOutput."""
    return analyst_prompt() | model.with_structured_output(AnalysisOutput)


class LLMAnalyst:
    """``Analyst`` backed by an LLM. Numbers come from the context; grounding is verified by the graph."""

    uses_llm = True

    def __init__(self, model: BaseChatModel, model_name: str) -> None:
        self.name = model_name
        self._chain = build_analysis_chain(model)

    def analyze(self, model_input: dict[str, Any]) -> AnalysisOutput:
        variables = {k: model_input.get(k, "") for k in ("question", "context_json", "correction_notes")}
        try:
            with timed(logger, "llm_call", chain="analysis", model=self.name, context_chars=len(variables["context_json"])):
                result = self._chain.invoke(variables)
        except Exception as exc:  # provider SDKs raise many unrelated types
            raise ChainError(f"analysis chain failed ({type(exc).__name__})") from exc
        if not isinstance(result, AnalysisOutput) or not result.answer.strip():
            raise ChainError("analysis chain returned an empty or invalid answer")
        return result
