"""Planner chain: decides which twin operation answers a free-text message and extracts
only the parameters the user stated (validated by the same schemas as the API)."""
from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from ai_module.chains.base import ChainError
from ai_module.config.logging import get_logger, timed
from ai_module.prompts import planner_prompt
from ai_module.schemas.analysis import TwinQueryPlan

logger = get_logger(__name__)


def build_planner_chain(model: BaseChatModel) -> Runnable:
    """Reusable runnable. Input: {question}. Output: TwinQueryPlan."""
    return planner_prompt() | model.with_structured_output(TwinQueryPlan)


class LLMPlanner:
    uses_llm = True

    def __init__(self, model: BaseChatModel, model_name: str) -> None:
        self.name = model_name
        self._chain = build_planner_chain(model)

    def plan(self, question: str) -> TwinQueryPlan:
        try:
            with timed(logger, "llm_call", chain="planner", model=self.name, question_chars=len(question)):
                result = self._chain.invoke({"question": question})
        except Exception as exc:
            raise ChainError(f"planner chain failed ({type(exc).__name__})") from exc
        if not isinstance(result, TwinQueryPlan):
            raise ChainError("planner chain returned an invalid plan")
        return result


_SIMULATE = re.compile(r"simul|escenario|scenario|what[- ]if|qu[eé] pasa(r[ií]a)? si|surge|oleada|v[ií]ctimas|casualt|llegan|arrive", re.I)


class RulePlanner:
    """Deterministic planner used without an LLM: keyword intent, no parameter extraction
    (structured ``parameters`` in the request are the way to pass values in that mode)."""

    name = "rules"
    uses_llm = False

    def plan(self, question: str) -> TwinQueryPlan:
        return TwinQueryPlan(intent="simulate" if _SIMULATE.search(question) else "state")
