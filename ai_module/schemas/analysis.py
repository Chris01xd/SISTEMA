"""Structured outputs requested from the LLM (LangChain ``with_structured_output``)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ai_module.schemas.twin import PatientProfile, ScenarioInput

Intent = Literal["state", "simulate", "predict", "general"]


class TwinQueryPlan(BaseModel):
    """Which digital-twin operations answer the user's message. The LLM only extracts
    parameters the *user* stated; it never supplies twin values."""

    intent: Intent = Field(description="state: current status; simulate: what-if surge; predict: one patient; general: other.")
    scenario: ScenarioInput | None = Field(default=None, description="Only for intent=simulate; only values the user stated.")
    patient: PatientProfile | None = Field(default=None, description="Only for intent=predict; only values the user stated.")


class Finding(BaseModel):
    statement: str = Field(description="One fact, citing numbers exactly as they appear in the context.")
    source: Literal["observed", "simulated", "predicted", "assumption"] = Field(
        description="Context section the numbers of this statement come from."
    )


class AnalysisOutput(BaseModel):
    """The LLM's interpretation. Numbers must be copied from the twin context."""

    answer: str = Field(description="Direct answer to the user's question, in the user's language.")
    findings: list[Finding] = Field(default_factory=list, max_length=8)
    recommendations: list[str] = Field(default_factory=list, max_length=5, description="Decision-support suggestions, not orders.")
    caveats: list[str] = Field(default_factory=list, max_length=5)
