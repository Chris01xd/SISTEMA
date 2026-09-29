"""Public request/response contract of the module (``AIModule.invoke``, ``POST /ai/invoke``)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_module.schemas.analysis import AnalysisOutput, Intent
from ai_module.schemas.twin import PatientProfile, ScenarioInput


class InvokeParameters(BaseModel):
    """Optional structured parameters. When given they take precedence over whatever the
    LLM extracts from the message (deterministic, auditable calls)."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent | None = None
    scenario: ScenarioInput | None = None
    patient: PatientProfile | None = None
    include_timeline: bool = False


class InvokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    parameters: InvokeParameters = Field(default_factory=InvokeParameters)


class InvokeResponse(BaseModel):
    request_id: str
    response: str
    digital_twin: dict[str, Any]
    analysis: AnalysisOutput | None = None
    metadata: dict[str, Any]
