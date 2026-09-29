"""Typed state of the LangGraph workflow. Nodes return partial updates; ``errors`` and
``execution_history`` are append-only (reducer ``operator.add``)."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Status = Literal["running", "needs_regeneration", "completed", "rejected", "failed"]


class ErrorRecord(TypedDict):
    node: str
    type: str
    message: str  # safe for clients: no stack traces or secrets
    fatal: bool


class NodeRecord(TypedDict):
    node: str
    status: str
    duration_ms: float


class GraphState(TypedDict, total=False):
    request_id: str
    user_input: dict[str, Any]  # raw {"message", "parameters"} as received
    validated_input: dict[str, Any] | None  # InvokeRequest.model_dump()
    language: str
    plan: dict[str, Any] | None  # TwinQueryPlan.model_dump()
    digital_twin_context: dict[str, Any] | None  # TwinContext.model_dump()
    model_input: dict[str, Any] | None
    model_output: dict[str, Any] | None  # AnalysisOutput.model_dump()
    response: str | None
    attempts: int
    validation_feedback: list[str]
    status: Status
    metadata: dict[str, Any]
    errors: Annotated[list[ErrorRecord], operator.add]
    execution_history: Annotated[list[NodeRecord], operator.add]


def initial_state(request_id: str, message: Any, parameters: Any) -> GraphState:
    return GraphState(
        request_id=request_id,
        user_input={"message": message, "parameters": parameters if parameters is not None else {}},
        validated_input=None,
        plan=None,
        digital_twin_context=None,
        model_input=None,
        model_output=None,
        response=None,
        attempts=0,
        validation_feedback=[],
        status="running",
        metadata={},
        errors=[],
        execution_history=[],
    )
