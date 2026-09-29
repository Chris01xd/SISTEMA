"""Graph nodes. Each node is a plain function ``(state, deps) -> partial update`` that
delegates the actual work to a service/chain, so every node is testable in isolation."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ai_module.chains.base import Analyst, ChainError, Planner
from ai_module.chains.template_analyst import TemplateAnalyst
from ai_module.config.logging import get_logger, log_event, reset_request_id, set_request_id
from ai_module.config.settings import AISettings
from ai_module.digital_twin.errors import DigitalTwinError, TwinDataNotPreparedError
from ai_module.digital_twin.service import DigitalTwinService
from ai_module.graphs.state import ErrorRecord, GraphState
from ai_module.schemas.analysis import AnalysisOutput, TwinQueryPlan
from ai_module.schemas.api import InvokeRequest
from ai_module.schemas.twin import PatientProfile, ScenarioInput
from ai_module.services.grounding import ungrounded_numbers
from ai_module.services.language import detect_language
from ai_module.services.response_renderer import render_response

logger = get_logger(__name__)


@dataclass
class GraphDependencies:
    """Everything the nodes need, injected (tests pass fakes)."""

    settings: AISettings
    twin: DigitalTwinService
    analyst: Analyst
    planner: Planner
    fallback_analyst: Analyst


def _error(node: str, exc: BaseException | str, fatal: bool) -> ErrorRecord:
    message = exc if isinstance(exc, str) else str(exc)
    return ErrorRecord(node=node, type="message" if isinstance(exc, str) else type(exc).__name__, message=message, fatal=fatal)


def instrument(name: str, fn: Callable[[GraphState, GraphDependencies], dict[str, Any]], deps: GraphDependencies) -> Callable[[GraphState], dict[str, Any]]:
    """Adds timing, structured logging and execution history to a node; an unexpected
    exception becomes a controlled ``failed`` status instead of crashing the graph."""

    def node(state: GraphState) -> dict[str, Any]:
        # LangGraph may run nodes outside the caller's context: re-bind the correlation id.
        token = set_request_id(state.get("request_id"))
        start = time.perf_counter()
        try:
            update = fn(state, deps)
            status = "ok"
        except Exception:  # last-resort guard; expected errors are handled inside nodes
            log_event(logger, "node_unhandled_exception", logging.ERROR, node=name)
            logger.debug("node_unhandled_exception traceback", exc_info=True)
            update = {"status": "failed", "errors": [_error(name, f"internal error in {name}", fatal=True)]}
            status = "error"
        duration = round((time.perf_counter() - start) * 1000, 1)
        log_event(logger, "graph_node", node=name, status=update.get("status", status), duration_ms=duration)
        reset_request_id(token)
        update["execution_history"] = [{"node": name, "status": update.get("status", status), "duration_ms": duration}]
        return update

    node.__name__ = name
    return node


# ---------------------------------------------------------------------------------------- nodes


def receive_input(state: GraphState, deps: GraphDependencies) -> dict[str, Any]:
    return {"status": "running", "attempts": 0, "metadata": {**state.get("metadata", {}), "received_at": time.time()}}


def validate_input(state: GraphState, deps: GraphDependencies) -> dict[str, Any]:
    raw = state["user_input"]
    try:
        request = InvokeRequest.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors(include_url=False, include_input=False))
        return {"status": "rejected", "errors": [_error("validate_input", f"invalid input: {details}", fatal=True)]}
    message = request.message.strip()
    if not message:
        return {"status": "rejected", "errors": [_error("validate_input", "message must not be blank", fatal=True)]}
    if len(message) > deps.settings.max_message_chars:
        return {"status": "rejected", "errors": [_error("validate_input", f"message longer than {deps.settings.max_message_chars} characters", fatal=True)]}
    request.message = message
    return {"validated_input": request.model_dump(), "language": detect_language(message)}


def plan_twin_queries(state: GraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Explicit parameters win (deterministic); otherwise the planner (LLM or rules) decides."""
    request = state["validated_input"] or {}
    params = request.get("parameters") or {}
    errors: list[ErrorRecord] = []
    if params.get("intent") or params.get("scenario") or params.get("patient"):
        intent = params.get("intent") or ("simulate" if params.get("scenario") else "predict" if params.get("patient") else "state")
        plan = TwinQueryPlan(intent=intent, scenario=params.get("scenario"), patient=params.get("patient"))
        source = "parameters"
    else:
        try:
            plan = deps.planner.plan(request["message"])
            source = deps.planner.name
        except (ChainError, ValidationError) as exc:
            errors.append(_error("plan_twin_queries", f"planner unavailable, using twin state only ({exc})", fatal=False))
            plan, source = TwinQueryPlan(intent="state"), "fallback"
    if plan.intent == "simulate" and plan.scenario is None:
        errors.append(_error("plan_twin_queries", "no scenario parameters were given; showing the baseline only", fatal=False))
    if plan.intent == "predict" and plan.patient is None:
        errors.append(_error("plan_twin_queries", "no patient profile was given; prediction skipped", fatal=False))
    return {"plan": {**plan.model_dump(), "source": source}, "errors": errors}


def load_digital_twin_context(state: GraphState, deps: GraphDependencies) -> dict[str, Any]:
    plan = state["plan"] or {}
    include_timeline = bool((state["validated_input"] or {}).get("parameters", {}).get("include_timeline"))
    scenario = ScenarioInput.model_validate(plan["scenario"]) if plan.get("scenario") else None
    patient = PatientProfile.model_validate(plan["patient"]) if plan.get("patient") else None
    try:
        context = deps.twin.get_context(scenario=scenario, patient=patient, include_timeline=include_timeline)
    except TwinDataNotPreparedError as exc:
        return {"status": "failed", "errors": [_error("load_digital_twin_context", exc, fatal=True)]}
    except DigitalTwinError as exc:
        return {"status": "rejected", "errors": [_error("load_digital_twin_context", exc, fatal=True)]}
    errors = [_error("load_digital_twin_context", w, fatal=False) for w in context.warnings]
    return {"digital_twin_context": context.model_dump(), "errors": errors}


def reason_or_analyze(state: GraphState, deps: GraphDependencies) -> dict[str, Any]:
    context = state["digital_twin_context"] or {}
    feedback = state.get("validation_feedback") or []
    model_input = {
        "question": state["validated_input"]["message"],
        "context": context,
        "context_json": json.dumps(context, ensure_ascii=False, default=str),
        "correction_notes": (
            "Your previous answer contained numbers that are not in the context: "
            + ", ".join(feedback) + ". Rewrite it using only numbers present in the context." if feedback else ""
        ),
        "language": state.get("language", "es"),
    }
    attempts = state.get("attempts", 0) + 1
    metadata = dict(state.get("metadata", {}))
    try:
        analysis = deps.analyst.analyze(model_input)
        metadata.update(analyst=deps.analyst.name, llm_used=deps.analyst.uses_llm)
        errors: list[ErrorRecord] = []
    except ChainError as exc:
        analysis = deps.fallback_analyst.analyze(model_input)
        metadata.update(analyst=deps.fallback_analyst.name, llm_used=False, llm_fallback=True)
        errors = [_error("reason_or_analyze", f"LLM unavailable, deterministic answer used ({exc})", fatal=False)]
    # model_input is stored without the duplicated context to keep the state light.
    stored_input = {k: v for k, v in model_input.items() if k != "context"}
    return {"model_input": stored_input, "model_output": analysis.model_dump(), "attempts": attempts,
            "metadata": metadata, "errors": errors, "status": "running"}


def _uses_llm(state: GraphState) -> bool:
    return bool(state.get("metadata", {}).get("llm_used"))


def generate_response(state: GraphState, deps: GraphDependencies) -> dict[str, Any]:
    analysis = AnalysisOutput.model_validate(state["model_output"])
    text = render_response(analysis, state["digital_twin_context"] or {}, state.get("language", "es"),
                           uses_llm=_uses_llm(state), model_name=deps.analyst.name)
    return {"response": text}


def validate_response(state: GraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Grounding: numbers written by the LLM must exist in the twin context or the user input."""
    analysis = AnalysisOutput.model_validate(state["model_output"])
    metadata = dict(state.get("metadata", {}))
    if not (state.get("response") or "").strip():
        return {"status": "failed", "errors": [_error("validate_response", "empty response", fatal=True)]}
    if not _uses_llm(state):
        metadata["grounded"] = True
        return {"status": "completed", "metadata": metadata}
    texts = [analysis.answer, *(f.statement for f in analysis.findings), *analysis.recommendations, *analysis.caveats]
    missing = ungrounded_numbers(texts, state["digital_twin_context"], state["user_input"])
    if not missing:
        metadata["grounded"] = True
        return {"status": "completed", "metadata": metadata, "validation_feedback": []}
    log_event(logger, "response_ungrounded", ungrounded_count=len(missing), attempt=state.get("attempts", 0))
    if state.get("attempts", 0) <= deps.settings.graph_max_regenerations:
        return {"status": "needs_regeneration", "validation_feedback": missing}
    metadata["grounded"] = False
    return {"status": "needs_regeneration", "validation_feedback": missing, "metadata": metadata}


def fallback_response(state: GraphState, deps: GraphDependencies) -> dict[str, Any]:
    """The LLM could not produce a grounded answer: answer deterministically instead."""
    model_input = {**(state.get("model_input") or {}), "context": state["digital_twin_context"]}
    analysis = deps.fallback_analyst.analyze(model_input)
    metadata = {**state.get("metadata", {}), "analyst": deps.fallback_analyst.name, "llm_used": False, "grounding_fallback": True}
    text = render_response(analysis, state["digital_twin_context"] or {}, state.get("language", "es"), uses_llm=False, model_name="")
    return {
        "model_output": analysis.model_dump(), "response": text, "metadata": metadata, "status": "completed",
        "errors": [_error("validate_response", "LLM answer contained values not found in the twin; replaced by a deterministic answer "
                                               f"(ungrounded: {', '.join(state.get('validation_feedback', []))})", fatal=False)],
    }


def handle_error(state: GraphState, deps: GraphDependencies) -> dict[str, Any]:
    fatal = [e["message"] for e in state.get("errors", []) if e["fatal"]]
    status = state.get("status") if state.get("status") in ("rejected", "failed") else "failed"
    return {"status": status, "response": "; ".join(fatal) or "The request could not be processed."}


def default_fallback_analyst() -> Analyst:
    return TemplateAnalyst()
