"""``AIModule``: the public facade of the package (what the backend API, a CLI or a future
Langflow component call). It wires settings -> LLM -> chains -> twin -> graph once and
exposes plain methods with validated inputs and JSON-serializable outputs."""
from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import ValidationError

from ai_module.chains.analysis_chain import LLMAnalyst
from ai_module.chains.base import Analyst, Planner
from ai_module.chains.planner_chain import LLMPlanner, RulePlanner
from ai_module.chains.template_analyst import TemplateAnalyst
from ai_module.config.logging import get_logger, log_event, reset_request_id, set_request_id
from ai_module.config.settings import AISettings, get_settings
from ai_module.digital_twin.errors import DigitalTwinError
from ai_module.digital_twin.service import DigitalTwinService
from ai_module.graphs.builder import build_graph
from ai_module.graphs.nodes import GraphDependencies
from ai_module.graphs.state import initial_state
from ai_module.prompts import PROMPT_NAMES, prompt_version
from ai_module.schemas.api import InvokeResponse
from ai_module.schemas.twin import PatientProfile, PredictionResult, ScenarioInput, SimulationResult, TwinState
from ai_module.services.llm_factory import LLMConfigurationError, build_chat_model, configuration_problem

logger = get_logger(__name__)


class AIModule:
    def __init__(self, settings: AISettings | None = None, twin: DigitalTwinService | None = None,
                 analyst: Analyst | None = None, planner: Planner | None = None) -> None:
        """Explicit ``analyst``/``planner`` override the configured LLM (used by tests and integrations)."""
        self.settings = settings or get_settings()
        self.twin = twin or DigitalTwinService(self.settings)
        self.llm_problem: str | None = None
        if analyst is None or planner is None:
            llm_analyst, llm_planner = self._llm_components()
            analyst = analyst or llm_analyst
            planner = planner or llm_planner
        self.analyst: Analyst = analyst
        self.planner: Planner = planner
        self.graph = build_graph(GraphDependencies(
            settings=self.settings, twin=self.twin, analyst=self.analyst, planner=self.planner, fallback_analyst=TemplateAnalyst(),
        ))

    def _llm_components(self) -> tuple[Analyst, Planner]:
        try:
            model = build_chat_model(self.settings)
        except LLMConfigurationError as exc:
            self.llm_problem = str(exc)
            if self.settings.llm_enabled:
                log_event(logger, "llm_unavailable", reason=self.llm_problem)
            return TemplateAnalyst(), RulePlanner()
        name = f"{self.settings.llm_provider}:{self.settings.llm_model}"
        return LLMAnalyst(model, name), LLMPlanner(model, name)

    # ---------------------------------------------------------------- graph

    def invoke(self, message: Any, parameters: Any = None, request_id: str | None = None) -> InvokeResponse:
        """Runs the LangGraph workflow. Never raises for bad input or twin errors: the result's
        ``metadata.status`` is ``completed``, ``rejected`` (invalid input) or ``failed``."""
        request_id = request_id or str(uuid.uuid4())
        token = set_request_id(request_id)
        start = time.perf_counter()
        try:
            final = self.graph.invoke(initial_state(request_id, message, parameters), config={"recursion_limit": 25})
        finally:
            reset_request_id(token)
        duration = round((time.perf_counter() - start) * 1000, 1)
        context = final.get("digital_twin_context") or {}
        meta = final.get("metadata", {})
        plan = final.get("plan") or {}
        response = InvokeResponse(
            request_id=request_id,
            response=final.get("response") or "",
            digital_twin={
                "observed": context.get("observed"),
                "simulated": context.get("simulated"),
                "predicted": context.get("predicted"),
                "assumptions": context.get("assumptions"),
            } if context else {},
            analysis=final.get("model_output") if final.get("status") == "completed" else None,
            metadata={
                "status": final.get("status"),
                "intent": plan.get("intent"),
                "plan_source": plan.get("source"),
                "language": final.get("language"),
                "llm": {
                    "provider": self.settings.llm_provider,
                    "model": self.settings.llm_model,
                    "configured": self.analyst.uses_llm,
                    "used": bool(meta.get("llm_used")),
                    "fallback": bool(meta.get("llm_fallback") or meta.get("grounding_fallback")),
                    "unavailable_reason": self.llm_problem,
                },
                "grounded": meta.get("grounded"),
                "attempts": final.get("attempts", 0),
                "duration_ms": duration,
                "execution_history": final.get("execution_history", []),
                "errors": final.get("errors", []),
                "dataset": (context.get("provenance") or {}).get("dataset", {}).get("name"),
                "model_version": ((context.get("predicted") or {}).get("model") or {}).get("version"),
                "prompt_versions": {name: prompt_version(name) for name in PROMPT_NAMES},
            },
        )
        log_event(logger, "ai_invoke", request_id=request_id, status=final.get("status"), intent=plan.get("intent"), duration_ms=duration,
                  llm_used=bool(meta.get("llm_used")), attempts=final.get("attempts", 0))
        return response

    # ---------------------------------------------------------------- twin (direct, no LLM)

    def get_state(self) -> TwinState:
        return self.twin.get_current_state()

    def simulate(self, scenario: ScenarioInput | dict[str, Any]) -> SimulationResult:
        """Raises ``pydantic.ValidationError`` for invalid input and ``DigitalTwinError`` for twin problems."""
        return self.twin.simulate(scenario if isinstance(scenario, ScenarioInput) else ScenarioInput.model_validate(scenario))

    def predict(self, patient: PatientProfile | dict[str, Any]) -> PredictionResult:
        return self.twin.predict(patient if isinstance(patient, PatientProfile) else PatientProfile.model_validate(patient))

    def status(self) -> dict[str, Any]:
        """Readiness report (never includes secrets)."""
        try:
            state = self.twin.get_current_state()
            twin = {"ready": True, "status_color": state.status_color, "built_at": state.built_at}
        except DigitalTwinError as exc:
            twin = {"ready": False, "detail": str(exc)}
        return {
            "digital_twin": twin,
            "prediction_model": {"ready": self.twin.model_available()},
            "llm": {"provider": self.settings.llm_provider, "model": self.settings.llm_model, "ready": self.analyst.uses_llm,
                    "detail": None if self.analyst.uses_llm else (self.llm_problem or configuration_problem(self.settings))},
        }


__all__ = ["AIModule", "ValidationError", "DigitalTwinError"]
