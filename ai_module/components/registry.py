"""Framework-agnostic component registry: the seam for a future Langflow integration.

Each ``ComponentSpec`` describes one building block (name, kind, typed inputs/outputs and
a factory returning a plain callable). A Langflow custom component — or any other UI/
orchestrator — only needs to wrap ``spec.build(module)``; no business logic lives in the
adapter, and this package never imports Langflow. See ``docs/langflow-integration.md``."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from ai_module.prompts import PROMPT_NAMES, load_prompt_text
from ai_module.schemas.analysis import AnalysisOutput, TwinQueryPlan
from ai_module.schemas.api import InvokeRequest, InvokeResponse
from ai_module.schemas.twin import PatientProfile, PredictionResult, ScenarioInput, SimulationResult, TwinState
from ai_module.services.ai_service import AIModule

Kind = Literal["service", "tool", "chain", "graph", "prompt"]


class _PromptOutput(BaseModel):
    name: str
    text: str


class _AnalysisInput(BaseModel):
    question: str
    context: dict[str, Any]
    language: str = "es"


class _PlannerInput(BaseModel):
    question: str


class _NoInput(BaseModel):
    pass


class _PromptInput(BaseModel):
    name: str


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: Kind
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    factory: Callable[[AIModule], Callable[..., Any]]

    def build(self, module: AIModule) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """A callable ``dict -> dict`` with validated input and JSON-serializable output."""
        target = self.factory(module)

        def run(payload: dict[str, Any] | None = None) -> dict[str, Any]:
            data = self.input_schema.model_validate(payload or {})
            result = target(data)
            return result.model_dump(mode="json") if isinstance(result, BaseModel) else result

        run.__name__ = self.name
        return run

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "description": self.description,
            "inputs": self.input_schema.model_json_schema(), "outputs": self.output_schema.model_json_schema(),
        }


def _analysis(module: AIModule) -> Callable[[_AnalysisInput], AnalysisOutput]:
    import json

    return lambda d: module.analyst.analyze({
        "question": d.question, "context": d.context, "language": d.language, "correction_notes": "",
        "context_json": json.dumps(d.context, ensure_ascii=False, default=str),
    })


COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("digital_twin_state", "service", "Current digital twin state (observed profile, capacity, simulated baseline).",
                  _NoInput, TwinState, lambda m: lambda _d: m.get_state()),
    ComponentSpec("digital_twin_simulate", "tool", "Monte Carlo what-if simulation of ward/ICU beds.",
                  ScenarioInput, SimulationResult, lambda m: m.simulate),
    ComponentSpec("digital_twin_predict", "tool", "ICU need probability and length of stay for one admission profile.",
                  PatientProfile, PredictionResult, lambda m: m.predict),
    ComponentSpec("twin_query_planner", "chain", "Chooses the twin operation for a message and extracts user-stated parameters.",
                  _PlannerInput, TwinQueryPlan, lambda m: lambda d: m.planner.plan(d.question)),
    ComponentSpec("twin_analyst", "chain", "Structured interpretation of a twin context (LLM or deterministic).",
                  _AnalysisInput, AnalysisOutput, _analysis),
    ComponentSpec("twin_graph", "graph", "Full LangGraph workflow: validation -> twin context -> analysis -> grounded response.",
                  InvokeRequest, InvokeResponse, lambda m: lambda d: m.invoke(d.message, d.parameters.model_dump())),
    ComponentSpec("prompt", "prompt", f"Prompt templates: {', '.join(PROMPT_NAMES)}.",
                  _PromptInput, _PromptOutput, lambda _m: lambda d: _PromptOutput(name=d.name, text=load_prompt_text(d.name))),
)


def get_component(name: str) -> ComponentSpec:
    for spec in COMPONENTS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown component '{name}'. Available: {[c.name for c in COMPONENTS]}")


def describe_components() -> list[dict[str, Any]]:
    """JSON description of every component (inputs/outputs as JSON Schema)."""
    return [spec.describe() for spec in COMPONENTS]
