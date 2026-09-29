"""LangChain tools over ``DigitalTwinService``. Twin values reach an LLM only through
these tools; each result carries a ``source`` field (observed / simulated / predicted)."""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from ai_module.digital_twin.errors import DigitalTwinError
from ai_module.digital_twin.service import DigitalTwinService
from ai_module.schemas.twin import PatientProfile, ScenarioInput


def build_twin_tools(service: DigitalTwinService) -> list[BaseTool]:
    """Tools bound to one service instance."""

    def get_digital_twin_state() -> dict[str, Any]:
        state = service.get_current_state()
        return {
            "observed": state.observed.model_dump(),
            "capacity": state.capacity.model_dump(),
            "baseline": state.baseline.model_dump(exclude={"timeline"}),
            "status_color": state.status_color,
            "provenance": {"dataset": state.provenance.dataset["name"], "version": state.provenance.dataset["version"],
                           "license": state.provenance.dataset["license"]},
        }

    def simulate_scenario(**parameters: Any) -> dict[str, Any]:
        try:
            return service.simulate(ScenarioInput(**parameters)).model_dump(exclude={"timeline"})
        except DigitalTwinError as exc:
            return {"error": str(exc)}

    def predict_admission(**profile: Any) -> dict[str, Any]:
        try:
            return service.predict(PatientProfile(**profile)).model_dump()
        except DigitalTwinError as exc:
            return {"error": str(exc)}

    return [
        StructuredTool.from_function(
            get_digital_twin_state, name="get_digital_twin_state",
            description="Current digital twin state: observed dataset statistics, modeled bed capacity and simulated baseline.",
        ),
        StructuredTool.from_function(
            simulate_scenario, name="simulate_scenario", args_schema=ScenarioInput,
            description="Monte Carlo what-if simulation of ward and ICU beds, e.g. a surge of extra patients. "
                        "Only pass values the user asked for; omitted fields use the twin's configuration.",
        ),
        StructuredTool.from_function(
            predict_admission, name="predict_admission", args_schema=PatientProfile,
            description="ML prediction of ICU need probability and length of stay for one admission profile.",
        ),
    ]
