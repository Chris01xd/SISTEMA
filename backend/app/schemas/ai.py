"""API contract of the AI module endpoints. Deep validation of `parameters`, scenarios and
patient profiles is done by the module's own schemas (single source of truth), whose field
errors are returned as 422."""
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints


class AIInvokeRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    # {"intent"?, "scenario"?: ScenarioInput, "patient"?: PatientProfile, "include_timeline"?: bool}
    parameters: dict[str, Any] = Field(default_factory=dict)


class AIInvokeResponse(BaseModel):
    request_id: str
    response: str
    # Grouped by origin: observed (public dataset), simulated, predicted, assumptions.
    digital_twin: dict[str, Any]
    analysis: dict[str, Any] | None = None
    metadata: dict[str, Any]


class AIStatus(BaseModel):
    available: bool
    detail: Any = None
    digital_twin: dict[str, Any] | None = None
    prediction_model: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
