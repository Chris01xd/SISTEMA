"""Direct access to the public-data digital twin of `ai_module` (no LLM involved)."""
from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from app.authentication.dependencies import get_current_user, require_permission
from app.services import ai_module_service

router = APIRouter()


@router.get("/state", dependencies=[Depends(get_current_user)])
def get_state() -> dict:
    """Current digital state: observed profile of the public dataset, modeled capacity
    (assumptions) and the simulated baseline."""
    return ai_module_service.get_state()


@router.post("/simulate", dependencies=[Depends(require_permission("simulation:run"))])
def simulate(
    scenario: dict[str, Any] = Body(default_factory=dict, examples=[{"surge_patients": 60, "surge_duration_hours": 4, "surge_start_hour": 24}]),
    include_timeline: bool = Query(False),
) -> dict:
    """Monte Carlo what-if simulation (fields: surge_patients, surge_start_hour, surge_duration_hours,
    surge_esi_mix, baseline_daily_admissions, ward_beds, icu_beds, horizon_hours, runs, seed)."""
    return ai_module_service.simulate(scenario, include_timeline)


@router.post("/predict", dependencies=[Depends(require_permission("simulation:run"))])
def predict(patient: dict[str, Any] = Body(..., examples=[{"age": 72, "sex": "M", "ed_acuity_esi": 2}])) -> dict:
    """ICU-need probability and length of stay for one admission profile (trained model)."""
    return ai_module_service.predict(patient)
