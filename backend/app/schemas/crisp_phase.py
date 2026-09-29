import uuid
from datetime import date, datetime

from pydantic import BaseModel

from app.models.crisp_phase import PhaseApprovalAction, PhaseStatus


class KpiBaselineTarget(BaseModel):
    """A single business indicator, tracked so a target is never confused
    with a measured result: `actual_value` stays null until a real
    post-pilot measurement exists."""

    indicator: str  # e.g. "MECHANICAL_AVAILABILITY", "OEE", "MTBF", "MTTR",
    #                       "UNPLANNED_STOPS", "MAINTENANCE_COST", "DOWNTIME_HOURS"
    unit: str = ""
    baseline_value: float | None = None
    baseline_date: date | None = None
    target_value: float | None = None
    actual_value: float | None = None
    actual_measured_at: date | None = None


class BusinessUnderstandingData(BaseModel):
    """Phase-1 payload, stored as `crisp_phases.phase_data` for phase_number=1."""

    business_problem: str = ""
    affected_loading_unit_ids: list[uuid.UUID] = []
    maintenance_processes_involved: list[str] = []
    general_objective: str = ""
    specific_objectives: list[str] = []
    interested_users: list[str] = []
    constraints: list[str] = []
    initial_risks: list[str] = []
    available_resources: list[str] = []
    kpis: list[KpiBaselineTarget] = []
    expected_goals: str = ""
    business_success_criteria: list[str] = []


class CrispPhaseOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    phase_number: int
    status: PhaseStatus
    phase_data: dict
    started_at: datetime | None
    completed_at: datetime | None
    approved_by_id: uuid.UUID | None
    approved_at: datetime | None
    last_rejection_reason: str
    model_config = {"from_attributes": True}


class CrispPhaseDataUpdate(BaseModel):
    """Generic envelope: the router validates `data` against the
    phase-specific schema (e.g. BusinessUnderstandingData for phase 1)."""

    data: dict


class CrispPhaseActivityCreate(BaseModel):
    description: str
    result: str = ""
    observations: str = ""


class CrispPhaseActivityOut(CrispPhaseActivityCreate):
    id: uuid.UUID
    phase_id: uuid.UUID
    performed_by_id: uuid.UUID | None
    performed_at: datetime
    model_config = {"from_attributes": True}


class CrispPhaseApprovalRequest(BaseModel):
    action: PhaseApprovalAction
    comments: str = ""
    # Required when action == RETURN_TO_PREVIOUS: which earlier phase to reopen.
    target_phase_number: int | None = None


class CrispPhaseApprovalOut(BaseModel):
    id: uuid.UUID
    phase_id: uuid.UUID
    action: PhaseApprovalAction
    approver_id: uuid.UUID
    comments: str
    previous_status: str
    new_status: str
    model_config = {"from_attributes": True}
