import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.emergency import EventType
from app.models.simulation import SimulationStatus


class SeverityDistribution(BaseModel):
    minor: float = 0.4
    delayed: float = 0.3
    immediate: float = 0.2
    critical: float = 0.08
    expectant: float = 0.02


class EmergencyEventCreate(BaseModel):
    hospital_id: uuid.UUID
    name: str
    event_type: EventType
    magnitude: float = 0.0
    expected_victims: int = Field(gt=0)
    severity_distribution: SeverityDistribution = SeverityDistribution()
    arrival_curve: str = "surge_decay"  # surge_decay | linear | poisson_steady
    arrival_window_minutes: int = 60


class EmergencyEventOut(BaseModel):
    id: uuid.UUID
    name: str
    event_type: EventType
    magnitude: float
    expected_victims: int
    started_at: datetime
    params: dict
    model_config = {"from_attributes": True}


class SimulationCreate(BaseModel):
    event_id: uuid.UUID
    horizon_minutes: int = 360
    timestep_minutes: int = 1
    use_ai_optimizer: bool = False
    assignment_strategy: str = "greedy"  # greedy | genetic | rl | milp


class SimulationOut(BaseModel):
    id: uuid.UUID
    event_id: uuid.UUID
    horizon_minutes: int
    timestep_minutes: int
    status: SimulationStatus
    started_at: datetime | None
    finished_at: datetime | None
    model_config = {"from_attributes": True}


class SimulationResultOut(BaseModel):
    tick: int
    timestamp: datetime
    metrics: dict
    model_config = {"from_attributes": True}


class SimulationRunSummary(BaseModel):
    simulation_id: uuid.UUID
    ticks_executed: int
    final_metrics: dict
    recommendations: list[str] = []
