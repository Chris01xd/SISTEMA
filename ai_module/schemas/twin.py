"""Typed contracts of the digital twin (inputs are strict: unknown fields are rejected)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

StatusColor = Literal["GREEN", "YELLOW", "RED"]
ValueSource = Literal["observed", "simulated", "predicted", "assumption"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------- inputs


class ScenarioInput(_Strict):
    """A what-if scenario: a mass-casualty surge on top of the baseline admission flow.
    Every field is optional; omitted fields fall back to the twin's configured assumptions."""

    surge_patients: int = Field(default=0, ge=0, le=2000, description="Extra patients arriving during the surge.")
    surge_start_hour: int = Field(default=0, ge=0, le=720, description="Hours after the start of the horizon.")
    surge_duration_hours: int = Field(default=6, ge=1, le=72)
    surge_esi_mix: dict[int, float] | None = Field(
        default=None,
        description="Share of surge patients per ESI triage level (1 = most urgent). Default: observed ED mix.",
    )
    baseline_daily_admissions: float | None = Field(default=None, gt=0, le=1000)
    ward_beds: int | None = Field(default=None, gt=0, le=5000)
    icu_beds: int | None = Field(default=None, gt=0, le=1000)
    horizon_hours: int | None = Field(default=None, ge=1, le=720)
    runs: int | None = Field(default=None, ge=1, le=1000)
    seed: int | None = None

    @field_validator("surge_esi_mix")
    @classmethod
    def _check_mix(cls, mix: dict[int, float] | None) -> dict[int, float] | None:
        if mix is None:
            return None
        if not mix or any(level not in range(1, 6) for level in mix):
            raise ValueError("surge_esi_mix keys must be ESI levels 1-5")
        if any(share < 0 for share in mix.values()) or sum(mix.values()) <= 0:
            raise ValueError("surge_esi_mix shares must be non-negative and not all zero")
        return mix


class PatientProfile(_Strict):
    """Information available when a patient is admitted (no outcome information)."""

    age: float = Field(ge=0, le=120)
    sex: Literal["F", "M"]
    admission_type: str = Field(default="EW EMER.", max_length=64)
    admission_location: str = Field(default="EMERGENCY ROOM", max_length=64)
    admit_hour: int = Field(default=12, ge=0, le=23)
    arrived_via_ed: bool = True
    arrival_transport: str = Field(default="AMBULANCE", max_length=32)
    ed_acuity_esi: int | None = Field(default=None, ge=1, le=5)
    triage_temperature_f: float | None = Field(default=None, ge=90, le=110)
    triage_heartrate: float | None = Field(default=None, ge=20, le=250)
    triage_resprate: float | None = Field(default=None, ge=4, le=60)
    triage_o2sat: float | None = Field(default=None, ge=50, le=100)
    triage_sbp: float | None = Field(default=None, ge=50, le=300)
    triage_dbp: float | None = Field(default=None, ge=20, le=200)

    @model_validator(mode="after")
    def _ed_consistency(self) -> "PatientProfile":
        if not self.arrived_via_ed and self.ed_acuity_esi is not None:
            raise ValueError("ed_acuity_esi requires arrived_via_ed=true")
        return self


# ---------------------------------------------------------------------------- outputs


class Provenance(BaseModel):
    dataset: dict[str, Any]
    processed_sha256: str
    processed_rows: int
    processed_patients: int
    prepared_at: str


class DistributionStats(BaseModel):
    mean: float
    median: float
    p90: float
    n: int


class ObservedProfile(BaseModel):
    """Statistics computed directly from the public dataset (source = observed)."""

    source: Literal["observed"] = "observed"
    admissions: int
    patients: int
    age: DistributionStats
    sex_counts: dict[str, int]
    ed_arrival_pct: float
    ambulance_arrival_pct_of_ed: float
    esi_distribution: dict[str, int]
    icu_rate_pct: float
    icu_rate_pct_by_esi: dict[str, float]
    in_hospital_mortality_pct: float
    los_hours: DistributionStats
    los_hours_icu_patients: DistributionStats | None
    icu_hours_icu_patients: DistributionStats | None
    arrivals_by_hour_pct: list[float] = Field(min_length=24, max_length=24)
    notes: list[str]


class CapacityAssumptions(BaseModel):
    """Configuration of the modeled hospital (source = assumption, not data)."""

    source: Literal["assumption"] = "assumption"
    baseline_daily_admissions: float
    ward_beds: int
    icu_beds: int
    sizing_method: Literal["configured", "littles_law"]
    target_occupancy: float


class UnitStats(BaseModel):
    capacity: int
    mean_occupancy: float
    mean_utilization_pct: float
    peak_occupancy_mean: float
    peak_occupancy_p90: float
    hours_at_capacity_mean: float
    probability_reach_capacity_pct: float


class QueueStats(BaseModel):
    max_waiting_mean: float
    max_waiting_p90: float
    mean_wait_hours_per_patient: float
    patients_waited_pct: float


class TimelinePoint(BaseModel):
    hour: int
    ward_mean: float
    ward_p90: float
    icu_mean: float
    icu_p90: float
    waiting_mean: float


class SimulationResult(BaseModel):
    """Monte Carlo result (source = simulated). Aggregated over ``runs`` replications."""

    source: Literal["simulated"] = "simulated"
    scenario: dict[str, Any]
    capacity: CapacityAssumptions
    runs: int
    horizon_hours: int
    warmup_hours: int
    seed: int
    surge_esi_mix: dict[str, float]
    arrivals_mean: dict[str, float]
    ward: UnitStats
    icu: UnitStats
    waiting: QueueStats
    status_color: StatusColor
    timeline: list[TimelinePoint]
    assumptions: list[str]


class ModelInfo(BaseModel):
    name: str
    version: str
    trained_at: str
    dataset_sha256: str
    test_metrics: dict[str, dict[str, float]]


class PredictionResult(BaseModel):
    """ML prediction for one admission profile (source = predicted)."""

    source: Literal["predicted"] = "predicted"
    icu_probability: float
    predicted_los_hours: float
    predicted_los_days: float
    model: ModelInfo
    unknown_categories: dict[str, str]
    notes: list[str]


class TwinState(BaseModel):
    """The persisted digital state: observed profile + modeled hospital + baseline run."""

    built_at: str
    # Hash of the settings the state was built with; a change triggers a rebuild.
    config_fingerprint: str
    provenance: Provenance
    observed: ObservedProfile
    capacity: CapacityAssumptions
    baseline: SimulationResult
    status_color: StatusColor


class TwinContext(BaseModel):
    """Everything the reasoning layer may cite, grouped by the origin of each value."""

    observed: dict[str, Any]
    simulated: dict[str, Any] | None = None
    predicted: dict[str, Any] | None = None
    assumptions: dict[str, Any]
    provenance: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
