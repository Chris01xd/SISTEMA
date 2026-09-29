import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.crisp_evaluation import EvaluationDecision


class OperationalMetrics(BaseModel):
    anticipation_time_hours: float | None = None
    critical_failures_detected: int | None = None
    critical_failures_missed: int | None = None
    false_alerts: int | None = None


class BusinessMetricsEstimate(BaseModel):
    """Every field is an *estimate* derived from validation-set results — not
    a measured business outcome, which requires pilot data (see phase 5's
    business-metrics slide/rationale)."""

    estimated_unplanned_stop_reduction: float | None = None
    estimated_mtbf_impact: float | None = None
    estimated_mttr_impact: float | None = None
    estimated_savings: float | None = None


class ModelEvaluationCreate(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: uuid.UUID
    operational_metrics: OperationalMetrics = OperationalMetrics()
    business_metrics_estimate: BusinessMetricsEstimate = BusinessMetricsEstimate()
    specialist_opinion: str = ""
    decision: EvaluationDecision
    justification: str


class ModelEvaluationOut(BaseModel):
    id: uuid.UUID
    model_id: uuid.UUID
    evaluator_id: uuid.UUID
    operational_metrics: dict
    business_metrics_estimate: dict
    specialist_opinion: str
    decision: EvaluationDecision | None
    justification: str
    evaluated_at: datetime | None
    model_config = {"from_attributes": True, "protected_namespaces": ()}
