import uuid
from datetime import date, datetime

from pydantic import BaseModel

from app.models.crisp_deployment import DeploymentServiceStatus, DriftType, EquipmentPredictedState


class ModelDeploymentCreate(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: uuid.UUID
    project_id: uuid.UUID
    loading_unit_ids: list[uuid.UUID]
    alert_thresholds: dict = {"warning": 0.4, "critical": 0.75}
    inference_frequency: str = "PER_READING"
    responsible_id: uuid.UUID | None = None
    rollback_plan: str = ""
    next_retraining_date: date | None = None


class ModelDeploymentOut(BaseModel):
    id: uuid.UUID
    model_id: uuid.UUID
    project_id: uuid.UUID
    version: str
    deployed_at: datetime
    api_endpoint: str
    service_status: DeploymentServiceStatus
    alert_thresholds: dict
    inference_frequency: str
    responsible_id: uuid.UUID | None
    rollback_plan: str
    next_retraining_date: date | None
    model_config = {"from_attributes": True, "protected_namespaces": ()}


class PredictionRequest(BaseModel):
    deployment_id: uuid.UUID
    loading_unit_id: uuid.UUID


class PredictionOut(BaseModel):
    id: uuid.UUID
    deployment_id: uuid.UUID
    loading_unit_id: uuid.UUID
    predicted_at: datetime
    equipment_state: EquipmentPredictedState
    risk_level: float
    anomaly_type: str | None
    failure_probability: float | None
    affected_component: str | None
    estimated_rul_hours: float | None
    severity: str
    maintenance_recommendation: str
    linked_alert_id: uuid.UUID | None
    model_config = {"from_attributes": True}


class ModelMonitoringOut(BaseModel):
    id: uuid.UUID
    deployment_id: uuid.UUID
    recorded_at: datetime
    current_accuracy: float | None
    inference_latency_ms: float | None
    prediction_count: int
    alerts_generated_count: int
    false_alarm_count: int
    needs_retraining: bool
    notes: str
    model_config = {"from_attributes": True}


class DataDriftRecordOut(BaseModel):
    id: uuid.UUID
    deployment_id: uuid.UUID
    detected_at: datetime
    drift_type: DriftType
    variable_name: str | None
    drift_score: float
    method: str
    threshold_used: float
    is_significant: bool
    model_config = {"from_attributes": True}
