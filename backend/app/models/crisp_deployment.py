"""CRISP-DM phase 6 (deployment): the model in production, its predictions,
and the monitoring/drift trail that triggers retraining."""
import enum
from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class DeploymentServiceStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"


class EquipmentPredictedState(str, enum.Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class DriftType(str, enum.Enum):
    DATA_DRIFT = "DATA_DRIFT"
    MODEL_DRIFT = "MODEL_DRIFT"


class ModelDeployment(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "model_deployments"

    model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_models.id"), index=True)
    project_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_projects.id"), index=True)
    version: Mapped[str] = mapped_column(String(50), default="v1")
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    api_endpoint: Mapped[str] = mapped_column(String(255), default="")
    service_status: Mapped[DeploymentServiceStatus] = mapped_column(
        Enum(DeploymentServiceStatus), default=DeploymentServiceStatus.ACTIVE
    )
    # per-severity thresholds on the model's risk score, e.g. {"warning": 0.4, "critical": 0.75}
    alert_thresholds: Mapped[dict] = mapped_column(JSON, default=dict)
    inference_frequency: Mapped[str] = mapped_column(String(50), default="PER_READING")
    responsible_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    rollback_plan: Mapped[str] = mapped_column(Text, default="")
    next_retraining_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    loading_units: Mapped[list["ModelDeploymentLoadingUnit"]] = relationship(
        back_populates="deployment", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["ModelPrediction"]] = relationship(back_populates="deployment")
    monitoring_records: Mapped[list["ModelMonitoring"]] = relationship(back_populates="deployment")
    drift_records: Mapped[list["DataDriftRecord"]] = relationship(back_populates="deployment")


class ModelDeploymentLoadingUnit(Base, UUIDPKMixin):
    __tablename__ = "model_deployment_loading_units"

    deployment_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_deployments.id"), index=True)
    loading_unit_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("loading_units.id"), index=True)

    deployment: Mapped["ModelDeployment"] = relationship(back_populates="loading_units")


class ModelPrediction(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "model_predictions"

    deployment_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_deployments.id"), index=True)
    loading_unit_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("loading_units.id"), index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    equipment_state: Mapped[EquipmentPredictedState] = mapped_column(Enum(EquipmentPredictedState))
    risk_level: Mapped[float] = mapped_column(Float, default=0.0)
    anomaly_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    failure_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    affected_component: Mapped[str | None] = mapped_column(String(150), nullable=True)
    estimated_rul_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[str] = mapped_column(String(30), default="NORMAL")
    maintenance_recommendation: Mapped[str] = mapped_column(Text, default="")
    raw_input_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    linked_alert_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("alerts.id"), nullable=True)

    deployment: Mapped["ModelDeployment"] = relationship(back_populates="predictions")


class ModelMonitoring(Base, UUIDPKMixin, TimestampMixin):
    """Periodic health snapshot of a deployment — recorded_at rows form the
    rolling series shown on the monitoring dashboard."""

    __tablename__ = "model_monitoring"

    deployment_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_deployments.id"), index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    current_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_count: Mapped[int] = mapped_column(Integer, default=0)
    alerts_generated_count: Mapped[int] = mapped_column(Integer, default=0)
    false_alarm_count: Mapped[int] = mapped_column(Integer, default=0)
    data_distribution_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    needs_retraining: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")

    deployment: Mapped["ModelDeployment"] = relationship(back_populates="monitoring_records")


class DataDriftRecord(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "data_drift_records"

    deployment_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_deployments.id"), index=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    drift_type: Mapped[DriftType] = mapped_column(Enum(DriftType))
    variable_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    drift_score: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(50), default="PSI")
    threshold_used: Mapped[float] = mapped_column(Float, default=0.2)
    is_significant: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    deployment: Mapped["ModelDeployment"] = relationship(back_populates="drift_records")
