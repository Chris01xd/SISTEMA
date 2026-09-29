"""CRISP-DM phase 4 (modeling): experiment tracking and trained model registry."""
import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class ModelObjective(str, enum.Enum):
    FAILURE_CLASSIFICATION = "FAILURE_CLASSIFICATION"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    FAILURE_PREDICTION = "FAILURE_PREDICTION"
    REMAINING_USEFUL_LIFE = "REMAINING_USEFUL_LIFE"
    OPERATIONAL_VARIABLE_FORECAST = "OPERATIONAL_VARIABLE_FORECAST"


class Algorithm(str, enum.Enum):
    RANDOM_FOREST = "RANDOM_FOREST"
    XGBOOST = "XGBOOST"
    LSTM = "LSTM"
    ISOLATION_FOREST = "ISOLATION_FOREST"
    REGRESSION = "REGRESSION"


class ExperimentStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    QUEUED = "QUEUED"
    TRAINING = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MetricSplit(str, enum.Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class MlExperiment(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "ml_experiments"

    project_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_projects.id"), index=True)
    dataset_version_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("dataset_versions.id"))
    objective: Mapped[ModelObjective] = mapped_column(Enum(ModelObjective))
    algorithm: Mapped[Algorithm] = mapped_column(Enum(Algorithm))
    input_variables: Mapped[list] = mapped_column(JSON, default=list)
    target_variable: Mapped[str] = mapped_column(String(100), default="")
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_method: Mapped[str] = mapped_column(String(100), default="TEMPORAL_HOLDOUT")
    responsible_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[ExperimentStatus] = mapped_column(Enum(ExperimentStatus), default=ExperimentStatus.DRAFT)
    model_artifact_uri: Mapped[str] = mapped_column(String(500), default="")
    observations: Mapped[str] = mapped_column(Text, default="")

    models: Mapped[list["MlModel"]] = relationship(back_populates="experiment")
    metrics: Mapped[list["ModelMetric"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class MlModel(Base, UUIDPKMixin, TimestampMixin):
    """A registered, artifact-backed model produced by an experiment."""

    __tablename__ = "ml_models"

    experiment_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_experiments.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[str] = mapped_column(String(50), default="v1")
    algorithm: Mapped[Algorithm] = mapped_column(Enum(Algorithm))
    artifact_uri: Mapped[str] = mapped_column(String(500))
    is_candidate: Mapped[bool] = mapped_column(default=True)
    is_published: Mapped[bool] = mapped_column(default=False)

    experiment: Mapped["MlExperiment"] = relationship(back_populates="models")


class ModelMetric(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "model_metrics"

    experiment_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_experiments.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(50))
    metric_value: Mapped[float] = mapped_column(Float)
    dataset_split: Mapped[MetricSplit] = mapped_column(Enum(MetricSplit), default=MetricSplit.TEST)

    experiment: Mapped["MlExperiment"] = relationship(back_populates="metrics")
