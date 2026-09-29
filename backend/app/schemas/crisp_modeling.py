import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.crisp_modeling import Algorithm, ExperimentStatus, MetricSplit, ModelObjective


class MlExperimentCreate(BaseModel):
    project_id: uuid.UUID
    dataset_version_id: uuid.UUID
    objective: ModelObjective
    algorithm: Algorithm
    input_variables: list[str]
    target_variable: str
    hyperparameters: dict = {}
    validation_method: str = "TEMPORAL_HOLDOUT"
    responsible_id: uuid.UUID | None = None


class MlExperimentOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    dataset_version_id: uuid.UUID
    objective: ModelObjective
    algorithm: Algorithm
    input_variables: list
    target_variable: str
    hyperparameters: dict
    validation_method: str
    responsible_id: uuid.UUID | None
    trained_at: datetime | None
    duration_seconds: float | None
    status: ExperimentStatus
    model_artifact_uri: str
    observations: str
    model_config = {"from_attributes": True, "protected_namespaces": ()}


class ModelMetricOut(BaseModel):
    metric_name: str
    metric_value: float
    dataset_split: MetricSplit
    model_config = {"from_attributes": True}


class MlModelOut(BaseModel):
    id: uuid.UUID
    experiment_id: uuid.UUID
    name: str
    version: str
    algorithm: Algorithm
    artifact_uri: str
    is_candidate: bool
    is_published: bool
    model_config = {"from_attributes": True}


class ExperimentComparisonRow(BaseModel):
    experiment: MlExperimentOut
    metrics: list[ModelMetricOut]
