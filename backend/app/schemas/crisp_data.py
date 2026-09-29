import uuid
from datetime import date

from pydantic import BaseModel

from app.models.crisp_data import (
    DataQualityLevel,
    DataSourceType,
    DatasetVersionStatus,
    PreparationStepType,
)


class DataSourceCreate(BaseModel):
    project_id: uuid.UUID
    name: str
    source_type: DataSourceType
    loading_unit_id: uuid.UUID | None = None
    measured_variable: str = ""
    unit_of_measure: str = ""
    sampling_frequency: str = ""
    period_start: date | None = None
    period_end: date | None = None
    observations: str = ""


class DataSourceOut(DataSourceCreate):
    id: uuid.UUID
    record_count: int
    missing_values_pct: float
    duplicate_records: int
    outlier_count: int
    quality_level: DataQualityLevel
    eda_report: dict
    model_config = {"from_attributes": True}


class CrispDatasetCreate(BaseModel):
    project_id: uuid.UUID
    name: str
    description: str = ""


class CrispDatasetOut(CrispDatasetCreate):
    id: uuid.UUID
    model_config = {"from_attributes": True}


class DatasetVersionCreate(BaseModel):
    dataset_id: uuid.UUID
    name: str
    source_data_source_id: uuid.UUID | None = None
    # ordered list of {"step_type": "...", "parameters": {...}}
    operations_applied: list[dict] = []
    # {"train": ["2026-01-01","2026-03-01"], "validation": [...], "test": [...]}
    split_config: dict = {}


class DatasetVersionOut(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    name: str
    responsible_id: uuid.UUID | None
    source_data_source_id: uuid.UUID | None
    operations_applied: list
    record_count: int
    variables: list
    storage_uri: str
    split_config: dict
    validation_status: DatasetVersionStatus
    model_config = {"from_attributes": True}


class DataPreparationStepOut(BaseModel):
    id: uuid.UUID
    dataset_version_id: uuid.UUID
    step_type: PreparationStepType
    order_index: int
    parameters: dict
    applied_by_id: uuid.UUID | None
    model_config = {"from_attributes": True}


class EngineeredFeatureCreate(BaseModel):
    dataset_version_id: uuid.UUID
    feature_name: str
    definition: str = ""
    source_variables: list[str] = []


class EngineeredFeatureOut(EngineeredFeatureCreate):
    id: uuid.UUID
    created_by_id: uuid.UUID | None
    model_config = {"from_attributes": True}
