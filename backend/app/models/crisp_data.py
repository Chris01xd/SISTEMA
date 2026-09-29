"""CRISP-DM phases 2 (data understanding) and 3 (data preparation)."""
import enum
from datetime import date

from sqlalchemy import Date, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class DataSourceType(str, enum.Enum):
    IOT_SENSOR = "IOT_SENSOR"
    SCADA = "SCADA"
    ERP = "ERP"
    MAINTENANCE_ORDERS = "MAINTENANCE_ORDERS"
    FAILURE_HISTORY = "FAILURE_HISTORY"
    MANUFACTURER_DATA = "MANUFACTURER_DATA"
    MANUAL_ENTRY = "MANUAL_ENTRY"


class DataQualityLevel(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNASSESSED = "UNASSESSED"


class PreparationStepType(str, enum.Enum):
    CLEANING = "CLEANING"
    DEDUPLICATION = "DEDUPLICATION"
    MISSING_VALUE_IMPUTATION = "MISSING_VALUE_IMPUTATION"
    OUTLIER_TREATMENT = "OUTLIER_TREATMENT"
    TEMPORAL_SYNCHRONIZATION = "TEMPORAL_SYNCHRONIZATION"
    SOURCE_INTEGRATION = "SOURCE_INTEGRATION"
    NORMALIZATION = "NORMALIZATION"
    ENCODING = "ENCODING"
    FAILURE_LABELING = "FAILURE_LABELING"
    CLASS_BALANCING = "CLASS_BALANCING"
    TIME_WINDOWING = "TIME_WINDOWING"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"


class DatasetVersionStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    VALID = "VALID"
    INVALID = "INVALID"


class DataSource(Base, UUIDPKMixin, TimestampMixin):
    """A registered data source feeding a CRISP-DM project (phase 2)."""

    __tablename__ = "data_sources"

    project_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[DataSourceType] = mapped_column(Enum(DataSourceType))
    loading_unit_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("loading_units.id"), nullable=True
    )
    measured_variable: Mapped[str] = mapped_column(String(100), default="")
    unit_of_measure: Mapped[str] = mapped_column(String(30), default="")
    sampling_frequency: Mapped[str] = mapped_column(String(50), default="")
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    missing_values_pct: Mapped[float] = mapped_column(Float, default=0.0)
    duplicate_records: Mapped[int] = mapped_column(Integer, default=0)
    outlier_count: Mapped[int] = mapped_column(Integer, default=0)
    quality_level: Mapped[DataQualityLevel] = mapped_column(Enum(DataQualityLevel), default=DataQualityLevel.UNASSESSED)
    observations: Mapped[str] = mapped_column(Text, default="")
    # Exploratory analysis: descriptive stats, distributions, correlations,
    # time-series summary, normal-vs-pre-failure comparison — computed by
    # app/services/crisp/data_service.py, not hand-entered.
    eda_report: Mapped[dict] = mapped_column(JSON, default=dict)


class CrispDataset(Base, UUIDPKMixin, TimestampMixin):
    """Logical grouping of a data source into a named dataset that gets
    versioned during preparation. Named `crisp_datasets` (not `datasets`) to
    avoid colliding with the existing hospital-ETL `datasets` table."""

    __tablename__ = "crisp_datasets"

    project_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")

    versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "dataset_versions"

    dataset_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_datasets.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    responsible_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    source_data_source_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("data_sources.id"), nullable=True
    )
    operations_applied: Mapped[list] = mapped_column(JSON, default=list)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    variables: Mapped[list] = mapped_column(JSON, default=list)
    storage_uri: Mapped[str] = mapped_column(String(500), default="")
    # {"train": [start, end], "validation": [start, end], "test": [start, end]}
    # ranges must be strictly ordered in time — enforced in preparation_service.
    split_config: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_status: Mapped[DatasetVersionStatus] = mapped_column(
        Enum(DatasetVersionStatus), default=DatasetVersionStatus.DRAFT
    )

    dataset: Mapped["CrispDataset"] = relationship(back_populates="versions")
    preparation_steps: Mapped[list["DataPreparationStep"]] = relationship(
        back_populates="dataset_version", cascade="all, delete-orphan"
    )
    engineered_features: Mapped[list["EngineeredFeature"]] = relationship(
        back_populates="dataset_version", cascade="all, delete-orphan"
    )


class DataPreparationStep(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "data_preparation_steps"

    dataset_version_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("dataset_versions.id"), index=True)
    step_type: Mapped[PreparationStepType] = mapped_column(Enum(PreparationStepType))
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    applied_by_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    dataset_version: Mapped["DatasetVersion"] = relationship(back_populates="preparation_steps")


class EngineeredFeature(Base, UUIDPKMixin, TimestampMixin):
    """Derived variables such as rolling temperature average, vibration
    variance, pressure trend, hours since last maintenance, alarms per
    shift, cumulative operating hours, pre-failure behavior windows."""

    __tablename__ = "engineered_features"

    dataset_version_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("dataset_versions.id"), index=True)
    feature_name: Mapped[str] = mapped_column(String(150))
    definition: Mapped[str] = mapped_column(Text, default="")
    source_variables: Mapped[list] = mapped_column(JSON, default=list)
    created_by_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    dataset_version: Mapped["DatasetVersion"] = relationship(back_populates="engineered_features")
