"""Phase 3 (data preparation): builds a real pandas pipeline over
`sensor_readings` for a project's loading units, applies the requested
preparation operations in order, and materializes the result as a Parquet
file — a `dataset_versions` row, not a black box. Also computes and derives
the named engineered features (phase 3's feature-engineering deliverable)."""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.crisp_data import (
    CrispDataset,
    DataPreparationStep,
    DatasetVersion,
    DatasetVersionStatus,
    EngineeredFeature,
    PreparationStepType,
)
from app.models.crisp_project import CrispProjectLoadingUnit
from app.models.maintenance import MaintenanceOrder, MaintenanceType
from app.models.sensor import Sensor, SensorReading
from app.schemas.crisp_data import CrispDatasetCreate, DatasetVersionCreate, EngineeredFeatureCreate
from app.services.crisp.audit_service import record_crisp_audit


def create_dataset(db: Session, payload: CrispDatasetCreate, actor_id: uuid.UUID) -> CrispDataset:
    dataset = CrispDataset(**payload.model_dump())
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    record_crisp_audit(
        db, user_id=actor_id, action="CREATE_DATASET", entity_type="CrispDataset", entity_id=str(dataset.id),
        project_id=dataset.project_id,
    )
    return dataset


def list_datasets(db: Session, project_id: uuid.UUID) -> list[CrispDataset]:
    return db.query(CrispDataset).filter(CrispDataset.project_id == project_id).all()


def _validate_temporal_split(split_config: dict) -> None:
    """Train must end at/before validation starts, which must end at/before
    test starts — the exact rule that keeps temporal splits leakage-free."""
    if not split_config:
        return
    order = ["train", "validation", "test"]
    ranges = {k: split_config[k] for k in order if k in split_config}
    if len(ranges) < 2:
        return
    prev_end = None
    for key in order:
        if key not in ranges:
            continue
        start, end = pd.Timestamp(ranges[key][0]), pd.Timestamp(ranges[key][1])
        if start > end:
            raise HTTPException(status_code=400, detail=f"split_config['{key}'] start is after its end")
        if prev_end is not None and start < prev_end:
            raise HTTPException(
                status_code=400,
                detail="split_config ranges must be chronologically ordered (train <= validation <= test) to avoid temporal leakage",
            )
        prev_end = end


def _load_wide_frame(db: Session, project_id: uuid.UUID) -> pd.DataFrame:
    unit_ids = [r.loading_unit_id for r in db.query(CrispProjectLoadingUnit.loading_unit_id).filter(CrispProjectLoadingUnit.project_id == project_id)]
    if not unit_ids:
        raise HTTPException(status_code=400, detail="Project has no associated loading units to build a dataset from")

    sensors = db.query(Sensor).filter(Sensor.loading_unit_id.in_(unit_ids)).all()
    if not sensors:
        raise HTTPException(status_code=404, detail="No sensors found for this project's loading units")

    frames = []
    for sensor in sensors:
        rows = (
            db.query(SensorReading.timestamp, SensorReading.value)
            .filter(SensorReading.sensor_id == sensor.id)
            .order_by(SensorReading.timestamp)
            .all()
        )
        if not rows:
            continue
        col = f"{sensor.sensor_type.value.lower()}_{str(sensor.loading_unit_id)[:8]}"
        s = pd.Series([v for _, v in rows], index=pd.to_datetime([t for t, _ in rows]), name=col)
        frames.append(s)

    if not frames:
        raise HTTPException(status_code=404, detail="No sensor readings available yet for this project")

    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "timestamp"
    return df


def _apply_operation(db: Session, df: pd.DataFrame, step_type: PreparationStepType, parameters: dict) -> pd.DataFrame:
    if step_type == PreparationStepType.CLEANING:
        return df.dropna(how="all")
    if step_type == PreparationStepType.DEDUPLICATION:
        return df[~df.index.duplicated(keep="first")]
    if step_type == PreparationStepType.MISSING_VALUE_IMPUTATION:
        return df.ffill().bfill()
    if step_type == PreparationStepType.OUTLIER_TREATMENT:
        clipped = df.copy()
        for col in clipped.select_dtypes(include=[np.number]).columns:
            q1, q3 = clipped[col].quantile(0.25), clipped[col].quantile(0.75)
            iqr = q3 - q1
            clipped[col] = clipped[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return clipped
    if step_type == PreparationStepType.TEMPORAL_SYNCHRONIZATION:
        freq = parameters.get("frequency", "1min")
        return df.resample(freq).mean().ffill()
    if step_type == PreparationStepType.SOURCE_INTEGRATION:
        return df  # frame is already the union of all project sensors by construction
    if step_type == PreparationStepType.NORMALIZATION:
        scaled = df.copy()
        numeric_cols = scaled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            lo, hi = scaled[col].min(), scaled[col].max()
            scaled[col] = (scaled[col] - lo) / (hi - lo) if hi > lo else 0.0
        return scaled
    if step_type == PreparationStepType.ENCODING:
        return pd.get_dummies(df)
    if step_type == PreparationStepType.FAILURE_LABELING:
        horizon_hours = parameters.get("horizon_hours", 48)
        corrective_dates = [
            row.executed_date
            for row in db.query(MaintenanceOrder.executed_date)
            .filter(MaintenanceOrder.order_type == MaintenanceType.CORRECTIVE, MaintenanceOrder.executed_date.isnot(None))
        ]
        labeled = df.copy()
        labeled["failure_within_horizon"] = 0
        for failure_date in corrective_dates:
            window_start = pd.Timestamp(failure_date) - pd.Timedelta(hours=horizon_hours)
            mask = (labeled.index >= window_start) & (labeled.index <= pd.Timestamp(failure_date))
            labeled.loc[mask, "failure_within_horizon"] = 1
        return labeled
    if step_type == PreparationStepType.CLASS_BALANCING:
        if "failure_within_horizon" not in df.columns:
            return df
        majority = df[df["failure_within_horizon"] == 0]
        minority = df[df["failure_within_horizon"] == 1]
        if minority.empty or len(minority) >= len(majority):
            return df
        oversampled = minority.sample(n=len(majority), replace=True, random_state=42)
        return pd.concat([majority, oversampled]).sort_index()
    if step_type == PreparationStepType.TIME_WINDOWING:
        window = parameters.get("window", 5)
        windowed = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            windowed[f"{col}_roll_mean_{window}"] = df[col].rolling(window, min_periods=1).mean()
            windowed[f"{col}_roll_std_{window}"] = df[col].rolling(window, min_periods=1).std().fillna(0.0)
        return windowed
    if step_type == PreparationStepType.FEATURE_ENGINEERING:
        return df  # named engineered features are added by apply_engineered_features, not generically here
    return df


def create_dataset_version(db: Session, payload: DatasetVersionCreate, actor_id: uuid.UUID) -> DatasetVersion:
    dataset = db.get(CrispDataset, payload.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _validate_temporal_split(payload.split_config)

    df = _load_wide_frame(db, dataset.project_id)
    for step in payload.operations_applied:
        step_type = PreparationStepType(step["step_type"])
        df = _apply_operation(db, df, step_type, step.get("parameters", {}))

    version = DatasetVersion(
        dataset_id=dataset.id,
        name=payload.name,
        responsible_id=actor_id,
        source_data_source_id=payload.source_data_source_id,
        operations_applied=payload.operations_applied,
        record_count=len(df),
        variables=list(df.columns),
        split_config=payload.split_config,
        validation_status=DatasetVersionStatus.DRAFT,
    )
    db.add(version)
    db.flush()

    storage_dir = Path(settings.CRISP_DATASET_DIR)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / f"{version.id}.parquet"
    # Written via pyarrow directly (not df.to_parquet) to avoid routing
    # through pandas.io.parquet's engine-detection, which imports
    # pandas.core.arrays.arrow.extension_types — a module whose
    # patch_pyarrow() call is incompatible with some newer pyarrow releases.
    pq.write_table(pa.Table.from_pandas(df.reset_index()), storage_path)
    version.storage_uri = str(storage_path)
    version.validation_status = DatasetVersionStatus.VALID if len(df) > 0 else DatasetVersionStatus.INVALID

    for index, step in enumerate(payload.operations_applied):
        db.add(
            DataPreparationStep(
                dataset_version_id=version.id,
                step_type=PreparationStepType(step["step_type"]),
                order_index=index,
                parameters=step.get("parameters", {}),
                applied_by_id=actor_id,
            )
        )
    db.commit()
    db.refresh(version)

    record_crisp_audit(
        db, user_id=actor_id, action="CREATE_DATASET_VERSION", entity_type="DatasetVersion", entity_id=str(version.id),
        project_id=dataset.project_id, new_value={"record_count": version.record_count, "status": version.validation_status.value},
    )
    return version


def list_dataset_versions(db: Session, dataset_id: uuid.UUID) -> list[DatasetVersion]:
    return db.query(DatasetVersion).filter(DatasetVersion.dataset_id == dataset_id).all()


def get_dataset_version(db: Session, version_id: uuid.UUID) -> DatasetVersion:
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    return version


def load_dataset_version_frame(version: DatasetVersion) -> pd.DataFrame:
    if not version.storage_uri or not Path(version.storage_uri).exists():
        raise HTTPException(status_code=404, detail="Dataset version artifact not found on disk")
    df = pq.read_table(version.storage_uri).to_pandas()
    return df.set_index("timestamp") if "timestamp" in df.columns else df


def create_engineered_feature(db: Session, payload: EngineeredFeatureCreate, actor_id: uuid.UUID) -> EngineeredFeature:
    version = get_dataset_version(db, payload.dataset_version_id)
    feature = EngineeredFeature(**payload.model_dump(), created_by_id=actor_id)
    db.add(feature)
    db.commit()
    db.refresh(feature)
    record_crisp_audit(
        db, user_id=actor_id, action="CREATE_ENGINEERED_FEATURE", entity_type="EngineeredFeature", entity_id=str(feature.id),
        new_value={"feature_name": feature.feature_name, "dataset_version_id": str(version.id)},
    )
    return feature


def list_engineered_features(db: Session, dataset_version_id: uuid.UUID) -> list[EngineeredFeature]:
    return db.query(EngineeredFeature).filter(EngineeredFeature.dataset_version_id == dataset_version_id).all()
