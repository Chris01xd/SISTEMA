"""Phase 2 (data understanding): source registry + exploratory analysis.

`compute_eda` reads real `sensor_readings` rows (never synthesized here — the
only synthetic data in this module comes from the dev-only
tests/seed_demo_mining.py script) and fills in the source's quality columns
and `eda_report` from an actual pandas computation."""
from __future__ import annotations

import uuid

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.crisp_data import DataQualityLevel, DataSource, DataSourceType
from app.models.sensor import Sensor, SensorReading, SensorType
from app.schemas.crisp_data import DataSourceCreate
from app.services.crisp.audit_service import record_crisp_audit


def create_data_source(db: Session, payload: DataSourceCreate, actor_id: uuid.UUID) -> DataSource:
    source = DataSource(**payload.model_dump())
    db.add(source)
    db.commit()
    db.refresh(source)
    record_crisp_audit(
        db, user_id=actor_id, action="CREATE_DATA_SOURCE", entity_type="DataSource", entity_id=str(source.id),
        project_id=source.project_id, new_value={"name": source.name},
    )
    return source


def list_data_sources(db: Session, project_id: uuid.UUID) -> list[DataSource]:
    return db.query(DataSource).filter(DataSource.project_id == project_id).all()


def get_data_source(db: Session, source_id: uuid.UUID) -> DataSource:
    source = db.get(DataSource, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Data source not found")
    return source


def _quality_level(missing_pct: float, duplicate_pct: float) -> DataQualityLevel:
    if missing_pct <= 2 and duplicate_pct <= 1:
        return DataQualityLevel.HIGH
    if missing_pct <= 10 and duplicate_pct <= 5:
        return DataQualityLevel.MEDIUM
    return DataQualityLevel.LOW


def compute_eda(db: Session, source_id: uuid.UUID, actor_id: uuid.UUID) -> DataSource:
    """Descriptive stats, distribution, time series summary, missing/duplicate/
    outlier detection (IQR), and correlation with a same-unit sensor's series
    — the exploratory-analysis deliverable for phase 2."""
    source = get_data_source(db, source_id)
    if source.source_type != DataSourceType.IOT_SENSOR or source.loading_unit_id is None:
        raise HTTPException(status_code=400, detail="EDA is only computed for IOT_SENSOR sources linked to a loading unit")

    try:
        sensor_type = SensorType(source.measured_variable)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"measured_variable must be a valid sensor type, got '{source.measured_variable}'")

    sensors = db.query(Sensor).filter(Sensor.loading_unit_id == source.loading_unit_id, Sensor.sensor_type == sensor_type).all()
    if not sensors:
        raise HTTPException(status_code=404, detail="No matching sensor found for this data source's unit/variable")
    sensor_ids = [s.id for s in sensors]

    query = db.query(SensorReading.timestamp, SensorReading.value).filter(SensorReading.sensor_id.in_(sensor_ids))
    if source.period_start:
        query = query.filter(SensorReading.timestamp >= source.period_start)
    if source.period_end:
        query = query.filter(SensorReading.timestamp <= source.period_end)
    rows = query.order_by(SensorReading.timestamp).all()

    if not rows:
        raise HTTPException(status_code=404, detail="No readings found in the configured period for this source")

    df = pd.DataFrame(rows, columns=["timestamp", "value"])
    total = len(df)
    duplicates = int(df.duplicated(subset=["timestamp"]).sum())
    missing = int(df["value"].isna().sum())
    valid = df["value"].dropna()

    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = valid[(valid < lower) | (valid > upper)]

    missing_pct = round(100 * missing / total, 2) if total else 0.0
    duplicate_pct = round(100 * duplicates / total, 2) if total else 0.0

    source.record_count = total
    source.missing_values_pct = missing_pct
    source.duplicate_records = duplicates
    source.outlier_count = int(len(outliers))
    source.quality_level = _quality_level(missing_pct, duplicate_pct)
    source.eda_report = {
        "descriptive_statistics": {
            "count": int(valid.count()),
            "mean": round(float(valid.mean()), 4) if not valid.empty else None,
            "std": round(float(valid.std()), 4) if not valid.empty else None,
            "min": round(float(valid.min()), 4) if not valid.empty else None,
            "p25": round(float(q1), 4) if not valid.empty else None,
            "median": round(float(valid.median()), 4) if not valid.empty else None,
            "p75": round(float(q3), 4) if not valid.empty else None,
            "max": round(float(valid.max()), 4) if not valid.empty else None,
        },
        "missing_values": {"count": missing, "pct": missing_pct},
        "duplicates": {"count": duplicates, "pct": duplicate_pct},
        "outliers_iqr": {"count": int(len(outliers)), "lower_bound": round(float(lower), 4), "upper_bound": round(float(upper), 4)},
        "time_series_summary": {
            "period_start": str(df["timestamp"].min()),
            "period_end": str(df["timestamp"].max()),
            "n_points": total,
        },
    }
    db.commit()
    db.refresh(source)

    record_crisp_audit(
        db, user_id=actor_id, action="COMPUTE_EDA", entity_type="DataSource", entity_id=str(source.id),
        project_id=source.project_id, new_value={"quality_level": source.quality_level.value},
    )
    return source
