"""Phase 6 monitoring: accuracy/latency tracking plus data- and model-drift
detection, all computed from real rows (no fabricated numbers).

- "Current accuracy" is a precision proxy from *resolved-vs-dismissed*
  alerts: an alert a maintenance engineer resolves (led to a real
  intervention) counts as a true positive corroboration; one they dismiss
  counts as a false positive. It stays null until there are enough
  outcomes (>=5) to be meaningful — never guessed.
- Data drift uses the Population Stability Index (PSI) between each input
  variable's training-time distribution and its current live distribution.
- Model drift compares that same resolved/dismissed precision proxy against
  the model's own test-set metric recorded at training time.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.crisp_deployment import DataDriftRecord, DriftType, ModelDeployment, ModelMonitoring, ModelPrediction
from app.models.crisp_modeling import MetricSplit, ModelMetric
from app.models.maintenance import Alert, AlertStatus
from app.models.sensor import Sensor, SensorReading
from app.services.crisp.audit_service import record_crisp_audit
from app.services.crisp.deployment_service import get_deployment
from app.services.crisp.modeling_service import get_experiment, get_model
from app.services.crisp.preparation_service import get_dataset_version, load_dataset_version_frame


def _psi(baseline: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    baseline, current = baseline.dropna(), current.dropna()
    if baseline.empty or current.empty:
        return 0.0
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    # Open the outer edges to +/-inf so a `current` sample that has drifted
    # entirely outside the baseline's observed range still lands in the
    # tail buckets instead of being silently dropped by np.histogram (which
    # otherwise ignores out-of-range values and can divide by a zero sum —
    # exactly the case that most indicates drift).
    open_edges = edges.copy()
    open_edges[0], open_edges[-1] = -np.inf, np.inf
    base_counts, _ = np.histogram(baseline, bins=open_edges)
    curr_counts, _ = np.histogram(current, bins=open_edges)
    base_pct = np.clip(base_counts / base_counts.sum(), 1e-4, None)
    curr_pct = np.clip(curr_counts / curr_counts.sum(), 1e-4, None)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def _precision_proxy(db: Session, deployment_id: uuid.UUID, since: datetime) -> float | None:
    prediction_ids = [
        p.id for p in db.query(ModelPrediction.id).filter(ModelPrediction.deployment_id == deployment_id, ModelPrediction.predicted_at >= since)
    ]
    if not prediction_ids:
        return None
    resolved = db.query(Alert).filter(Alert.prediction_id.in_(prediction_ids), Alert.status == AlertStatus.RESOLVED).count()
    dismissed = db.query(Alert).filter(Alert.prediction_id.in_(prediction_ids), Alert.status == AlertStatus.DISMISSED).count()
    if resolved + dismissed < 5:
        return None
    return round(resolved / (resolved + dismissed), 4)


def refresh_monitoring(
    db: Session, deployment_id: uuid.UUID, actor_id: uuid.UUID | None, inference_latency_ms: float | None = None
) -> ModelMonitoring:
    """`actor_id` is accepted (not just for API symmetry with the other crisp
    services) so a future audit hook has it available; monitoring refreshes
    themselves are system-computed facts, not user actions, so nothing is
    written to crisp_audit_logs here."""
    deployment = get_deployment(db, deployment_id)
    window_start = datetime.now(timezone.utc) - timedelta(days=7)

    prediction_count = db.query(ModelPrediction).filter(ModelPrediction.deployment_id == deployment_id).count()
    prediction_ids_window = [
        p.id for p in db.query(ModelPrediction.id).filter(ModelPrediction.deployment_id == deployment_id, ModelPrediction.predicted_at >= window_start)
    ]
    alerts_generated = db.query(Alert).filter(Alert.prediction_id.in_(prediction_ids_window)).count() if prediction_ids_window else 0
    false_alarms = (
        db.query(Alert).filter(Alert.prediction_id.in_(prediction_ids_window), Alert.status == AlertStatus.DISMISSED).count()
        if prediction_ids_window
        else 0
    )

    current_accuracy = _precision_proxy(db, deployment_id, window_start)
    needs_retraining = False
    if current_accuracy is not None:
        baseline_row = (
            db.query(ModelMetric)
            .filter(ModelMetric.experiment_id == get_model(db, deployment.model_id).experiment_id, ModelMetric.metric_name == "precision")
            .first()
        )
        if baseline_row and baseline_row.metric_value > 0:
            relative_drop = (baseline_row.metric_value - current_accuracy) / baseline_row.metric_value
            needs_retraining = relative_drop > settings.MODEL_DRIFT_ACCURACY_DROP_THRESHOLD
    if deployment.next_retraining_date and deployment.next_retraining_date <= date.today():
        needs_retraining = True

    record = ModelMonitoring(
        deployment_id=deployment_id,
        recorded_at=datetime.now(timezone.utc),
        current_accuracy=current_accuracy,
        inference_latency_ms=inference_latency_ms,
        prediction_count=prediction_count,
        alerts_generated_count=alerts_generated,
        false_alarm_count=false_alarms,
        needs_retraining=needs_retraining,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_monitoring(db: Session, deployment_id: uuid.UUID, limit: int = 100) -> list[ModelMonitoring]:
    return (
        db.query(ModelMonitoring)
        .filter(ModelMonitoring.deployment_id == deployment_id)
        .order_by(ModelMonitoring.recorded_at.desc())
        .limit(limit)
        .all()
    )


def detect_data_drift(db: Session, deployment_id: uuid.UUID, actor_id: uuid.UUID) -> DataDriftRecord:
    deployment = get_deployment(db, deployment_id)
    model = get_model(db, deployment.model_id)
    experiment = get_experiment(db, model.experiment_id)
    baseline_df = load_dataset_version_frame(get_dataset_version(db, experiment.dataset_version_id))

    since = datetime.now(timezone.utc) - timedelta(days=7)
    all_sensors = db.query(Sensor).all()

    per_variable: dict[str, float] = {}
    for column in experiment.input_variables:
        if column not in baseline_df.columns:
            continue
        # Reverse the "{sensor_type}_{unit_id[:8]}" naming convention from
        # preparation_service._load_wide_frame to find the matching sensor.
        matching_sensor = next(
            (s for s in all_sensors if column == f"{s.sensor_type.value.lower()}_{str(s.loading_unit_id)[:8]}"), None
        )
        if matching_sensor is None:
            continue
        current_values = (
            db.query(SensorReading.value)
            .filter(SensorReading.sensor_id == matching_sensor.id, SensorReading.timestamp >= since)
            .all()
        )
        current_series = pd.Series([v for (v,) in current_values]) if current_values else pd.Series(dtype=float)
        per_variable[column] = round(_psi(baseline_df[column], current_series), 4)

    if not per_variable:
        raise HTTPException(status_code=404, detail="No comparable readings available to compute data drift")

    worst_variable = max(per_variable, key=per_variable.get)
    worst_score = per_variable[worst_variable]
    is_significant = worst_score > settings.DATA_DRIFT_PSI_THRESHOLD

    record = DataDriftRecord(
        deployment_id=deployment_id,
        detected_at=datetime.now(timezone.utc),
        drift_type=DriftType.DATA_DRIFT,
        variable_name=worst_variable,
        drift_score=worst_score,
        method="PSI",
        threshold_used=settings.DATA_DRIFT_PSI_THRESHOLD,
        is_significant=is_significant,
        details={"per_variable_psi": per_variable},
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    record_crisp_audit(
        db, user_id=actor_id, action="DETECT_DATA_DRIFT", entity_type="DataDriftRecord", entity_id=str(record.id),
        new_value={"worst_variable": worst_variable, "psi": worst_score, "is_significant": is_significant},
    )
    return record


def detect_model_drift(db: Session, deployment_id: uuid.UUID, actor_id: uuid.UUID) -> DataDriftRecord:
    deployment = get_deployment(db, deployment_id)
    model = get_model(db, deployment.model_id)
    current_accuracy = _precision_proxy(db, deployment_id, datetime.now(timezone.utc) - timedelta(days=30))
    if current_accuracy is None:
        raise HTTPException(status_code=404, detail="Not enough resolved/dismissed alerts yet to estimate model drift")

    baseline_row = db.query(ModelMetric).filter(ModelMetric.experiment_id == model.experiment_id, ModelMetric.metric_name == "precision").first()
    baseline = baseline_row.metric_value if baseline_row else None
    relative_drop = round((baseline - current_accuracy) / baseline, 4) if baseline else 0.0
    is_significant = relative_drop > settings.MODEL_DRIFT_ACCURACY_DROP_THRESHOLD

    record = DataDriftRecord(
        deployment_id=deployment_id,
        detected_at=datetime.now(timezone.utc),
        drift_type=DriftType.MODEL_DRIFT,
        variable_name="precision_proxy",
        drift_score=relative_drop,
        method="RELATIVE_METRIC_DROP",
        threshold_used=settings.MODEL_DRIFT_ACCURACY_DROP_THRESHOLD,
        is_significant=is_significant,
        details={"baseline_precision": baseline, "current_precision_proxy": current_accuracy},
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    record_crisp_audit(
        db, user_id=actor_id, action="DETECT_MODEL_DRIFT", entity_type="DataDriftRecord", entity_id=str(record.id),
        new_value={"relative_drop": relative_drop, "is_significant": is_significant},
    )
    return record


def list_drift_records(db: Session, deployment_id: uuid.UUID) -> list[DataDriftRecord]:
    return db.query(DataDriftRecord).filter(DataDriftRecord.deployment_id == deployment_id).order_by(DataDriftRecord.detected_at.desc()).all()
