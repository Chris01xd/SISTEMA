"""Phase 6 (deployment): puts an evaluated model into service and runs real
inference against stored sensor readings — steps 4-8 of the required flow
(model processes variables -> prediction saved -> digital twin updates ->
frontend reads state -> alert if over threshold).

Known, documented simplification: because phase-3 datasets are built as one
wide frame with per-unit-suffixed sensor columns (see preparation_service.
_load_wide_frame), a trained model's `input_variables` are tied to the
specific loading units present at training time. Deploying to a unit not
covered by those columns is rejected rather than silently producing a
meaningless prediction."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.digital_twin.mining_serializer import apply_prediction_to_twin
from app.models.crisp_deployment import (
    DeploymentServiceStatus,
    EquipmentPredictedState,
    ModelDeployment,
    ModelDeploymentLoadingUnit,
    ModelMonitoring,
    ModelPrediction,
)
from app.models.crisp_evaluation import EvaluationDecision, ModelEvaluation
from app.models.crisp_modeling import Algorithm, ModelObjective
from app.models.equipment import LoadingUnit, LoadingUnitDigitalTwin
from app.models.maintenance import Alert, AlertSeverity, AlertSource, AlertStatus
from app.models.sensor import Sensor, SensorReading
from app.schemas.crisp_deployment import ModelDeploymentCreate
from app.services.crisp.audit_service import record_crisp_audit
from app.services.crisp.modeling_service import get_experiment, get_model

STATE_ORDER = [EquipmentPredictedState.NORMAL, EquipmentPredictedState.WARNING, EquipmentPredictedState.CRITICAL]


def _validate_units_covered(input_variables: list[str], loading_unit_ids: list[uuid.UUID]) -> None:
    uncovered = [
        str(unit_id) for unit_id in loading_unit_ids if not any(str(unit_id)[:8] in col for col in input_variables)
    ]
    if uncovered:
        raise HTTPException(
            status_code=400,
            detail=f"The model's training features don't cover these loading units: {uncovered}",
        )


def deploy_model(db: Session, payload: ModelDeploymentCreate, actor_id: uuid.UUID) -> ModelDeployment:
    model = get_model(db, payload.model_id)
    if not model.is_candidate:
        raise HTTPException(status_code=400, detail="Only an approved candidate model (phase 5) can be deployed")
    latest_eval = (
        db.query(ModelEvaluation)
        .filter(ModelEvaluation.model_id == model.id, ModelEvaluation.decision == EvaluationDecision.APPROVED)
        .order_by(ModelEvaluation.evaluated_at.desc())
        .first()
    )
    if latest_eval is None:
        raise HTTPException(status_code=400, detail="Model has no APPROVED evaluation on record (phase 5 required first)")

    experiment = get_experiment(db, model.experiment_id)
    _validate_units_covered(experiment.input_variables, payload.loading_unit_ids)

    deployment = ModelDeployment(
        model_id=model.id,
        project_id=payload.project_id,
        deployed_at=datetime.now(timezone.utc),
        alert_thresholds=payload.alert_thresholds,
        inference_frequency=payload.inference_frequency,
        responsible_id=payload.responsible_id,
        rollback_plan=payload.rollback_plan,
        next_retraining_date=payload.next_retraining_date,
        service_status=DeploymentServiceStatus.ACTIVE,
    )
    db.add(deployment)
    db.flush()
    deployment.api_endpoint = f"/api/v1/crisp/deployments/{deployment.id}/predict"
    for unit_id in payload.loading_unit_ids:
        db.add(ModelDeploymentLoadingUnit(deployment_id=deployment.id, loading_unit_id=unit_id))
    db.commit()
    db.refresh(deployment)

    record_crisp_audit(
        db, user_id=actor_id, action="DEPLOY_MODEL", entity_type="ModelDeployment", entity_id=str(deployment.id),
        project_id=deployment.project_id, new_value={"model_id": str(model.id)},
    )
    return deployment


def list_deployments(db: Session, project_id: uuid.UUID) -> list[ModelDeployment]:
    return db.query(ModelDeployment).filter(ModelDeployment.project_id == project_id).all()


def get_deployment(db: Session, deployment_id: uuid.UUID) -> ModelDeployment:
    deployment = db.get(ModelDeployment, deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment


def _latest_feature_row(db: Session, loading_unit_id: uuid.UUID, input_variables: list[str]) -> pd.DataFrame:
    unit_prefix = str(loading_unit_id)[:8]
    sensors = {s.sensor_type.value.lower(): s for s in db.query(Sensor).filter(Sensor.loading_unit_id == loading_unit_id)}
    values: dict[str, float] = {}
    for column in input_variables:
        if unit_prefix not in column:
            values[column] = 0.0
            continue
        sensor_type = column.replace(f"_{unit_prefix}", "")
        sensor = sensors.get(sensor_type)
        if sensor is None:
            values[column] = 0.0
            continue
        latest = (
            db.query(SensorReading.value)
            .filter(SensorReading.sensor_id == sensor.id)
            .order_by(SensorReading.timestamp.desc())
            .first()
        )
        values[column] = float(latest[0]) if latest else 0.0
    return pd.DataFrame([values], columns=input_variables)


def _severity_and_state(risk_level: float, thresholds: dict) -> tuple[str, EquipmentPredictedState]:
    critical = thresholds.get("critical", 0.75)
    warning = thresholds.get("warning", 0.4)
    if risk_level >= critical:
        return "CRITICAL", EquipmentPredictedState.CRITICAL
    if risk_level >= warning:
        return "WARNING", EquipmentPredictedState.WARNING
    return "NORMAL", EquipmentPredictedState.NORMAL


def run_inference(db: Session, deployment_id: uuid.UUID, loading_unit_id: uuid.UUID) -> ModelPrediction:
    """Steps 4-8 of the required flow. Called by the sensor-reading ingestion
    endpoint right after a reading is persisted (see api/v1/mining.py)."""
    deployment = get_deployment(db, deployment_id)
    if deployment.service_status != DeploymentServiceStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Deployment is not ACTIVE")

    model = get_model(db, deployment.model_id)
    experiment = get_experiment(db, model.experiment_id)
    estimator = joblib.load(model.artifact_uri)

    X = _latest_feature_row(db, loading_unit_id, experiment.input_variables)

    anomaly_type = None
    failure_probability = None
    estimated_rul_hours = None

    if model.algorithm == Algorithm.ISOLATION_FOREST:
        raw = estimator.predict(X)[0]
        is_anomaly = raw == -1
        risk_level = 1.0 if is_anomaly else 0.1
        anomaly_type = "SENSOR_PATTERN_ANOMALY" if is_anomaly else None
    elif experiment.objective in {ModelObjective.REMAINING_USEFUL_LIFE, ModelObjective.OPERATIONAL_VARIABLE_FORECAST}:
        pred = float(estimator.predict(X)[0])
        estimated_rul_hours = pred if experiment.objective == ModelObjective.REMAINING_USEFUL_LIFE else None
        reference_hours = float(experiment.hyperparameters.get("rul_reference_hours", 500))
        risk_level = max(0.0, min(1.0, 1 - (pred / reference_hours))) if experiment.objective == ModelObjective.REMAINING_USEFUL_LIFE else 0.0
    else:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)[0]
            risk_level = float(proba[1]) if len(proba) > 1 else float(proba[0])
            failure_probability = risk_level
        else:
            risk_level = float(estimator.predict(X)[0])
            failure_probability = risk_level

    severity, equipment_state = _severity_and_state(risk_level, deployment.alert_thresholds)
    recommendations = {
        "NORMAL": "Sin acción requerida; continuar monitoreo estándar.",
        "WARNING": "Programar inspección en la próxima parada planificada.",
        "CRITICAL": "Generar orden de mantenimiento predictivo de forma prioritaria.",
    }

    prediction = ModelPrediction(
        deployment_id=deployment.id,
        loading_unit_id=loading_unit_id,
        predicted_at=datetime.now(timezone.utc),
        equipment_state=equipment_state,
        risk_level=round(risk_level, 4),
        anomaly_type=anomaly_type,
        failure_probability=round(failure_probability, 4) if failure_probability is not None else None,
        estimated_rul_hours=round(estimated_rul_hours, 1) if estimated_rul_hours is not None else None,
        severity=severity,
        maintenance_recommendation=recommendations[severity],
        raw_input_snapshot=X.iloc[0].to_dict(),
    )
    db.add(prediction)
    db.flush()

    twin = db.query(LoadingUnitDigitalTwin).filter(LoadingUnitDigitalTwin.loading_unit_id == loading_unit_id).first()
    if twin is not None:
        apply_prediction_to_twin(
            db, twin, equipment_state.value, risk_level,
            {"prediction_id": str(prediction.id), "severity": severity, "predicted_at": prediction.predicted_at.isoformat()},
        )

    if severity in {"WARNING", "CRITICAL"}:
        alert = Alert(
            loading_unit_id=loading_unit_id,
            digital_twin_id=twin.id if twin else None,
            prediction_id=prediction.id,
            severity=AlertSeverity.CRITICAL if severity == "CRITICAL" else AlertSeverity.WARNING,
            source=AlertSource.MODEL_PREDICTION,
            message=f"Riesgo {severity.lower()} detectado (nivel {risk_level:.2f}) por el modelo {model.name} {model.version}.",
            status=AlertStatus.OPEN,
            triggered_at=prediction.predicted_at,
        )
        db.add(alert)
        db.flush()
        prediction.linked_alert_id = alert.id

    db.commit()
    db.refresh(prediction)

    _record_monitoring_tick(db, deployment)
    return prediction


def _record_monitoring_tick(db: Session, deployment: ModelDeployment) -> None:
    """Cheap rolling counters updated on every inference call — the fuller
    accuracy/latency computation lives in monitoring_service.refresh_monitoring."""
    last = (
        db.query(ModelMonitoring)
        .filter(ModelMonitoring.deployment_id == deployment.id)
        .order_by(ModelMonitoring.recorded_at.desc())
        .first()
    )
    prediction_count = (last.prediction_count if last else 0) + 1
    alerts_count = db.query(Alert).filter(Alert.digital_twin_id.isnot(None)).count()  # coarse, refined in monitoring_service
    db.add(
        ModelMonitoring(
            deployment_id=deployment.id,
            recorded_at=datetime.now(timezone.utc),
            prediction_count=prediction_count,
            alerts_generated_count=alerts_count,
        )
    )
    db.commit()
