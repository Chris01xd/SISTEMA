"""Phase 4 (modeling): experiment tracking plus real training against a
materialized dataset version. Algorithms reuse the libraries already pinned
in requirements.txt (scikit-learn, xgboost) — same convention as
app/artificial_intelligence/*. LSTM degrades the same way
demand_forecasting.py does when torch isn't installed: a clear 501, not a
silent fake result."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from fastapi import HTTPException
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.crisp_modeling import Algorithm, ExperimentStatus, MetricSplit, MlExperiment, MlModel, ModelMetric, ModelObjective
from app.schemas.crisp_modeling import MlExperimentCreate
from app.services.crisp.audit_service import record_crisp_audit
from app.services.crisp.preparation_service import get_dataset_version, load_dataset_version_frame

CLASSIFICATION_OBJECTIVES = {ModelObjective.FAILURE_CLASSIFICATION, ModelObjective.FAILURE_PREDICTION}
REGRESSION_OBJECTIVES = {ModelObjective.REMAINING_USEFUL_LIFE, ModelObjective.OPERATIONAL_VARIABLE_FORECAST}


def create_experiment(db: Session, payload: MlExperimentCreate, actor_id: uuid.UUID) -> MlExperiment:
    get_dataset_version(db, payload.dataset_version_id)  # 404s early if missing
    experiment = MlExperiment(**payload.model_dump())
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    record_crisp_audit(
        db, user_id=actor_id, action="CREATE_EXPERIMENT", entity_type="MlExperiment", entity_id=str(experiment.id),
        project_id=experiment.project_id, new_value={"algorithm": experiment.algorithm.value, "objective": experiment.objective.value},
    )
    return experiment


def list_experiments(db: Session, project_id: uuid.UUID) -> list[MlExperiment]:
    return db.query(MlExperiment).filter(MlExperiment.project_id == project_id).order_by(MlExperiment.created_at.desc()).all()


def get_experiment(db: Session, experiment_id: uuid.UUID) -> MlExperiment:
    experiment = db.get(MlExperiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


def _temporal_split(df, split_config: dict):
    if split_config and "test" in split_config:
        import pandas as pd

        test_start = pd.Timestamp(split_config["test"][0])
        val_start = pd.Timestamp(split_config.get("validation", [split_config["test"][0]])[0])
        train_df = df[df.index < val_start]
        val_df = df[(df.index >= val_start) & (df.index < test_start)]
        test_df = df[df.index >= test_start]
        if len(train_df) and len(test_df):
            return train_df, val_df, test_df
    # Fallback: chronological 70/15/15 split (index is already time-sorted).
    n = len(df)
    train_end, val_end = int(n * 0.7), int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _fit_model(algorithm: Algorithm, objective: ModelObjective, hyperparameters: dict, X_train, y_train=None):
    if algorithm == Algorithm.RANDOM_FOREST:
        model = (
            RandomForestRegressor(**hyperparameters) if objective in REGRESSION_OBJECTIVES else RandomForestClassifier(**hyperparameters)
        )
        model.fit(X_train, y_train)
        return model
    if algorithm == Algorithm.XGBOOST:
        import xgboost as xgb

        model = xgb.XGBRegressor(**hyperparameters) if objective in REGRESSION_OBJECTIVES else xgb.XGBClassifier(**hyperparameters)
        model.fit(X_train, y_train)
        return model
    if algorithm == Algorithm.ISOLATION_FOREST:
        model = IsolationForest(**{"random_state": 42, **hyperparameters})
        model.fit(X_train)
        return model
    if algorithm == Algorithm.REGRESSION:
        model = LinearRegression(**hyperparameters) if objective in REGRESSION_OBJECTIVES else LogisticRegression(**hyperparameters)
        model.fit(X_train, y_train)
        return model
    if algorithm == Algorithm.LSTM:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="LSTM requires torch, which is not installed by default (see backend/requirements-ai-extra.txt).",
            )
        raise HTTPException(status_code=501, detail="LSTM training pipeline not implemented in this environment.")
    raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algorithm}")


def _compute_metrics(algorithm: Algorithm, objective: ModelObjective, model, X_test, y_test) -> dict[str, float]:
    if algorithm == Algorithm.ISOLATION_FOREST:
        raw_pred = model.predict(X_test)  # -1 anomaly, 1 normal
        anomaly_flag = (raw_pred == -1).astype(int)
        metrics = {"anomaly_rate": float(anomaly_flag.mean())}
        if y_test is not None and len(np.unique(y_test)) > 1:
            metrics["precision"] = float(precision_score(y_test, anomaly_flag, zero_division=0))
            metrics["recall"] = float(recall_score(y_test, anomaly_flag, zero_division=0))
        return metrics

    if objective in REGRESSION_OBJECTIVES:
        pred = model.predict(X_test)
        return {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(root_mean_squared_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else 0.0,
        }

    pred = model.predict(X_test)
    metrics = {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) > 1:
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    tn = int(((y_test == 0) & (pred == 0)).sum())
    fp = int(((y_test == 0) & (pred == 1)).sum())
    metrics["false_alarm_rate"] = round(fp / (fp + tn), 4) if (fp + tn) else 0.0
    return metrics


def train_experiment(db: Session, experiment_id: uuid.UUID, actor_id: uuid.UUID) -> MlExperiment:
    experiment = get_experiment(db, experiment_id)
    version = get_dataset_version(db, experiment.dataset_version_id)
    df = load_dataset_version_frame(version)

    missing_cols = [c for c in experiment.input_variables if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"input_variables not found in dataset version: {missing_cols}")
    if experiment.algorithm != Algorithm.ISOLATION_FOREST and experiment.target_variable not in df.columns:
        raise HTTPException(status_code=400, detail=f"target_variable '{experiment.target_variable}' not found in dataset version")

    experiment.status = ExperimentStatus.TRAINING
    db.commit()

    train_df, _val_df, test_df = _temporal_split(df, version.split_config)
    if train_df.empty or test_df.empty:
        experiment.status = ExperimentStatus.FAILED
        db.commit()
        raise HTTPException(status_code=400, detail="Not enough data to form a non-empty train/test temporal split")

    X_train = train_df[experiment.input_variables].fillna(0)
    X_test = test_df[experiment.input_variables].fillna(0)
    y_train = train_df[experiment.target_variable] if experiment.algorithm != Algorithm.ISOLATION_FOREST else None
    y_test = test_df[experiment.target_variable] if experiment.target_variable in test_df.columns else None

    started = time.monotonic()
    try:
        model = _fit_model(experiment.algorithm, experiment.objective, experiment.hyperparameters, X_train, y_train)
    except HTTPException:
        experiment.status = ExperimentStatus.FAILED
        db.commit()
        raise
    duration = time.monotonic() - started

    metrics = _compute_metrics(experiment.algorithm, experiment.objective, model, X_test, y_test)

    artifact_dir = Path(settings.CRISP_MODEL_ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{experiment.id}.joblib"
    joblib.dump(model, artifact_path)

    experiment.status = ExperimentStatus.COMPLETED
    experiment.trained_at = datetime.now(timezone.utc)
    experiment.duration_seconds = round(duration, 3)
    experiment.model_artifact_uri = str(artifact_path)
    db.commit()

    for name, value in metrics.items():
        db.add(ModelMetric(experiment_id=experiment.id, metric_name=name, metric_value=value, dataset_split=MetricSplit.TEST))

    existing_models = db.query(MlModel).filter(MlModel.experiment_id == experiment.id).count()
    ml_model = MlModel(
        experiment_id=experiment.id,
        name=f"{experiment.objective.value}_{experiment.algorithm.value}",
        version=f"v{existing_models + 1}",
        algorithm=experiment.algorithm,
        artifact_uri=str(artifact_path),
        is_candidate=True,
    )
    db.add(ml_model)
    db.commit()
    db.refresh(experiment)

    record_crisp_audit(
        db, user_id=actor_id, action="TRAIN_EXPERIMENT", entity_type="MlExperiment", entity_id=str(experiment.id),
        project_id=experiment.project_id, new_value={"metrics": metrics, "duration_seconds": experiment.duration_seconds},
    )
    return experiment


def get_experiment_metrics(db: Session, experiment_id: uuid.UUID) -> list[ModelMetric]:
    return db.query(ModelMetric).filter(ModelMetric.experiment_id == experiment_id).all()


def compare_experiments(db: Session, project_id: uuid.UUID) -> list[dict]:
    experiments = list_experiments(db, project_id)
    return [{"experiment": e, "metrics": get_experiment_metrics(db, e.id)} for e in experiments]


def list_models_for_experiment(db: Session, experiment_id: uuid.UUID) -> list[MlModel]:
    return db.query(MlModel).filter(MlModel.experiment_id == experiment_id).all()


def get_model(db: Session, model_id: uuid.UUID) -> MlModel:
    model = db.get(MlModel, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


def publish_model(db: Session, model_id: uuid.UUID, actor_id: uuid.UUID) -> MlModel:
    model = get_model(db, model_id)
    model.is_published = True
    db.commit()
    db.refresh(model)
    record_crisp_audit(
        db, user_id=actor_id, action="PUBLISH_MODEL", entity_type="MlModel", entity_id=str(model.id), new_value={"version": model.version},
    )
    return model
