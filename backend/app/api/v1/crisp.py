import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.authentication.dependencies import get_current_user, has_permission, require_any_permission, require_permission
from app.database.session import get_db
from app.models.crisp_governance import CrispDocument
from app.models.user import User
from app.schemas.crisp_data import (
    CrispDatasetCreate,
    CrispDatasetOut,
    DataSourceCreate,
    DataSourceOut,
    DatasetVersionCreate,
    DatasetVersionOut,
    EngineeredFeatureCreate,
    EngineeredFeatureOut,
)
from app.schemas.crisp_deployment import (
    DataDriftRecordOut,
    ModelDeploymentCreate,
    ModelDeploymentOut,
    ModelMonitoringOut,
    PredictionOut,
    PredictionRequest,
)
from app.schemas.crisp_evaluation import ModelEvaluationCreate, ModelEvaluationOut
from app.schemas.crisp_governance import CrispAuditLogOut
from app.schemas.crisp_modeling import ExperimentComparisonRow, MlExperimentCreate, MlExperimentOut, MlModelOut, ModelMetricOut
from app.schemas.crisp_phase import (
    CrispPhaseActivityCreate,
    CrispPhaseActivityOut,
    CrispPhaseApprovalRequest,
    CrispPhaseDataUpdate,
    CrispPhaseOut,
)
from app.schemas.crisp_project import CrispProjectCreate, CrispProjectOut, CrispProjectUpdate
from app.services.crisp import (
    audit_service,
    data_service,
    deployment_service,
    evaluation_service,
    modeling_service,
    monitoring_service,
    phase_service,
    preparation_service,
    project_service,
    report_service,
)

router = APIRouter()

_VIEW = Depends(require_permission("crisp:project:view"))
_CREATE = Depends(require_permission("crisp:project:create"))
_EDIT = Depends(require_permission("crisp:project:edit"))
_DELETE = Depends(require_permission("crisp:project:delete"))
_EDIT_DATA = Depends(require_permission("crisp:phase:edit:data"))
_APPROVE = Depends(require_any_permission("crisp:phase:approve", "crisp:phase:approve:business", "crisp:phase:approve:operational"))
_EXPERIMENT = Depends(require_permission("crisp:experiment:manage"))
_PUBLISH = Depends(require_permission("crisp:model:publish"))
_DEPLOY = Depends(require_permission("crisp:deployment:manage"))
_REPORT = Depends(require_permission("crisp:report:download"))
_AUDIT = Depends(require_permission("crisp:audit:view"))

PHASE_EDIT_PERMISSIONS: dict[int, tuple[str, ...]] = {
    1: ("crisp:phase:edit:business",),
    2: ("crisp:phase:edit:data",),
    3: ("crisp:phase:edit:data",),
    4: ("crisp:phase:edit:model",),
    5: ("crisp:phase:edit:business", "crisp:phase:edit:data"),
    6: ("crisp:deployment:manage",),
}


# ---------------------------------------------------------------- Projects --
@router.post("/projects", response_model=CrispProjectOut, dependencies=[_CREATE])
def create_project(payload: CrispProjectCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> CrispProjectOut:
    return project_service.create_project(db, payload, user.id)


@router.get("/projects", response_model=list[CrispProjectOut], dependencies=[_VIEW])
def list_projects(db: Session = Depends(get_db)) -> list[CrispProjectOut]:
    return project_service.list_projects(db)


@router.get("/projects/{project_id}", response_model=CrispProjectOut, dependencies=[_VIEW])
def get_project(project_id: uuid.UUID, db: Session = Depends(get_db)) -> CrispProjectOut:
    return project_service.get_project(db, project_id)


@router.put("/projects/{project_id}", response_model=CrispProjectOut, dependencies=[_EDIT])
def update_project(project_id: uuid.UUID, payload: CrispProjectUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> CrispProjectOut:
    return project_service.update_project(db, project_id, payload, user.id)


@router.delete("/projects/{project_id}", status_code=204, dependencies=[_DELETE])
def delete_project(project_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> None:
    project_service.delete_project(db, project_id, user.id)


# -------------------------------------------------------------------- Phases --
@router.get("/projects/{project_id}/phases", response_model=list[CrispPhaseOut], dependencies=[_VIEW])
def list_phases(project_id: uuid.UUID, db: Session = Depends(get_db)) -> list[CrispPhaseOut]:
    return phase_service.list_phases(db, project_id)


@router.get("/phases/{phase_id}", response_model=CrispPhaseOut, dependencies=[_VIEW])
def get_phase(phase_id: uuid.UUID, db: Session = Depends(get_db)) -> CrispPhaseOut:
    return phase_service.get_phase(db, phase_id)


@router.put("/phases/{phase_id}", response_model=CrispPhaseOut)
def update_phase(phase_id: uuid.UUID, payload: CrispPhaseDataUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> CrispPhaseOut:
    phase = phase_service.get_phase(db, phase_id)
    allowed_perms = PHASE_EDIT_PERMISSIONS.get(phase.phase_number, ())
    if not any(has_permission(user, p) for p in allowed_perms):
        raise HTTPException(status_code=403, detail=f"Missing permission to edit phase {phase.phase_number}: one of {allowed_perms}")
    return phase_service.update_phase_data(db, phase_id, payload.data, user.id)


@router.post("/phases/{phase_id}/activities", response_model=CrispPhaseActivityOut, dependencies=[_VIEW])
def log_phase_activity(phase_id: uuid.UUID, payload: CrispPhaseActivityCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> CrispPhaseActivityOut:
    return phase_service.log_activity(db, phase_id, payload, user.id)


@router.get("/phases/{phase_id}/activities", response_model=list[CrispPhaseActivityOut], dependencies=[_VIEW])
def list_phase_activities(phase_id: uuid.UUID, db: Session = Depends(get_db)) -> list[CrispPhaseActivityOut]:
    return phase_service.list_activities(db, phase_id)


@router.post("/phases/{phase_id}/approve", response_model=CrispPhaseOut, dependencies=[_APPROVE])
def decide_phase(phase_id: uuid.UUID, payload: CrispPhaseApprovalRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> CrispPhaseOut:
    phase = phase_service.decide_phase(db, phase_id, payload, user.id)
    project_service.recompute_progress(db, phase.project_id)
    return phase


# ----------------------------------------------------------- Data understanding --
@router.post("/datasources", response_model=DataSourceOut, dependencies=[_EDIT_DATA])
def create_data_source(payload: DataSourceCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DataSourceOut:
    return data_service.create_data_source(db, payload, user.id)


@router.get("/projects/{project_id}/datasources", response_model=list[DataSourceOut], dependencies=[_VIEW])
def list_data_sources(project_id: uuid.UUID, db: Session = Depends(get_db)) -> list[DataSourceOut]:
    return data_service.list_data_sources(db, project_id)


@router.post("/datasources/{source_id}/compute-eda", response_model=DataSourceOut, dependencies=[_EDIT_DATA])
def compute_eda(source_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DataSourceOut:
    return data_service.compute_eda(db, source_id, user.id)


# ------------------------------------------------------------ Data preparation --
@router.post("/datasets", response_model=CrispDatasetOut, dependencies=[_EDIT_DATA])
def create_dataset(payload: CrispDatasetCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> CrispDatasetOut:
    return preparation_service.create_dataset(db, payload, user.id)


@router.get("/projects/{project_id}/datasets", response_model=list[CrispDatasetOut], dependencies=[_VIEW])
def list_datasets(project_id: uuid.UUID, db: Session = Depends(get_db)) -> list[CrispDatasetOut]:
    return preparation_service.list_datasets(db, project_id)


@router.post("/datasets/{dataset_id}/versions", response_model=DatasetVersionOut, dependencies=[_EDIT_DATA])
def create_dataset_version(dataset_id: uuid.UUID, payload: DatasetVersionCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DatasetVersionOut:
    if payload.dataset_id != dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id in path and body must match")
    return preparation_service.create_dataset_version(db, payload, user.id)


@router.get("/datasets/{dataset_id}/versions", response_model=list[DatasetVersionOut], dependencies=[_VIEW])
def list_dataset_versions(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> list[DatasetVersionOut]:
    return preparation_service.list_dataset_versions(db, dataset_id)


@router.post("/dataset-versions/{version_id}/features", response_model=EngineeredFeatureOut, dependencies=[_EDIT_DATA])
def create_engineered_feature(version_id: uuid.UUID, payload: EngineeredFeatureCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> EngineeredFeatureOut:
    if payload.dataset_version_id != version_id:
        raise HTTPException(status_code=400, detail="dataset_version_id in path and body must match")
    return preparation_service.create_engineered_feature(db, payload, user.id)


@router.get("/dataset-versions/{version_id}/features", response_model=list[EngineeredFeatureOut], dependencies=[_VIEW])
def list_engineered_features(version_id: uuid.UUID, db: Session = Depends(get_db)) -> list[EngineeredFeatureOut]:
    return preparation_service.list_engineered_features(db, version_id)


# -------------------------------------------------------------------- Modeling --
@router.post("/experiments", response_model=MlExperimentOut, dependencies=[_EXPERIMENT])
def create_experiment(payload: MlExperimentCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> MlExperimentOut:
    return modeling_service.create_experiment(db, payload, user.id)


@router.get("/projects/{project_id}/experiments", response_model=list[MlExperimentOut], dependencies=[_VIEW])
def list_experiments(project_id: uuid.UUID, db: Session = Depends(get_db)) -> list[MlExperimentOut]:
    return modeling_service.list_experiments(db, project_id)


@router.get("/experiments/{experiment_id}", response_model=MlExperimentOut, dependencies=[_VIEW])
def get_experiment(experiment_id: uuid.UUID, db: Session = Depends(get_db)) -> MlExperimentOut:
    return modeling_service.get_experiment(db, experiment_id)


@router.post("/experiments/{experiment_id}/train", response_model=MlExperimentOut, dependencies=[_EXPERIMENT])
def train_experiment(experiment_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> MlExperimentOut:
    return modeling_service.train_experiment(db, experiment_id, user.id)


@router.get("/experiments/{experiment_id}/metrics", response_model=list[ModelMetricOut], dependencies=[_VIEW])
def get_experiment_metrics(experiment_id: uuid.UUID, db: Session = Depends(get_db)) -> list[ModelMetricOut]:
    return modeling_service.get_experiment_metrics(db, experiment_id)


@router.get("/projects/{project_id}/experiments/compare", response_model=list[ExperimentComparisonRow], dependencies=[_VIEW])
def compare_experiments(project_id: uuid.UUID, db: Session = Depends(get_db)) -> list[ExperimentComparisonRow]:
    return modeling_service.compare_experiments(db, project_id)


@router.get("/experiments/{experiment_id}/models", response_model=list[MlModelOut], dependencies=[_VIEW])
def list_models(experiment_id: uuid.UUID, db: Session = Depends(get_db)) -> list[MlModelOut]:
    return modeling_service.list_models_for_experiment(db, experiment_id)


@router.post("/models/{model_id}/publish", response_model=MlModelOut, dependencies=[_PUBLISH])
def publish_model(model_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> MlModelOut:
    return modeling_service.publish_model(db, model_id, user.id)


# ------------------------------------------------------------------ Evaluation --
@router.post("/models/{model_id}/evaluate", response_model=ModelEvaluationOut, dependencies=[_APPROVE])
def evaluate_model(model_id: uuid.UUID, payload: ModelEvaluationCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ModelEvaluationOut:
    if payload.model_id != model_id:
        raise HTTPException(status_code=400, detail="model_id in path and body must match")
    return evaluation_service.create_evaluation(db, payload, user.id)


@router.get("/models/{model_id}/evaluations", response_model=list[ModelEvaluationOut], dependencies=[_VIEW])
def list_evaluations(model_id: uuid.UUID, db: Session = Depends(get_db)) -> list[ModelEvaluationOut]:
    return evaluation_service.list_evaluations_for_model(db, model_id)


# ------------------------------------------------------------------ Deployment --
@router.post("/models/{model_id}/deploy", response_model=ModelDeploymentOut, dependencies=[_DEPLOY])
def deploy_model(model_id: uuid.UUID, payload: ModelDeploymentCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ModelDeploymentOut:
    if payload.model_id != model_id:
        raise HTTPException(status_code=400, detail="model_id in path and body must match")
    return deployment_service.deploy_model(db, payload, user.id)


@router.get("/projects/{project_id}/deployments", response_model=list[ModelDeploymentOut], dependencies=[_VIEW])
def list_deployments(project_id: uuid.UUID, db: Session = Depends(get_db)) -> list[ModelDeploymentOut]:
    return deployment_service.list_deployments(db, project_id)


@router.get("/deployments/{deployment_id}", response_model=ModelDeploymentOut, dependencies=[_VIEW])
def get_deployment(deployment_id: uuid.UUID, db: Session = Depends(get_db)) -> ModelDeploymentOut:
    return deployment_service.get_deployment(db, deployment_id)


@router.post("/predictions", response_model=PredictionOut, dependencies=[_DEPLOY])
def create_prediction(payload: PredictionRequest, db: Session = Depends(get_db)) -> PredictionOut:
    """Manual/on-demand inference trigger — the automatic path runs from
    POST /crisp/mining/sensor-readings each time a new reading arrives."""
    return deployment_service.run_inference(db, payload.deployment_id, payload.loading_unit_id)


@router.get("/deployments/{deployment_id}/monitoring", response_model=list[ModelMonitoringOut], dependencies=[_VIEW])
def get_monitoring(deployment_id: uuid.UUID, db: Session = Depends(get_db)) -> list[ModelMonitoringOut]:
    return monitoring_service.list_monitoring(db, deployment_id)


@router.post("/deployments/{deployment_id}/monitoring/refresh", response_model=ModelMonitoringOut, dependencies=[_VIEW])
def refresh_monitoring(deployment_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ModelMonitoringOut:
    return monitoring_service.refresh_monitoring(db, deployment_id, user.id)


@router.post("/deployments/{deployment_id}/drift/data", response_model=DataDriftRecordOut, dependencies=[_VIEW])
def detect_data_drift(deployment_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DataDriftRecordOut:
    return monitoring_service.detect_data_drift(db, deployment_id, user.id)


@router.post("/deployments/{deployment_id}/drift/model", response_model=DataDriftRecordOut, dependencies=[_VIEW])
def detect_model_drift(deployment_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DataDriftRecordOut:
    return monitoring_service.detect_model_drift(db, deployment_id, user.id)


@router.get("/deployments/{deployment_id}/drift", response_model=list[DataDriftRecordOut], dependencies=[_VIEW])
def list_drift(deployment_id: uuid.UUID, db: Session = Depends(get_db)) -> list[DataDriftRecordOut]:
    return monitoring_service.list_drift_records(db, deployment_id)


# --------------------------------------------------------- Audit & reporting --
@router.get("/projects/{project_id}/audit", response_model=list[CrispAuditLogOut], dependencies=[_AUDIT])
def get_audit_log(project_id: uuid.UUID, entity_type: str | None = None, db: Session = Depends(get_db)) -> list[CrispAuditLogOut]:
    return audit_service.list_audit_logs(db, project_id, entity_type)


@router.get("/projects/{project_id}/report", dependencies=[_REPORT])
def get_final_report(
    project_id: uuid.UUID, fmt: str = Query("pdf", pattern="^(pdf|xlsx)$"), db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> FileResponse:
    document = report_service.generate_final_report(db, project_id, fmt, user.id)
    return FileResponse(document.storage_path, filename=document.name)


@router.get("/documents/{document_id}/download", dependencies=[_REPORT])
def download_document(document_id: uuid.UUID, db: Session = Depends(get_db)) -> FileResponse:
    document = db.get(CrispDocument, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(document.storage_path, filename=document.name)
