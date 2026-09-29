"""Phase lifecycle: edit phase_data, log activities, and approve/reject/
return-to-previous — the mechanics that make CRISP-DM iterative in this
module. A phase can never be approved with missing required fields."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.crisp_data import CrispDataset, DataSource, DatasetVersion, DatasetVersionStatus
from app.models.crisp_deployment import DeploymentServiceStatus, ModelDeployment
from app.models.crisp_evaluation import ModelEvaluation
from app.models.crisp_modeling import ExperimentStatus, MlExperiment, MlModel
from app.models.crisp_phase import CrispPhase, CrispPhaseActivity, CrispPhaseApproval, PhaseApprovalAction, PhaseStatus
from app.models.crisp_project import CrispProject
from app.schemas.crisp_phase import BusinessUnderstandingData, CrispPhaseActivityCreate, CrispPhaseApprovalRequest
from app.services.crisp.audit_service import record_crisp_audit


def get_phase(db: Session, phase_id: uuid.UUID) -> CrispPhase:
    phase = db.get(CrispPhase, phase_id)
    if phase is None:
        raise HTTPException(status_code=404, detail="Phase not found")
    return phase


def list_phases(db: Session, project_id: uuid.UUID) -> list[CrispPhase]:
    return db.query(CrispPhase).filter(CrispPhase.project_id == project_id).order_by(CrispPhase.phase_number).all()


def update_phase_data(db: Session, phase_id: uuid.UUID, data: dict, actor_id: uuid.UUID) -> CrispPhase:
    phase = get_phase(db, phase_id)
    if phase.phase_number == 1:
        # Validates shape only (types/structure); completeness for approval
        # is a separate, stricter check in _missing_required_fields below.
        data = BusinessUnderstandingData.model_validate(data).model_dump(mode="json")

    old_value = phase.phase_data
    phase.phase_data = data
    if phase.status == PhaseStatus.NOT_STARTED:
        phase.status = PhaseStatus.IN_PROGRESS
        phase.started_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(phase)

    record_crisp_audit(
        db, user_id=actor_id, action="UPDATE_PHASE_DATA", entity_type="CrispPhase", entity_id=str(phase.id),
        project_id=phase.project_id, phase_id=phase.id, old_value=old_value, new_value=data,
    )
    return phase


def list_activities(db: Session, phase_id: uuid.UUID) -> list[CrispPhaseActivity]:
    return db.query(CrispPhaseActivity).filter(CrispPhaseActivity.phase_id == phase_id).order_by(CrispPhaseActivity.performed_at).all()


def log_activity(db: Session, phase_id: uuid.UUID, payload: CrispPhaseActivityCreate, actor_id: uuid.UUID) -> CrispPhaseActivity:
    phase = get_phase(db, phase_id)
    activity = CrispPhaseActivity(
        phase_id=phase.id,
        performed_by_id=actor_id,
        performed_at=datetime.now(timezone.utc),
        **payload.model_dump(),
    )
    db.add(activity)
    db.commit()
    db.refresh(activity)

    record_crisp_audit(
        db, user_id=actor_id, action="LOG_PHASE_ACTIVITY", entity_type="CrispPhaseActivity", entity_id=str(activity.id),
        project_id=phase.project_id, phase_id=phase.id, new_value={"description": activity.description},
    )
    return activity


def _missing_required_fields(db: Session, phase: CrispPhase) -> list[str]:
    """Business rule per phase — mirrors the presentation's own requirement
    that a phase cannot be marked done without its minimum deliverable."""
    missing: list[str] = []

    if phase.phase_number == 1:
        data = phase.phase_data or {}
        if not data.get("business_problem"):
            missing.append("problema empresarial")
        if not data.get("general_objective"):
            missing.append("objetivo general")
        if not data.get("business_success_criteria"):
            missing.append("criterios de éxito del negocio")

    elif phase.phase_number == 2:
        count = db.query(DataSource).filter(DataSource.project_id == phase.project_id).count()
        if count == 0:
            missing.append("al menos una fuente de datos registrada")

    elif phase.phase_number == 3:
        dataset_ids = [d.id for d in db.query(CrispDataset.id).filter(CrispDataset.project_id == phase.project_id)]
        valid = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id.in_(dataset_ids), DatasetVersion.validation_status == DatasetVersionStatus.VALID)
            .count()
            if dataset_ids
            else 0
        )
        if valid == 0:
            missing.append("al menos una versión de dataset validada")

    elif phase.phase_number == 4:
        completed = (
            db.query(MlExperiment)
            .filter(MlExperiment.project_id == phase.project_id, MlExperiment.status == ExperimentStatus.COMPLETED)
            .count()
        )
        if completed == 0:
            missing.append("al menos un experimento entrenado (estado COMPLETED)")

    elif phase.phase_number == 5:
        experiment_ids = [e.id for e in db.query(MlExperiment.id).filter(MlExperiment.project_id == phase.project_id)]
        model_ids = [m.id for m in db.query(MlModel.id).filter(MlModel.experiment_id.in_(experiment_ids))] if experiment_ids else []
        evaluated = db.query(ModelEvaluation).filter(ModelEvaluation.model_id.in_(model_ids)).count() if model_ids else 0
        if evaluated == 0:
            missing.append("al menos un modelo evaluado con decisión registrada")

    elif phase.phase_number == 6:
        active = (
            db.query(ModelDeployment)
            .filter(ModelDeployment.project_id == phase.project_id, ModelDeployment.service_status == DeploymentServiceStatus.ACTIVE)
            .count()
        )
        if active == 0:
            missing.append("al menos un despliegue activo")

    return missing


def decide_phase(db: Session, phase_id: uuid.UUID, payload: CrispPhaseApprovalRequest, actor_id: uuid.UUID) -> CrispPhase:
    phase = get_phase(db, phase_id)
    previous_status = phase.status.value

    if payload.action == PhaseApprovalAction.APPROVE:
        missing = _missing_required_fields(db, phase)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"No se puede aprobar la fase: faltan campos obligatorios ({', '.join(missing)}).",
            )
        phase.status = PhaseStatus.APPROVED
        phase.approved_by_id = actor_id
        phase.approved_at = datetime.now(timezone.utc)
        phase.completed_at = phase.approved_at
        project = db.get(CrispProject, phase.project_id)
        if project is not None and phase.phase_number < 6:
            next_phase = (
                db.query(CrispPhase)
                .filter(CrispPhase.project_id == project.id, CrispPhase.phase_number == phase.phase_number + 1)
                .first()
            )
            if next_phase is not None and next_phase.status == PhaseStatus.NOT_STARTED:
                next_phase.status = PhaseStatus.IN_PROGRESS
                next_phase.started_at = datetime.now(timezone.utc)
            project.current_phase_number = phase.phase_number + 1

    elif payload.action == PhaseApprovalAction.REJECT:
        phase.status = PhaseStatus.NEEDS_CORRECTION
        phase.last_rejection_reason = payload.comments

    elif payload.action == PhaseApprovalAction.RETURN_TO_PREVIOUS:
        if not payload.comments:
            raise HTTPException(status_code=400, detail="Debe registrar el motivo para regresar a una fase anterior.")
        if payload.target_phase_number is None or not (1 <= payload.target_phase_number < phase.phase_number):
            raise HTTPException(status_code=400, detail="target_phase_number debe ser una fase anterior válida.")
        phase.status = PhaseStatus.NEEDS_CORRECTION
        phase.last_rejection_reason = payload.comments
        target = (
            db.query(CrispPhase)
            .filter(CrispPhase.project_id == phase.project_id, CrispPhase.phase_number == payload.target_phase_number)
            .first()
        )
        if target is not None:
            target.status = PhaseStatus.IN_PROGRESS
            project = db.get(CrispProject, phase.project_id)
            if project is not None:
                project.current_phase_number = target.phase_number

    db.add(
        CrispPhaseApproval(
            phase_id=phase.id,
            action=payload.action,
            approver_id=actor_id,
            comments=payload.comments,
            previous_status=previous_status,
            new_status=phase.status.value,
        )
    )
    db.commit()
    db.refresh(phase)

    record_crisp_audit(
        db, user_id=actor_id, action=f"PHASE_{payload.action.value}", entity_type="CrispPhase", entity_id=str(phase.id),
        project_id=phase.project_id, phase_id=phase.id,
        old_value={"status": previous_status}, new_value={"status": phase.status.value},
    )
    return phase
