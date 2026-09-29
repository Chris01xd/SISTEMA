"""CRUD for `crisp_projects`. Creating a project also creates its six
`crisp_phases` rows (phase 1 starts IN_PROGRESS, the rest NOT_STARTED) so the
phase stepper always has something to render."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.crisp_phase import CrispPhase, PhaseStatus
from app.models.crisp_project import CrispProject, CrispProjectLoadingUnit, CrispProjectMember
from app.schemas.crisp_project import CrispProjectCreate, CrispProjectUpdate
from app.services.crisp.audit_service import record_crisp_audit


def create_project(db: Session, payload: CrispProjectCreate, actor_id: uuid.UUID) -> CrispProject:
    data = payload.model_dump(exclude={"loading_unit_ids", "member_ids"})
    project = CrispProject(**data, created_by_id=actor_id, updated_by_id=actor_id)
    db.add(project)
    db.flush()

    for unit_id in payload.loading_unit_ids:
        db.add(CrispProjectLoadingUnit(project_id=project.id, loading_unit_id=unit_id))
    for user_id, role_in_project in payload.member_ids:
        db.add(CrispProjectMember(project_id=project.id, user_id=user_id, role_in_project=role_in_project))

    for phase_number in range(1, 7):
        db.add(
            CrispPhase(
                project_id=project.id,
                phase_number=phase_number,
                status=PhaseStatus.IN_PROGRESS if phase_number == 1 else PhaseStatus.NOT_STARTED,
                started_at=datetime.now(timezone.utc) if phase_number == 1 else None,
            )
        )
    db.commit()
    db.refresh(project)

    record_crisp_audit(
        db, user_id=actor_id, action="CREATE_PROJECT", entity_type="CrispProject", entity_id=str(project.id),
        project_id=project.id, new_value={"name": project.name},
    )
    return project


def list_projects(db: Session) -> list[CrispProject]:
    return db.query(CrispProject).order_by(CrispProject.created_at.desc()).all()


def get_project(db: Session, project_id: uuid.UUID) -> CrispProject:
    project = db.get(CrispProject, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="CRISP-DM project not found")
    return project


def update_project(db: Session, project_id: uuid.UUID, payload: CrispProjectUpdate, actor_id: uuid.UUID) -> CrispProject:
    project = get_project(db, project_id)
    old_value = {"name": project.name, "overall_status": project.overall_status.value}
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(project, field, value)
    project.updated_by_id = actor_id
    db.commit()
    db.refresh(project)

    record_crisp_audit(
        db, user_id=actor_id, action="UPDATE_PROJECT", entity_type="CrispProject", entity_id=str(project.id),
        project_id=project.id, old_value=old_value, new_value={"name": project.name},
    )
    return project


def delete_project(db: Session, project_id: uuid.UUID, actor_id: uuid.UUID) -> None:
    project = get_project(db, project_id)
    db.delete(project)
    db.commit()
    record_crisp_audit(
        db, user_id=actor_id, action="DELETE_PROJECT", entity_type="CrispProject", entity_id=str(project_id),
        project_id=None,
    )


def recompute_progress(db: Session, project_id: uuid.UUID) -> float:
    """% of the six phases currently APPROVED — the simplest, most auditable
    definition of project progress for a six-phase iterative process."""
    phases = db.query(CrispPhase).filter(CrispPhase.project_id == project_id).all()
    if not phases:
        return 0.0
    approved = sum(1 for p in phases if p.status == PhaseStatus.APPROVED)
    percent = round(100.0 * approved / len(phases), 1)
    project = get_project(db, project_id)
    project.progress_percent = percent
    db.commit()
    return percent
