"""Writes to `crisp_audit_logs`. Called explicitly from every write path in
the CRISP-DM services — richer than the generic AuditEvent/AuditLogMixin
(app.database.base) because it captures project/phase context and an
old/new value diff, both required by the module spec."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.crisp_governance import CrispAuditLog


def record_crisp_audit(
    db: Session,
    *,
    user_id: uuid.UUID | None,
    action: str,
    entity_type: str,
    entity_id: str,
    project_id: uuid.UUID | None = None,
    phase_id: uuid.UUID | None = None,
    old_value: dict | None = None,
    new_value: dict | None = None,
    ip_address: str | None = None,
) -> CrispAuditLog:
    entry = CrispAuditLog(
        user_id=user_id,
        action=action,
        occurred_at=datetime.now(timezone.utc),
        project_id=project_id,
        phase_id=phase_id,
        entity_type=entity_type,
        entity_id=entity_id,
        old_value=old_value or {},
        new_value=new_value or {},
        ip_address=ip_address,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_audit_logs(
    db: Session, project_id: uuid.UUID | None = None, entity_type: str | None = None, limit: int = 200
) -> list[CrispAuditLog]:
    query = db.query(CrispAuditLog)
    if project_id is not None:
        query = query.filter(CrispAuditLog.project_id == project_id)
    if entity_type is not None:
        query = query.filter(CrispAuditLog.entity_type == entity_type)
    return query.order_by(CrispAuditLog.occurred_at.desc()).limit(limit).all()
