"""Declarative base + audit mixin shared by all ORM models."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, event
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class UUIDPKMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class AuditedEntityMixin(UUIDPKMixin, TimestampMixin):
    """Shared columns for CRISP-DM governance tables: id, created_at, updated_at,
    created_by, updated_by, status — plus who made each change, for traceability."""

    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    updated_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(30), default="ACTIVE")


class AuditLogMixin:
    """Attaches insert/update audit trail entries for regulatory traceability.

    Any model mixing this in gets its changes appended to `audit_events`
    (see app.models.audit.AuditEvent) via SQLAlchemy ORM events registered
    in app.database.audit.register_audit_listeners().
    """

    __audit__: bool = True


def register_audit_listeners(model_cls) -> None:
    from app.models.audit import AuditEvent
    from app.database.session import SessionLocal

    def _write_audit(mapper, connection, target, action: str) -> None:
        entry = AuditEvent(
            entity_type=target.__class__.__name__,
            entity_id=str(getattr(target, "id", "")),
            action=action,
            snapshot={
                c.key: str(getattr(target, c.key))
                for c in mapper.column_attrs
                if hasattr(target, c.key)
            },
        )
        session = SessionLocal(bind=connection)
        try:
            session.add(entry)
            session.commit()
        finally:
            session.close()

    event.listen(model_cls, "after_insert", lambda m, c, t: _write_audit(m, c, t, "INSERT"))
    event.listen(model_cls, "after_update", lambda m, c, t: _write_audit(m, c, t, "UPDATE"))
