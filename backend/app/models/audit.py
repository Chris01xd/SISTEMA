from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class AuditEvent(Base, UUIDPKMixin, TimestampMixin):
    """Immutable audit trail row. Written automatically for audited models."""

    __tablename__ = "audit_events"

    entity_type: Mapped[str] = mapped_column(String(100), index=True)
    entity_id: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(20))  # INSERT / UPDATE / DELETE
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
