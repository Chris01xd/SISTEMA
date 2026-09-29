"""Evidence/document attachments and the CRISP-DM-specific audit trail.

`CrispAuditLog` is deliberately richer than the generic `AuditEvent`
(app.models.audit): it captures project/phase context and an old/new value
diff, which the spec requires and the generic snapshot-only audit does not
provide. Written explicitly by app/services/crisp/audit_service.py."""
import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class DocumentType(str, enum.Enum):
    EVIDENCE = "EVIDENCE"
    FINAL_REPORT_PDF = "FINAL_REPORT_PDF"
    FINAL_REPORT_XLSX = "FINAL_REPORT_XLSX"
    OTHER = "OTHER"


class CrispDocument(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "crisp_documents"

    project_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_projects.id"), index=True)
    phase_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_phases.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255))
    document_type: Mapped[DocumentType] = mapped_column(Enum(DocumentType), default=DocumentType.EVIDENCE)
    storage_path: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text, default="")
    uploaded_by_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)


class CrispAuditLog(Base, UUIDPKMixin):
    __tablename__ = "crisp_audit_logs"

    user_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    project_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_projects.id"), nullable=True)
    phase_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_phases.id"), nullable=True)
    entity_type: Mapped[str] = mapped_column(String(100))
    entity_id: Mapped[str] = mapped_column(String(64))
    old_value: Mapped[dict] = mapped_column(JSON, default=dict)
    new_value: Mapped[dict] = mapped_column(JSON, default=dict)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
