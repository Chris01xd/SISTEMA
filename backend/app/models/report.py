import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class ReportFormat(str, enum.Enum):
    DOCX = "DOCX"
    PDF = "PDF"
    XLSX = "XLSX"


class Report(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "reports"

    simulation_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("simulations.id"))
    author_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    format: Mapped[ReportFormat] = mapped_column(Enum(ReportFormat))
    storage_path: Mapped[str] = mapped_column(String(500))
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
