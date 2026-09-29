"""The six CRISP-DM phases per project, their logged activities, and the
approve/reject/return audit trail that makes the process iterative."""
import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class PhaseNumber(int, enum.Enum):
    BUSINESS_UNDERSTANDING = 1
    DATA_UNDERSTANDING = 2
    DATA_PREPARATION = 3
    MODELING = 4
    EVALUATION = 5
    DEPLOYMENT = 6


PHASE_NAMES: dict[int, str] = {
    1: "Comprensión del negocio",
    2: "Comprensión de los datos",
    3: "Preparación de los datos",
    4: "Modelado",
    5: "Evaluación",
    6: "Despliegue",
}


class PhaseStatus(str, enum.Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    NEEDS_CORRECTION = "NEEDS_CORRECTION"


class PhaseApprovalAction(str, enum.Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    RETURN_TO_PREVIOUS = "RETURN_TO_PREVIOUS"


class CrispPhase(Base, UUIDPKMixin, TimestampMixin):
    """One row per (project, phase_number). `phase_data` holds the
    phase-specific structured payload (e.g. business-understanding fields for
    phase 1) validated server-side by a phase-specific Pydantic schema."""

    __tablename__ = "crisp_phases"

    project_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_projects.id"), index=True)
    phase_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[PhaseStatus] = mapped_column(Enum(PhaseStatus), default=PhaseStatus.NOT_STARTED)
    phase_data: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_by_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_rejection_reason: Mapped[str] = mapped_column(Text, default="")

    project: Mapped["CrispProject"] = relationship(back_populates="phases")
    activities: Mapped[list["CrispPhaseActivity"]] = relationship(
        back_populates="phase", cascade="all, delete-orphan"
    )
    approvals: Mapped[list["CrispPhaseApproval"]] = relationship(
        back_populates="phase", cascade="all, delete-orphan"
    )


class CrispPhaseActivity(Base, UUIDPKMixin, TimestampMixin):
    """Free-form activity/result/observation log entry within a phase."""

    __tablename__ = "crisp_phase_activities"

    phase_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_phases.id"), index=True)
    description: Mapped[str] = mapped_column(Text)
    result: Mapped[str] = mapped_column(Text, default="")
    observations: Mapped[str] = mapped_column(Text, default="")
    performed_by_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    performed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    phase: Mapped["CrispPhase"] = relationship(back_populates="activities")


class CrispPhaseApproval(Base, UUIDPKMixin, TimestampMixin):
    """Immutable history row for every approve/reject/return-to-previous
    decision — required so an iterative CRISP-DM cycle keeps its reasons."""

    __tablename__ = "crisp_phase_approvals"

    phase_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_phases.id"), index=True)
    action: Mapped[PhaseApprovalAction] = mapped_column(Enum(PhaseApprovalAction))
    approver_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    comments: Mapped[str] = mapped_column(Text, default="")
    previous_status: Mapped[str] = mapped_column(String(30))
    new_status: Mapped[str] = mapped_column(String(30))

    phase: Mapped["CrispPhase"] = relationship(back_populates="approvals")
