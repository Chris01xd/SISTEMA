"""CRISP-DM project: the umbrella entity that ties the six iterative phases
to one or more loading units and their digital twin."""
import enum
from datetime import date

from sqlalchemy import Date, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import AuditedEntityMixin, Base, UUIDPKMixin


class ProjectOverallStatus(str, enum.Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    ON_HOLD = "ON_HOLD"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class ProjectMemberRole(str, enum.Enum):
    LEAD = "LEAD"
    MAINTENANCE_ENGINEER = "MAINTENANCE_ENGINEER"
    DATA_SCIENTIST = "DATA_SCIENTIST"
    SUPERVISOR = "SUPERVISOR"
    CONSULTANT = "CONSULTANT"


class CrispProject(Base, AuditedEntityMixin):
    __tablename__ = "crisp_projects"

    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    digital_twin_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("loading_unit_digital_twins.id"), nullable=True
    )
    responsible_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    start_date: Mapped[date] = mapped_column(Date)
    estimated_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    actual_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    current_phase_number: Mapped[int] = mapped_column(Integer, default=1)
    overall_status: Mapped[ProjectOverallStatus] = mapped_column(
        Enum(ProjectOverallStatus), default=ProjectOverallStatus.NOT_STARTED
    )
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0)

    phases: Mapped[list["CrispPhase"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    loading_units: Mapped[list["CrispProjectLoadingUnit"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    members: Mapped[list["CrispProjectMember"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class CrispProjectLoadingUnit(Base, UUIDPKMixin):
    __tablename__ = "crisp_project_loading_units"

    project_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_projects.id"), index=True)
    loading_unit_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("loading_units.id"), index=True)

    project: Mapped["CrispProject"] = relationship(back_populates="loading_units")


class CrispProjectMember(Base, UUIDPKMixin):
    __tablename__ = "crisp_project_members"

    project_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crisp_projects.id"), index=True)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    role_in_project: Mapped[ProjectMemberRole] = mapped_column(Enum(ProjectMemberRole))

    project: Mapped["CrispProject"] = relationship(back_populates="members")
