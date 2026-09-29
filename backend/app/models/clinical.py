import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class SeverityLevel(int, enum.Enum):
    """START-triage inspired scale: 1 = minor (green) .. 4 = critical (red), 5 = expectant/deceased."""

    MINOR = 1
    DELAYED = 2
    IMMEDIATE = 3
    CRITICAL = 4
    EXPECTANT = 5


class PatientStatus(str, enum.Enum):
    ARRIVED = "ARRIVED"
    TRIAGED = "TRIAGED"
    WAITING = "WAITING"
    IN_TREATMENT = "IN_TREATMENT"
    ICU = "ICU"
    WARD = "WARD"
    DISCHARGED = "DISCHARGED"
    DECEASED = "DECEASED"


class StaffRole(str, enum.Enum):
    DOCTOR = "DOCTOR"
    SURGEON = "SURGEON"
    NURSE = "NURSE"
    PARAMEDIC = "PARAMEDIC"


class Shift(str, enum.Enum):
    DAY = "DAY"
    NIGHT = "NIGHT"
    ON_CALL = "ON_CALL"


class ResourceType(str, enum.Enum):
    VENTILATOR = "VENTILATOR"
    AMBULANCE = "AMBULANCE"
    IMAGING = "IMAGING"
    OR_KIT = "OR_KIT"
    BLOOD_UNIT = "BLOOD_UNIT"


class Patient(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "patients"

    event_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("emergency_events.id"), nullable=True
    )
    room_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("rooms.id"), nullable=True)

    age: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(10))
    diagnosis: Mapped[str] = mapped_column(String(255), default="")
    severity_level: Mapped[SeverityLevel] = mapped_column(Enum(SeverityLevel))
    triage_priority: Mapped[int] = mapped_column(Integer, default=0)
    arrival_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    max_wait_time_minutes: Mapped[int] = mapped_column(Integer, default=60)
    # {"bed_type": "ICU", "needs_surgery": true, "needs_icu": true, "needs_imaging": false}
    resource_needs: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[PatientStatus] = mapped_column(Enum(PatientStatus), default=PatientStatus.ARRIVED)

    attending_staff_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("medical_staff.id"), nullable=True
    )


class MedicalStaff(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "medical_staff"

    hospital_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("hospitals.id"))
    full_name: Mapped[str] = mapped_column(String(255), default="")
    role: Mapped[StaffRole] = mapped_column(Enum(StaffRole))
    specialty: Mapped[str] = mapped_column(String(100), default="general")
    shift: Mapped[Shift] = mapped_column(Enum(Shift), default=Shift.DAY)
    assigned_area: Mapped[str] = mapped_column(String(100), default="")
    capacity: Mapped[int] = mapped_column(Integer, default=4)
    workload_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)


class Resource(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "resources"

    hospital_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("hospitals.id"))
    resource_type: Mapped[ResourceType] = mapped_column(Enum(ResourceType))
    quantity_total: Mapped[int] = mapped_column(Integer, default=0)
    quantity_available: Mapped[int] = mapped_column(Integer, default=0)
