import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class EventType(str, enum.Enum):
    EARTHQUAKE = "EARTHQUAKE"
    EXPLOSION = "EXPLOSION"
    TRANSPORT = "TRANSPORT"
    INDUSTRIAL = "INDUSTRIAL"
    OTHER = "OTHER"


class EmergencyEvent(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "emergency_events"

    hospital_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("hospitals.id"))
    name: Mapped[str] = mapped_column(String(255))
    event_type: Mapped[EventType] = mapped_column(Enum(EventType))
    magnitude: Mapped[float] = mapped_column(Float, default=0.0)
    expected_victims: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # arrival_curve, severity_distribution, resource_shock params, etc.
    params: Mapped[dict] = mapped_column(JSON, default=dict)
