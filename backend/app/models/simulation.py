import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class SimulationStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Simulation(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "simulations"

    event_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("emergency_events.id"))
    created_by: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    horizon_minutes: Mapped[int] = mapped_column(Integer, default=360)
    timestep_minutes: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[SimulationStatus] = mapped_column(Enum(SimulationStatus), default=SimulationStatus.PENDING)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)


class SimulationResult(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "simulation_results"

    simulation_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("simulations.id"), index=True)
    tick: Mapped[int] = mapped_column(Integer, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # stocks/flows snapshot: patients_waiting, beds_occupied, staff_active, saturation_index, ...
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)


class AIPrediction(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "ai_predictions"

    simulation_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("simulations.id"), nullable=True
    )
    dataset_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=True
    )
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(50), default="v1")
    target: Mapped[str] = mapped_column(String(100))
    prediction: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
