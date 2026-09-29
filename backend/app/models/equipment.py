import enum

from sqlalchemy import Enum, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class LoadingUnitType(str, enum.Enum):
    EXCAVATOR = "EXCAVATOR"
    WHEEL_LOADER = "WHEEL_LOADER"
    HAUL_TRUCK = "HAUL_TRUCK"
    SHOVEL = "SHOVEL"


class OperationalStatus(str, enum.Enum):
    OPERATIONAL = "OPERATIONAL"
    MAINTENANCE = "MAINTENANCE"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class TwinState(str, enum.Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class LoadingUnit(Base, UUIDPKMixin, TimestampMixin):
    """A loading/haul unit operating in the open-pit mine (excavator, wheel
    loader, haul truck, shovel) — the physical asset the digital twin mirrors."""

    __tablename__ = "loading_units"

    code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    unit_type: Mapped[LoadingUnitType] = mapped_column(Enum(LoadingUnitType))
    brand: Mapped[str] = mapped_column(String(100), default="")
    model: Mapped[str] = mapped_column(String(100), default="")
    manufacture_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    capacity_tons: Mapped[float] = mapped_column(Float, default=0.0)
    fleet: Mapped[str] = mapped_column(String(100), default="")
    current_location: Mapped[str] = mapped_column(String(255), default="")
    operational_status: Mapped[OperationalStatus] = mapped_column(
        Enum(OperationalStatus), default=OperationalStatus.OPERATIONAL
    )

    digital_twin: Mapped["LoadingUnitDigitalTwin"] = relationship(
        back_populates="loading_unit", uselist=False, cascade="all, delete-orphan"
    )
    sensors: Mapped[list["Sensor"]] = relationship(back_populates="loading_unit", cascade="all, delete-orphan")


class LoadingUnitDigitalTwin(Base, UUIDPKMixin, TimestampMixin):
    """Virtual replica of a LoadingUnit: calibration parameters, last sync,
    and the current derived state consumed by the frontend twin view."""

    __tablename__ = "loading_unit_digital_twins"

    loading_unit_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("loading_units.id"), unique=True, index=True
    )
    model_reference: Mapped[str] = mapped_column(String(100), default="")
    calibration_parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    last_synchronized_at: Mapped[str | None] = mapped_column(String(50), nullable=True)
    current_state: Mapped[TwinState] = mapped_column(Enum(TwinState), default=TwinState.NORMAL)
    risk_level: Mapped[float] = mapped_column(Float, default=0.0)
    last_prediction: Mapped[dict] = mapped_column(JSON, default=dict)

    loading_unit: Mapped["LoadingUnit"] = relationship(back_populates="digital_twin")
