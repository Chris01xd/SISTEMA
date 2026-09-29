import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class SensorType(str, enum.Enum):
    TEMPERATURE = "TEMPERATURE"
    VIBRATION = "VIBRATION"
    HYDRAULIC_PRESSURE = "HYDRAULIC_PRESSURE"
    FUEL_CONSUMPTION = "FUEL_CONSUMPTION"
    ENGINE_RPM = "ENGINE_RPM"
    SPEED = "SPEED"
    OPERATING_HOURS = "OPERATING_HOURS"
    LOAD_WEIGHT = "LOAD_WEIGHT"
    ERROR_CODE = "ERROR_CODE"


class SensorSource(str, enum.Enum):
    IOT = "IOT"
    SCADA = "SCADA"
    MANUAL = "MANUAL"


class ReadingQuality(str, enum.Enum):
    GOOD = "GOOD"
    SUSPECT = "SUSPECT"
    BAD = "BAD"


class Sensor(Base, UUIDPKMixin, TimestampMixin):
    """Catalog entry for a physical/virtual sensor attached to a loading unit."""

    __tablename__ = "sensors"

    loading_unit_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("loading_units.id"), index=True)
    sensor_type: Mapped[SensorType] = mapped_column(Enum(SensorType))
    source: Mapped[SensorSource] = mapped_column(Enum(SensorSource), default=SensorSource.IOT)
    unit_of_measure: Mapped[str] = mapped_column(String(30), default="")
    sampling_frequency_seconds: Mapped[int] = mapped_column(default=60)
    is_active: Mapped[bool] = mapped_column(default=True)

    loading_unit: Mapped["LoadingUnit"] = relationship(back_populates="sensors")
    readings: Mapped[list["SensorReading"]] = relationship(back_populates="sensor", cascade="all, delete-orphan")


class SensorReading(Base, UUIDPKMixin):
    """A single timestamped reading. No TimestampMixin: `timestamp` is the
    sensor's own clock, distinct from row insertion time (`created_at`)."""

    __tablename__ = "sensor_readings"
    __table_args__ = (Index("ix_sensor_readings_sensor_ts", "sensor_id", "timestamp"),)

    sensor_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sensors.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    value: Mapped[float] = mapped_column(Float)
    quality: Mapped[ReadingQuality] = mapped_column(Enum(ReadingQuality), default=ReadingQuality.GOOD)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    sensor: Mapped["Sensor"] = relationship(back_populates="readings")
