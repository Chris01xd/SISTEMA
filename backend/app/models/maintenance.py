import enum
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class AlertSeverity(str, enum.Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class AlertSource(str, enum.Enum):
    MODEL_PREDICTION = "MODEL_PREDICTION"
    THRESHOLD_RULE = "THRESHOLD_RULE"
    MANUAL = "MANUAL"


class AlertStatus(str, enum.Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class MaintenanceType(str, enum.Enum):
    PREVENTIVE = "PREVENTIVE"
    PREDICTIVE = "PREDICTIVE"
    CORRECTIVE = "CORRECTIVE"


class MaintenanceOrderStatus(str, enum.Enum):
    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class Alert(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "alerts"

    loading_unit_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("loading_units.id"), index=True)
    digital_twin_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("loading_unit_digital_twins.id"), nullable=True
    )
    prediction_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    severity: Mapped[AlertSeverity] = mapped_column(Enum(AlertSeverity))
    source: Mapped[AlertSource] = mapped_column(Enum(AlertSource), default=AlertSource.THRESHOLD_RULE)
    message: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[AlertStatus] = mapped_column(Enum(AlertStatus), default=AlertStatus.OPEN)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    acknowledged_by_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    maintenance_orders: Mapped[list["MaintenanceOrder"]] = relationship(back_populates="origin_alert")


class MaintenanceOrder(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "maintenance_orders"

    loading_unit_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("loading_units.id"), index=True)
    order_type: Mapped[MaintenanceType] = mapped_column(Enum(MaintenanceType))
    origin_alert_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("alerts.id"), nullable=True)
    responsible_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    scheduled_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    executed_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[MaintenanceOrderStatus] = mapped_column(
        Enum(MaintenanceOrderStatus), default=MaintenanceOrderStatus.PENDING
    )
    observations: Mapped[str] = mapped_column(Text, default="")
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    origin_alert: Mapped["Alert"] = relationship(back_populates="maintenance_orders")
