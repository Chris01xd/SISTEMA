import uuid
from datetime import date, datetime

from pydantic import BaseModel

from app.models.equipment import LoadingUnitType, OperationalStatus, TwinState
from app.models.maintenance import (
    AlertSeverity,
    AlertSource,
    AlertStatus,
    MaintenanceOrderStatus,
    MaintenanceType,
)
from app.models.sensor import ReadingQuality, SensorSource, SensorType


class LoadingUnitCreate(BaseModel):
    code: str
    unit_type: LoadingUnitType
    brand: str = ""
    model: str = ""
    manufacture_year: int | None = None
    capacity_tons: float = 0.0
    fleet: str = ""
    current_location: str = ""
    operational_status: OperationalStatus = OperationalStatus.OPERATIONAL


class LoadingUnitOut(LoadingUnitCreate):
    id: uuid.UUID
    model_config = {"from_attributes": True}


class DigitalTwinOut(BaseModel):
    id: uuid.UUID
    loading_unit_id: uuid.UUID
    model_reference: str
    calibration_parameters: dict
    current_state: TwinState
    risk_level: float
    last_prediction: dict
    model_config = {"from_attributes": True, "protected_namespaces": ()}


class SensorCreate(BaseModel):
    loading_unit_id: uuid.UUID
    sensor_type: SensorType
    source: SensorSource = SensorSource.IOT
    unit_of_measure: str = ""
    sampling_frequency_seconds: int = 60


class SensorOut(SensorCreate):
    id: uuid.UUID
    is_active: bool
    model_config = {"from_attributes": True}


class SensorReadingCreate(BaseModel):
    sensor_id: uuid.UUID
    timestamp: datetime
    value: float
    quality: ReadingQuality = ReadingQuality.GOOD


class SensorReadingOut(SensorReadingCreate):
    id: uuid.UUID
    model_config = {"from_attributes": True}


class AlertOut(BaseModel):
    id: uuid.UUID
    loading_unit_id: uuid.UUID
    severity: AlertSeverity
    source: AlertSource
    message: str
    status: AlertStatus
    triggered_at: datetime
    acknowledged_by_id: uuid.UUID | None
    acknowledged_at: datetime | None
    model_config = {"from_attributes": True}


class AlertAcknowledge(BaseModel):
    comments: str = ""


class MaintenanceOrderCreate(BaseModel):
    loading_unit_id: uuid.UUID
    order_type: MaintenanceType
    origin_alert_id: uuid.UUID | None = None
    responsible_id: uuid.UUID | None = None
    scheduled_date: date | None = None
    observations: str = ""


class MaintenanceOrderOut(MaintenanceOrderCreate):
    id: uuid.UUID
    status: MaintenanceOrderStatus
    executed_date: date | None
    cost: float | None
    model_config = {"from_attributes": True}
