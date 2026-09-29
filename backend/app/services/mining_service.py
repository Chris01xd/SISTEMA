"""CRUD + ingestion for the mining domain: loading units, their digital
twins, sensors/readings, alerts and maintenance orders."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.equipment import LoadingUnit, LoadingUnitDigitalTwin
from app.models.maintenance import Alert, MaintenanceOrder
from app.models.sensor import Sensor, SensorReading
from app.schemas.mining import (
    LoadingUnitCreate,
    MaintenanceOrderCreate,
    SensorCreate,
    SensorReadingCreate,
)


def create_loading_unit(db: Session, payload: LoadingUnitCreate) -> LoadingUnit:
    unit = LoadingUnit(**payload.model_dump())
    db.add(unit)
    db.flush()
    db.add(LoadingUnitDigitalTwin(loading_unit_id=unit.id))
    db.commit()
    db.refresh(unit)
    return unit


def list_loading_units(db: Session) -> list[LoadingUnit]:
    return db.query(LoadingUnit).order_by(LoadingUnit.code).all()


def get_loading_unit(db: Session, unit_id: uuid.UUID) -> LoadingUnit | None:
    return db.get(LoadingUnit, unit_id)


def get_digital_twin(db: Session, unit_id: uuid.UUID) -> LoadingUnitDigitalTwin | None:
    return db.query(LoadingUnitDigitalTwin).filter(LoadingUnitDigitalTwin.loading_unit_id == unit_id).first()


def create_sensor(db: Session, payload: SensorCreate) -> Sensor:
    sensor = Sensor(**payload.model_dump())
    db.add(sensor)
    db.commit()
    db.refresh(sensor)
    return sensor


def list_sensors(db: Session, loading_unit_id: uuid.UUID) -> list[Sensor]:
    return db.query(Sensor).filter(Sensor.loading_unit_id == loading_unit_id).all()


def ingest_sensor_reading(db: Session, payload: SensorReadingCreate) -> SensorReading:
    """Step 2-3 of the required flow: validate (Pydantic already did) and
    persist. Callers (the router) trigger inference afterward — kept out of
    this function so ingestion always succeeds even if inference fails."""
    reading = SensorReading(**payload.model_dump(), created_at=datetime.now(timezone.utc))
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


def list_alerts(db: Session, loading_unit_id: uuid.UUID | None = None, status_filter: str | None = None) -> list[Alert]:
    query = db.query(Alert)
    if loading_unit_id is not None:
        query = query.filter(Alert.loading_unit_id == loading_unit_id)
    if status_filter is not None:
        query = query.filter(Alert.status == status_filter)
    return query.order_by(Alert.triggered_at.desc()).all()


def acknowledge_alert(db: Session, alert_id: uuid.UUID, user_id: uuid.UUID) -> Alert | None:
    alert = db.get(Alert, alert_id)
    if alert is None:
        return None
    alert.status = "ACKNOWLEDGED"
    alert.acknowledged_by_id = user_id
    alert.acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return alert


def create_maintenance_order(db: Session, payload: MaintenanceOrderCreate) -> MaintenanceOrder:
    order = MaintenanceOrder(**payload.model_dump())
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


def list_maintenance_orders(db: Session, loading_unit_id: uuid.UUID | None = None) -> list[MaintenanceOrder]:
    query = db.query(MaintenanceOrder)
    if loading_unit_id is not None:
        query = query.filter(MaintenanceOrder.loading_unit_id == loading_unit_id)
    return query.order_by(MaintenanceOrder.created_at.desc()).all()
