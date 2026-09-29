import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.authentication.dependencies import get_current_user, require_permission
from app.database.session import get_db
from app.models.crisp_deployment import DeploymentServiceStatus, ModelDeploymentLoadingUnit
from app.models.sensor import Sensor
from app.models.user import User
from app.schemas.mining import (
    AlertAcknowledge,
    AlertOut,
    DigitalTwinOut,
    LoadingUnitCreate,
    LoadingUnitOut,
    MaintenanceOrderCreate,
    MaintenanceOrderOut,
    SensorCreate,
    SensorOut,
    SensorReadingCreate,
    SensorReadingOut,
)
from app.services import mining_service
from app.services.crisp.deployment_service import get_deployment, run_inference
from app.services.crisp.monitoring_service import refresh_monitoring

router = APIRouter()

_VIEW = Depends(require_permission("crisp:project:view"))
_MANAGE = Depends(require_permission("crisp:mining:manage"))
_ACK = Depends(require_permission("crisp:alert:acknowledge"))
_MAINTAIN = Depends(require_permission("crisp:maintenance:manage"))


@router.post("/loading-units", response_model=LoadingUnitOut, dependencies=[_MANAGE])
def create_loading_unit(payload: LoadingUnitCreate, db: Session = Depends(get_db)) -> LoadingUnitOut:
    return mining_service.create_loading_unit(db, payload)


@router.get("/loading-units", response_model=list[LoadingUnitOut], dependencies=[_VIEW])
def list_loading_units(db: Session = Depends(get_db)) -> list[LoadingUnitOut]:
    return mining_service.list_loading_units(db)


@router.get("/loading-units/{unit_id}", response_model=LoadingUnitOut, dependencies=[_VIEW])
def get_loading_unit(unit_id: uuid.UUID, db: Session = Depends(get_db)) -> LoadingUnitOut:
    unit = mining_service.get_loading_unit(db, unit_id)
    if unit is None:
        raise HTTPException(status_code=404, detail="Loading unit not found")
    return unit


@router.get("/loading-units/{unit_id}/digital-twin", response_model=DigitalTwinOut, dependencies=[_VIEW])
def get_digital_twin(unit_id: uuid.UUID, db: Session = Depends(get_db)) -> DigitalTwinOut:
    twin = mining_service.get_digital_twin(db, unit_id)
    if twin is None:
        raise HTTPException(status_code=404, detail="Digital twin not found for this loading unit")
    return twin


@router.post("/sensors", response_model=SensorOut, dependencies=[_MANAGE])
def create_sensor(payload: SensorCreate, db: Session = Depends(get_db)) -> SensorOut:
    return mining_service.create_sensor(db, payload)


@router.get("/loading-units/{unit_id}/sensors", response_model=list[SensorOut], dependencies=[_VIEW])
def list_sensors(unit_id: uuid.UUID, db: Session = Depends(get_db)) -> list[SensorOut]:
    return mining_service.list_sensors(db, unit_id)


@router.post("/sensor-readings", response_model=SensorReadingOut, dependencies=[_MANAGE])
def ingest_sensor_reading(payload: SensorReadingCreate, db: Session = Depends(get_db)) -> SensorReadingOut:
    """Integration entry point for IoT/SCADA data (steps 1-3 of the required
    flow). If the reading's loading unit has an ACTIVE deployment, inference
    (steps 4-8) runs synchronously right after the reading is persisted."""
    reading = mining_service.ingest_sensor_reading(db, payload)

    sensor_row = db.get(Sensor, payload.sensor_id)
    if sensor_row is not None:
        deployment_link = (
            db.query(ModelDeploymentLoadingUnit)
            .filter(ModelDeploymentLoadingUnit.loading_unit_id == sensor_row.loading_unit_id)
            .first()
        )
        if deployment_link is not None:
            deployment = get_deployment(db, deployment_link.deployment_id)
            if deployment.service_status == DeploymentServiceStatus.ACTIVE:
                run_inference(db, deployment.id, sensor_row.loading_unit_id)
                refresh_monitoring(db, deployment.id, actor_id=None)  # system-triggered, no human actor

    return reading


@router.get("/alerts", response_model=list[AlertOut], dependencies=[_VIEW])
def list_alerts(
    loading_unit_id: uuid.UUID | None = None, status_filter: str | None = None, db: Session = Depends(get_db)
) -> list[AlertOut]:
    return mining_service.list_alerts(db, loading_unit_id, status_filter)


@router.post("/alerts/{alert_id}/acknowledge", response_model=AlertOut, dependencies=[_ACK])
def acknowledge_alert(alert_id: uuid.UUID, payload: AlertAcknowledge, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> AlertOut:
    alert = mining_service.acknowledge_alert(db, alert_id, user.id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.post("/maintenance-orders", response_model=MaintenanceOrderOut, dependencies=[_MAINTAIN])
def create_maintenance_order(payload: MaintenanceOrderCreate, db: Session = Depends(get_db)) -> MaintenanceOrderOut:
    return mining_service.create_maintenance_order(db, payload)


@router.get("/maintenance-orders", response_model=list[MaintenanceOrderOut], dependencies=[_VIEW])
def list_maintenance_orders(loading_unit_id: uuid.UUID | None = None, db: Session = Depends(get_db)) -> list[MaintenanceOrderOut]:
    return mining_service.list_maintenance_orders(db, loading_unit_id)
