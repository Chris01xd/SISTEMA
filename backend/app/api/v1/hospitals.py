import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.authentication.dependencies import require_permission
from app.database.session import get_db
from app.models.emergency import EmergencyEvent
from app.schemas.hospital import (
    BedCreate,
    BedOut,
    BedStatusUpdate,
    FloorCreate,
    FloorOut,
    HospitalCreate,
    HospitalOut,
    HospitalTwinSnapshot,
    RoomCreate,
    RoomOut,
    SeedDefaultLayoutRequest,
)
from app.schemas.simulation import EmergencyEventCreate, EmergencyEventOut
from app.services.hospital_service import (
    create_bed,
    create_floor,
    create_hospital,
    create_room,
    delete_bed,
    delete_floor,
    delete_room,
    get_hospital,
    get_twin_snapshot,
    list_floors,
    seed_default_layout,
    update_bed_status,
)

router = APIRouter()

_ADMIN = Depends(require_permission("twin:admin"))


@router.post("", response_model=HospitalOut, dependencies=[_ADMIN])
def create_hospital_endpoint(payload: HospitalCreate, db: Session = Depends(get_db)) -> HospitalOut:
    return create_hospital(db, payload)


@router.get("/{hospital_id}", response_model=HospitalOut)
def get_hospital_endpoint(hospital_id: uuid.UUID, db: Session = Depends(get_db)) -> HospitalOut:
    hospital = get_hospital(db, hospital_id)
    if hospital is None:
        raise HTTPException(status_code=404, detail="Hospital not found")
    return hospital


@router.get("/{hospital_id}/twin-snapshot", response_model=HospitalTwinSnapshot)
def twin_snapshot_endpoint(hospital_id: uuid.UUID, db: Session = Depends(get_db)) -> HospitalTwinSnapshot:
    snapshot = get_twin_snapshot(db, hospital_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Hospital not found")
    return snapshot


@router.post("/{hospital_id}/emergency-events", response_model=EmergencyEventOut)
def create_emergency_event(
    hospital_id: uuid.UUID, payload: EmergencyEventCreate, db: Session = Depends(get_db)
) -> EmergencyEventOut:
    event = EmergencyEvent(
        hospital_id=hospital_id,
        name=payload.name,
        event_type=payload.event_type,
        magnitude=payload.magnitude,
        expected_victims=payload.expected_victims,
        started_at=datetime.now(timezone.utc),
        params={
            "severity_distribution": payload.severity_distribution.model_dump(),
            "arrival_curve": payload.arrival_curve,
            "arrival_window_minutes": payload.arrival_window_minutes,
        },
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


@router.get("/{hospital_id}/floors", response_model=list[FloorOut])
def list_floors_endpoint(hospital_id: uuid.UUID, db: Session = Depends(get_db)) -> list[FloorOut]:
    return list_floors(db, hospital_id)


@router.post("/{hospital_id}/floors", response_model=FloorOut, dependencies=[_ADMIN])
def create_floor_endpoint(
    hospital_id: uuid.UUID, payload: FloorCreate, db: Session = Depends(get_db)
) -> FloorOut:
    return create_floor(db, hospital_id, payload)


@router.delete("/floors/{floor_id}", status_code=204, dependencies=[_ADMIN])
def delete_floor_endpoint(floor_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    delete_floor(db, floor_id)


@router.post("/floors/{floor_id}/rooms", response_model=RoomOut, dependencies=[_ADMIN])
def create_room_endpoint(floor_id: uuid.UUID, payload: RoomCreate, db: Session = Depends(get_db)) -> RoomOut:
    return create_room(db, floor_id, payload)


@router.delete("/rooms/{room_id}", status_code=204, dependencies=[_ADMIN])
def delete_room_endpoint(room_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    delete_room(db, room_id)


@router.post("/rooms/{room_id}/beds", response_model=BedOut, dependencies=[_ADMIN])
def create_bed_endpoint(room_id: uuid.UUID, payload: BedCreate, db: Session = Depends(get_db)) -> BedOut:
    return create_bed(db, room_id, payload)


@router.patch("/beds/{bed_id}", response_model=BedOut, dependencies=[_ADMIN])
def update_bed_endpoint(bed_id: uuid.UUID, payload: BedStatusUpdate, db: Session = Depends(get_db)) -> BedOut:
    bed = update_bed_status(db, bed_id, payload.status)
    if bed is None:
        raise HTTPException(status_code=404, detail="Bed not found")
    return bed


@router.delete("/beds/{bed_id}", status_code=204, dependencies=[_ADMIN])
def delete_bed_endpoint(bed_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    delete_bed(db, bed_id)


@router.post("/{hospital_id}/seed-default-layout", response_model=HospitalTwinSnapshot, dependencies=[_ADMIN])
def seed_default_layout_endpoint(
    hospital_id: uuid.UUID,
    payload: SeedDefaultLayoutRequest = SeedDefaultLayoutRequest(),
    db: Session = Depends(get_db),
) -> HospitalTwinSnapshot:
    """Generates a standard ER/ICU/Surgery/Ward floor/room/bed layout in one
    call, so a freshly created hospital has something for the 3D twin to
    render immediately instead of an empty scene."""
    hospital = get_hospital(db, hospital_id)
    if hospital is None:
        raise HTTPException(status_code=404, detail="Hospital not found")
    if list_floors(db, hospital_id):
        raise HTTPException(status_code=400, detail="Hospital already has floors; refusing to double-seed")
    seed_default_layout(db, hospital_id, payload)
    return get_twin_snapshot(db, hospital_id)
