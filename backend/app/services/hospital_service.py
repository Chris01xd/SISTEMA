"""CRUD + snapshot helpers for hospital infrastructure entities."""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models.hospital import Bed, BedStatus, BedType, Floor, FloorType, Hospital, Room, RoomType, ZoneStatus
from app.schemas.hospital import (
    BedCreate,
    FloorCreate,
    HospitalCreate,
    HospitalTwinSnapshot,
    RoomCreate,
    SeedDefaultLayoutRequest,
)


def create_hospital(db: Session, payload: HospitalCreate) -> Hospital:
    hospital = Hospital(**payload.model_dump())
    db.add(hospital)
    db.commit()
    db.refresh(hospital)
    return hospital


def get_hospital(db: Session, hospital_id: uuid.UUID) -> Hospital | None:
    return db.get(Hospital, hospital_id)


def list_floors(db: Session, hospital_id: uuid.UUID) -> list[Floor]:
    return db.query(Floor).filter(Floor.hospital_id == hospital_id).all()


def create_floor(db: Session, hospital_id: uuid.UUID, payload: FloorCreate) -> Floor:
    floor = Floor(hospital_id=hospital_id, **payload.model_dump())
    db.add(floor)
    db.commit()
    db.refresh(floor)
    return floor


def delete_floor(db: Session, floor_id: uuid.UUID) -> None:
    floor = db.get(Floor, floor_id)
    if floor is not None:
        db.delete(floor)
        db.commit()


def create_room(db: Session, floor_id: uuid.UUID, payload: RoomCreate) -> Room:
    room = Room(floor_id=floor_id, status=ZoneStatus.GREEN, **payload.model_dump())
    db.add(room)
    db.commit()
    db.refresh(room)
    return room


def delete_room(db: Session, room_id: uuid.UUID) -> None:
    room = db.get(Room, room_id)
    if room is not None:
        db.delete(room)
        db.commit()


def create_bed(db: Session, room_id: uuid.UUID, payload: BedCreate) -> Bed:
    bed = Bed(room_id=room_id, **payload.model_dump())
    db.add(bed)
    db.commit()
    db.refresh(bed)
    _recompute_room_status(db, room_id)
    return bed


def update_bed_status(db: Session, bed_id: uuid.UUID, status: BedStatus) -> Bed | None:
    bed = db.get(Bed, bed_id)
    if bed is None:
        return None
    bed.status = status
    db.commit()
    db.refresh(bed)
    _recompute_room_status(db, bed.room_id)
    return bed


def delete_bed(db: Session, bed_id: uuid.UUID) -> None:
    bed = db.get(Bed, bed_id)
    if bed is not None:
        room_id = bed.room_id
        db.delete(bed)
        db.commit()
        _recompute_room_status(db, room_id)


def _recompute_room_status(db: Session, room_id: uuid.UUID) -> None:
    """Mirrors the digital twin's saturation_to_color logic (see
    app/digital_twin/state_manager.py) so the REST snapshot and the
    WebSocket-driven twin agree on room status without duplicating state."""
    room = db.get(Room, room_id)
    if room is None:
        return
    beds = db.query(Bed).filter(Bed.room_id == room_id).all()
    if not beds:
        room.status = ZoneStatus.GREEN
    else:
        occupied_ratio = sum(1 for b in beds if b.status == BedStatus.OCCUPIED) / len(beds)
        if occupied_ratio < 0.6:
            room.status = ZoneStatus.GREEN
        elif occupied_ratio < 0.85:
            room.status = ZoneStatus.YELLOW
        else:
            room.status = ZoneStatus.RED
    db.commit()


def _split_beds(total: int, max_per_room: int) -> list[int]:
    """Splits a department's bed count across as many equal-ish-sized rooms
    as needed instead of one giant hall — the 3D twin lays each room's beds
    out in a fixed-size footprint, so a single room with dozens of beds would
    render them overlapping."""
    if total <= 0:
        return []
    n_rooms = max(1, -(-total // max_per_room))  # ceil division
    base, remainder = divmod(total, n_rooms)
    return [base + (1 if i < remainder else 0) for i in range(n_rooms)]


def seed_default_layout(db: Session, hospital_id: uuid.UUID, payload: SeedDefaultLayoutRequest) -> None:
    """Generates a standard ER/ICU/Surgery/Ward layout for a hospital that has
    no floors yet, so the 3D twin has something to render immediately."""
    floor_er = create_floor(db, hospital_id, FloorCreate(level=0, name="Urgencias", floor_type=FloorType.ER))
    floor_icu = create_floor(db, hospital_id, FloorCreate(level=1, name="UCI", floor_type=FloorType.ICU))
    floor_or = create_floor(db, hospital_id, FloorCreate(level=2, name="Quirófanos", floor_type=FloorType.SURGERY))
    floor_ward = create_floor(db, hospital_id, FloorCreate(level=3, name="Hospitalización", floor_type=FloorType.WARD))

    def fill_room(floor_id, code, room_type, capacity, x, z, bed_type, n_beds):
        room = create_room(
            db, floor_id,
            RoomCreate(code=code, room_type=room_type, capacity=capacity, position_x=x, position_y=0, position_z=z),
        )
        for i in range(n_beds):
            create_bed(db, room.id, BedCreate(code=f"{code}-B{i + 1}", bed_type=bed_type))

    def fill_department(floor_id, prefix, room_types, bed_type, total_beds, max_per_room):
        for i, n_beds in enumerate(_split_beds(total_beds, max_per_room)):
            row, col = divmod(i, 3)
            room_type = room_types[i % len(room_types)]
            fill_room(floor_id, f"{prefix}-{i + 1}", room_type, n_beds, col * 6, row * 6, bed_type, n_beds)

    fill_department(floor_er.id, "ER", [RoomType.TRIAGE, RoomType.RESUS], BedType.ER, payload.n_er_beds, 12)
    fill_department(floor_icu.id, "ICU", [RoomType.ICU], BedType.ICU, payload.n_icu_beds, 8)
    fill_department(floor_or.id, "OR", [RoomType.OR], BedType.OR_TABLE, payload.n_or_tables, 3)
    fill_department(floor_ward.id, "WARD", [RoomType.WARD], BedType.WARD, payload.n_ward_beds, 10)


def get_twin_snapshot(db: Session, hospital_id: uuid.UUID) -> HospitalTwinSnapshot | None:
    hospital = db.get(Hospital, hospital_id)
    if hospital is None:
        return None
    floors = db.query(Floor).filter(Floor.hospital_id == hospital_id).all()
    floor_ids = [f.id for f in floors]
    rooms = db.query(Room).filter(Room.floor_id.in_(floor_ids)).all() if floor_ids else []
    room_ids = [r.id for r in rooms]
    beds = db.query(Bed).filter(Bed.room_id.in_(room_ids)).all() if room_ids else []

    return HospitalTwinSnapshot(
        hospital=hospital,  # type: ignore[arg-type]
        floors=floors,  # type: ignore[arg-type]
        rooms=rooms,  # type: ignore[arg-type]
        beds=beds,  # type: ignore[arg-type]
    )
