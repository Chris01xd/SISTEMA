import uuid

from pydantic import BaseModel

from app.models.hospital import BedStatus, BedType, FloorType, RoomType, ZoneStatus


class HospitalCreate(BaseModel):
    name: str
    location: str = ""
    total_capacity: int = 0


class HospitalOut(HospitalCreate):
    id: uuid.UUID
    model_config = {"from_attributes": True}


class FloorCreate(BaseModel):
    level: int
    name: str
    floor_type: FloorType


class FloorOut(FloorCreate):
    id: uuid.UUID
    hospital_id: uuid.UUID
    model_config = {"from_attributes": True}


class RoomCreate(BaseModel):
    code: str
    room_type: RoomType
    capacity: int = 1
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0


class RoomOut(RoomCreate):
    id: uuid.UUID
    floor_id: uuid.UUID
    status: ZoneStatus
    model_config = {"from_attributes": True}


class BedCreate(BaseModel):
    code: str
    bed_type: BedType
    status: BedStatus = BedStatus.AVAILABLE


class BedOut(BaseModel):
    id: uuid.UUID
    room_id: uuid.UUID
    code: str
    bed_type: BedType
    status: BedStatus
    patient_id: uuid.UUID | None
    model_config = {"from_attributes": True}


class BedStatusUpdate(BaseModel):
    status: BedStatus


class SeedDefaultLayoutRequest(BaseModel):
    """Convenience: generates a standard ER/ICU/Surgery/Ward layout so the 3D
    twin has something to render without hand-building floors/rooms/beds."""

    n_er_beds: int = 48
    n_icu_beds: int = 16
    n_or_tables: int = 6
    n_ward_beds: int = 80


class HospitalTwinSnapshot(BaseModel):
    """Full nested snapshot used to bootstrap the 3D twin on client connect."""

    hospital: HospitalOut
    floors: list[FloorOut]
    rooms: list[RoomOut]
    beds: list[BedOut]
