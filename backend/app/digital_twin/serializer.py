"""Serializes ORM hospital snapshots into DigitalTwinState trees, and hybrid
simulation tick results into WebSocket-ready delta payloads."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.digital_twin.state_manager import BedTwinState, FloorTwinState, HospitalTwinState, RoomTwinState
from app.models.hospital import Bed, Floor, Hospital, Room
from app.simulation.hybrid_engine import HybridTickResult


def build_twin_from_db(db: Session, hospital: Hospital) -> HospitalTwinState:
    twin = HospitalTwinState(id=str(hospital.id), name=hospital.name)
    floors = db.query(Floor).filter(Floor.hospital_id == hospital.id).all()
    for floor in floors:
        floor_state = FloorTwinState(id=str(floor.id), name=floor.name, floor_type=floor.floor_type.value)
        rooms = db.query(Room).filter(Room.floor_id == floor.id).all()
        for room in rooms:
            room_state = RoomTwinState(
                id=str(room.id),
                floor_id=str(floor.id),
                code=room.code,
                room_type=room.room_type.value,
                capacity=room.capacity,
                position=(room.position_x, room.position_y, room.position_z),
            )
            beds = db.query(Bed).filter(Bed.room_id == room.id).all()
            for bed in beds:
                room_state.beds[str(bed.id)] = BedTwinState(
                    id=str(bed.id),
                    room_id=str(room.id),
                    bed_type=bed.bed_type.value,
                    status=bed.status.value,
                    patient_id=str(bed.patient_id) if bed.patient_id else None,
                )
            room_state.recompute_status()
            floor_state.rooms[str(room.id)] = room_state
        twin.floors[str(floor.id)] = floor_state
    return twin


def tick_result_to_ws_payload(simulation_id: str, result: HybridTickResult) -> dict:
    """Compact delta payload broadcast to all subscribed clients each tick."""
    return {
        "type": "simulation_tick",
        "simulation_id": simulation_id,
        "tick": result.tick,
        "census": result.census,
        "saturation_index": round(result.saturation_index, 4),
        "status_band": result.status_band,
        "staff_fatigue": round(result.staff_fatigue, 4),
    }
