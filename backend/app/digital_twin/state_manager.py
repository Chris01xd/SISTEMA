"""DigitalTwinStateManager: canonical live state of the physical hospital.

Owns a HospitalDigitalTwinState tree (hospital -> floors -> rooms -> beds),
each carrying a status_color derived from local saturation. Emits compact
delta payloads consumed by the WebSocket hub (see app/services/event_bus.py)
so the 3D frontend re-renders only what changed instead of the full scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def saturation_to_color(saturation_index: float) -> str:
    if saturation_index < 0.6:
        return "GREEN"
    if saturation_index < 0.85:
        return "YELLOW"
    return "RED"


@dataclass
class BedTwinState:
    id: str
    room_id: str
    bed_type: str
    status: str  # AVAILABLE | OCCUPIED | CLEANING | OUT_OF_SERVICE
    patient_id: str | None = None


@dataclass
class RoomTwinState:
    id: str
    floor_id: str
    code: str
    room_type: str
    capacity: int
    position: tuple[float, float, float]
    local_saturation: float = 0.0
    status_color: str = "GREEN"
    beds: dict[str, BedTwinState] = field(default_factory=dict)

    def recompute_status(self) -> None:
        if not self.beds:
            self.local_saturation = 0.0
        else:
            occupied = sum(1 for b in self.beds.values() if b.status == "OCCUPIED")
            self.local_saturation = occupied / len(self.beds)
        self.status_color = saturation_to_color(self.local_saturation)


@dataclass
class FloorTwinState:
    id: str
    name: str
    floor_type: str
    rooms: dict[str, RoomTwinState] = field(default_factory=dict)


@dataclass
class HospitalTwinState:
    id: str
    name: str
    global_saturation_index: float = 0.0
    global_status_color: str = "GREEN"
    floors: dict[str, FloorTwinState] = field(default_factory=dict)


class DigitalTwinStateManager:
    """In-memory canonical twin state; Postgres remains the persisted source of
    truth (loaded on boot / hospital CRUD), this manager mirrors it for
    low-latency real-time mutation during a running simulation."""

    def __init__(self) -> None:
        self._hospitals: dict[str, HospitalTwinState] = {}
        self._pending_deltas: list[dict[str, Any]] = []

    def register_hospital(self, twin: HospitalTwinState) -> None:
        self._hospitals[twin.id] = twin

    def get(self, hospital_id: str) -> HospitalTwinState | None:
        return self._hospitals.get(hospital_id)

    def apply_bed_status(self, hospital_id: str, floor_id: str, room_id: str, bed: BedTwinState) -> dict:
        hospital = self._hospitals[hospital_id]
        room = hospital.floors[floor_id].rooms[room_id]
        room.beds[bed.id] = bed
        room.recompute_status()
        delta = {
            "type": "bed_update",
            "hospital_id": hospital_id,
            "floor_id": floor_id,
            "room_id": room_id,
            "bed": bed.__dict__,
            "room_status_color": room.status_color,
            "room_local_saturation": room.local_saturation,
        }
        return delta

    def apply_global_saturation(self, hospital_id: str, saturation_index: float) -> dict:
        hospital = self._hospitals[hospital_id]
        hospital.global_saturation_index = saturation_index
        hospital.global_status_color = saturation_to_color(saturation_index)
        return {
            "type": "global_saturation_update",
            "hospital_id": hospital_id,
            "saturation_index": saturation_index,
            "status_color": hospital.global_status_color,
        }


# Process-wide singleton (prototype scope). For multi-worker deployment,
# replace with a Redis-backed shared state per docs/ARCHITECTURE.md roadmap phase 3.
twin_state_manager = DigitalTwinStateManager()
