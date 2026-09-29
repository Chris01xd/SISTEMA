"""One-off seed script: populates floors/rooms/beds for an existing hospital_id
so the 3D digital twin snapshot has something to render. There is no CRUD API
for floors/rooms/beds yet (tracked as a gap — see README "Known Gaps"), so
this writes directly via SQLAlchemy.

Usage:
    python tests/seed_demo_hospital.py <hospital_id>
"""
import sys

from app.database.session import SessionLocal
from app.models.hospital import Bed, BedStatus, BedType, Floor, FloorType, Room, RoomType, ZoneStatus


def seed(hospital_id: str) -> None:
    db = SessionLocal()
    try:
        floor_er = Floor(hospital_id=hospital_id, level=0, name="Urgencias", floor_type=FloorType.ER)
        floor_icu = Floor(hospital_id=hospital_id, level=1, name="UCI", floor_type=FloorType.ICU)
        floor_or = Floor(hospital_id=hospital_id, level=2, name="Quirófanos", floor_type=FloorType.SURGERY)
        floor_ward = Floor(hospital_id=hospital_id, level=3, name="Hospitalización", floor_type=FloorType.WARD)
        db.add_all([floor_er, floor_icu, floor_or, floor_ward])
        db.flush()

        def make_room(floor, code, room_type, capacity, x, z, bed_type, n_beds, bed_status=BedStatus.AVAILABLE):
            room = Room(
                floor_id=floor.id, code=code, room_type=room_type, capacity=capacity,
                status=ZoneStatus.GREEN, position_x=x, position_y=0, position_z=z,
            )
            db.add(room)
            db.flush()
            for i in range(n_beds):
                db.add(Bed(room_id=room.id, code=f"{code}-B{i+1}", bed_type=bed_type, status=bed_status))
            return room

        make_room(floor_er, "ER-1", RoomType.TRIAGE, 15, -6, -6, BedType.ER, 15)
        make_room(floor_er, "ER-2", RoomType.RESUS, 15, 6, -6, BedType.ER, 15, bed_status=BedStatus.OCCUPIED)

        make_room(floor_icu, "ICU-1", RoomType.ICU, 10, 0, 0, BedType.ICU, 10)

        make_room(floor_or, "OR-1", RoomType.OR, 4, 0, 6, BedType.OR_TABLE, 4)

        make_room(floor_ward, "WARD-1", RoomType.WARD, 25, -6, 12, BedType.WARD, 25)
        make_room(floor_ward, "WARD-2", RoomType.WARD, 25, 6, 12, BedType.WARD, 25)

        db.commit()
        print(f"Seeded floors/rooms/beds for hospital_id={hospital_id}")
    finally:
        db.close()


if __name__ == "__main__":
    seed(sys.argv[1])
