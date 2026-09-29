import enum

from sqlalchemy import Enum, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class FloorType(str, enum.Enum):
    ER = "ER"
    ICU = "ICU"
    SURGERY = "SURGERY"
    WARD = "WARD"
    IMAGING = "IMAGING"


class RoomType(str, enum.Enum):
    TRIAGE = "TRIAGE"
    RESUS = "RESUS"
    OR = "OR"
    ICU = "ICU"
    WARD = "WARD"
    IMAGING = "IMAGING"


class ZoneStatus(str, enum.Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class BedType(str, enum.Enum):
    ER = "ER"
    ICU = "ICU"
    WARD = "WARD"
    OR_TABLE = "OR_TABLE"


class BedStatus(str, enum.Enum):
    AVAILABLE = "AVAILABLE"
    OCCUPIED = "OCCUPIED"
    CLEANING = "CLEANING"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class Hospital(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "hospitals"

    name: Mapped[str] = mapped_column(String(255))
    location: Mapped[str] = mapped_column(String(255), default="")
    total_capacity: Mapped[int] = mapped_column(Integer, default=0)

    floors: Mapped[list["Floor"]] = relationship(back_populates="hospital", cascade="all, delete-orphan")


class Floor(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "floors"

    hospital_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("hospitals.id"))
    level: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(100))
    floor_type: Mapped[FloorType] = mapped_column(Enum(FloorType))

    hospital: Mapped["Hospital"] = relationship(back_populates="floors")
    rooms: Mapped[list["Room"]] = relationship(back_populates="floor", cascade="all, delete-orphan")


class Room(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "rooms"

    floor_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("floors.id"))
    code: Mapped[str] = mapped_column(String(50))
    room_type: Mapped[RoomType] = mapped_column(Enum(RoomType))
    capacity: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[ZoneStatus] = mapped_column(Enum(ZoneStatus), default=ZoneStatus.GREEN)

    # 3D placement within the twin scene
    position_x: Mapped[float] = mapped_column(Float, default=0.0)
    position_y: Mapped[float] = mapped_column(Float, default=0.0)
    position_z: Mapped[float] = mapped_column(Float, default=0.0)

    floor: Mapped["Floor"] = relationship(back_populates="rooms")
    beds: Mapped[list["Bed"]] = relationship(back_populates="room", cascade="all, delete-orphan")


class Bed(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "beds"

    room_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("rooms.id"))
    code: Mapped[str] = mapped_column(String(50))
    bed_type: Mapped[BedType] = mapped_column(Enum(BedType))
    status: Mapped[BedStatus] = mapped_column(Enum(BedStatus), default=BedStatus.AVAILABLE)
    patient_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True
    )

    room: Mapped["Room"] = relationship(back_populates="beds")
