import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class DatasetSource(str, enum.Enum):
    MIMIC_IV = "MIMIC_IV"
    PHYSIONET = "PHYSIONET"
    KAGGLE = "KAGGLE"
    GOV_OPEN = "GOV_OPEN"
    SYNTHETIC = "SYNTHETIC"


class Dataset(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(255))
    source: Mapped[DatasetSource] = mapped_column(Enum(DatasetSource))
    version: Mapped[str] = mapped_column(String(50), default="1.0")
    storage_uri: Mapped[str] = mapped_column(String(500), default="")
    schema_hash: Mapped[str] = mapped_column(String(64), default="")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
