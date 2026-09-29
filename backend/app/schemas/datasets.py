import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.dataset import DatasetSource


class PublicDatasetOut(BaseModel):
    """A dataset available in the catalog, and whether it is already ingested."""

    key: str
    name: str
    version: str
    description: str
    license: str
    homepage: str
    citation: str
    ingested: bool
    dataset_id: uuid.UUID | None = None


class DatasetOut(BaseModel):
    id: uuid.UUID
    name: str
    source: DatasetSource
    version: str
    schema_hash: str
    ingested_at: datetime
    row_count: int | None = None


class IngestRequest(BaseModel):
    key: str


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[dict]
    total_rows: int
