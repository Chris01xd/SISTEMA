import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.authentication.dependencies import get_current_user, require_permission
from app.database.session import get_db
from app.schemas.datasets import DatasetOut, DatasetPreview, IngestRequest, PublicDatasetOut
from app.services import dataset_service

router = APIRouter()


@router.get("/catalog", response_model=list[PublicDatasetOut], dependencies=[Depends(get_current_user)])
def get_catalog(db: Session = Depends(get_db)) -> list[dict]:
    return dataset_service.list_catalog(db)


@router.post("/ingest", response_model=DatasetOut, dependencies=[Depends(require_permission("datasets:ingest"))])
def ingest_dataset(payload: IngestRequest, db: Session = Depends(get_db)) -> dict:
    """Downloads (checksum-verified) and ingests a catalog dataset. Synchronous: the
    current catalog is only a few MB, but the first run is bound by PhysioNet's response
    time (measured 20-40 s); later runs reuse the verified local files. Move this to a
    background task if larger datasets are added to the catalog."""
    return dataset_service.to_out(dataset_service.ingest_public_dataset(db, payload.key))


@router.get("", response_model=list[DatasetOut], dependencies=[Depends(get_current_user)])
def list_datasets(db: Session = Depends(get_db)) -> list[dict]:
    return [dataset_service.to_out(d) for d in dataset_service.list_datasets(db)]


@router.get("/{dataset_id}/summary", dependencies=[Depends(get_current_user)])
def get_summary(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    return dataset_service.summary(dataset_service.get_dataset(db, dataset_id))


@router.get("/{dataset_id}/preview", response_model=DatasetPreview, dependencies=[Depends(get_current_user)])
def get_preview(dataset_id: uuid.UUID, limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)) -> dict:
    return dataset_service.preview(dataset_service.get_dataset(db, dataset_id), limit)
