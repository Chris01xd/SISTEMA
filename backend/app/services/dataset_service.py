"""Ingestion of public datasets: download -> ETL -> canonical Parquet -> `datasets` row."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.datasets import etl
from app.datasets.public_datasets import CATALOG, DatasetDownloadError, PublicDataset, download, get_public_dataset
from app.models.dataset import Dataset


def _catalog_entry(dataset: Dataset) -> PublicDataset | None:
    return next((p for p in CATALOG.values() if p.name == dataset.name and p.version == dataset.version), None)


def _row_count(dataset: Dataset) -> int | None:
    path = Path(dataset.storage_uri)
    return pq.read_metadata(path).num_rows if path.exists() else None


def to_out(dataset: Dataset) -> dict:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "source": dataset.source,
        "version": dataset.version,
        "schema_hash": dataset.schema_hash,
        "ingested_at": dataset.ingested_at,
        "row_count": _row_count(dataset),
    }


def list_catalog(db: Session) -> list[dict]:
    ingested = {(d.name, d.version): d for d in db.query(Dataset).all()}
    out = []
    for entry in CATALOG.values():
        row = ingested.get((entry.name, entry.version))
        out.append(
            {
                "key": entry.key,
                "name": entry.name,
                "version": entry.version,
                "description": entry.description,
                "license": entry.license,
                "homepage": entry.homepage,
                "citation": entry.citation,
                "ingested": row is not None,
                "dataset_id": row.id if row else None,
            }
        )
    return out


def list_datasets(db: Session) -> list[Dataset]:
    return db.query(Dataset).order_by(Dataset.ingested_at.desc()).all()


def get_dataset(db: Session, dataset_id: uuid.UUID) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


def ingest_public_dataset(db: Session, key: str) -> Dataset:
    """Idempotent: re-ingesting the same dataset+version refreshes its row and Parquet
    instead of creating a duplicate."""
    try:
        entry = get_public_dataset(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0])

    try:
        raw_dir = download(entry, Path(settings.PUBLIC_DATASET_RAW_DIR))
    except DatasetDownloadError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    result = etl.ingest(entry.source.value, raw_dir)

    out_dir = Path(settings.PUBLIC_DATASET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_path = out_dir / f"{entry.local_dirname}.parquet"
    # pyarrow directly, same as preparation_service: avoids pandas' parquet engine detection.
    pq.write_table(pa.Table.from_pandas(result.dataframe, preserve_index=False), storage_path)

    dataset = db.query(Dataset).filter(Dataset.name == entry.name, Dataset.version == entry.version).first()
    if dataset is None:
        dataset = Dataset(name=entry.name, source=entry.source, version=entry.version)
        db.add(dataset)
    dataset.storage_uri = str(storage_path)
    dataset.schema_hash = result.schema_hash
    dataset.ingested_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(dataset)
    return dataset


def load_frame(dataset: Dataset) -> pd.DataFrame:
    path = Path(dataset.storage_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk; ingest it again")
    return pq.read_table(path).to_pandas()


def summary(dataset: Dataset) -> dict:
    return etl.summarize(load_frame(dataset))


def preview(dataset: Dataset, limit: int) -> dict:
    df = load_frame(dataset)
    head = df.head(limit)
    # to_json handles timestamps / nullable ints / NA uniformly; round-trip to plain Python.
    return {"columns": list(df.columns), "rows": json.loads(head.to_json(orient="records", date_format="iso")), "total_rows": len(df)}
