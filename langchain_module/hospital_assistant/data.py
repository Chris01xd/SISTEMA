"""Loading of the canonical hospital-admissions table.

The module only depends on the *file format* produced by the backend's ETL
(Parquet with the canonical columns below), not on any backend code."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = [
    "patient_id", "age", "sex", "admission_time", "discharge_time",
    "diagnosis", "severity_level", "los_hours", "icu_flag", "outcome",
]


class DatasetError(RuntimeError):
    pass


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}. Import it first (backend: POST /api/v1/datasets/ingest "
            "with key 'mimic-iv-demo', or the 'Public data' page) or set HOSPITAL_DATASET_PATH."
        )
    df = pd.read_parquet(path)
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"{path} is not a canonical admissions table; missing columns: {missing}")
    return df
