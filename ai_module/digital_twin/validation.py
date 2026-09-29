"""Step 3 of the pipeline: schema and integrity checks, before and after cleaning.

Violations raise ``DataValidationError`` with every problem listed, so a broken or
changed upstream file stops the pipeline instead of silently producing a wrong twin."""
from __future__ import annotations

import pandas as pd

from ai_module.digital_twin.errors import DataValidationError
from ai_module.digital_twin.ingest import RawTables

REQUIRED_RAW_COLUMNS: dict[str, tuple[str, ...]] = {
    "admissions": ("subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "admission_location", "hospital_expire_flag"),
    "patients": ("subject_id", "gender", "anchor_age", "anchor_year"),
    "icustays": ("subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"),
    "edstays": ("subject_id", "hadm_id", "stay_id", "intime", "arrival_transport"),
    "triage": ("stay_id", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity"),
}

_UNIQUE_KEYS = {"admissions": "hadm_id", "patients": "subject_id", "icustays": "stay_id", "edstays": "stay_id", "triage": "stay_id"}

PROCESSED_COLUMNS: tuple[str, ...] = (
    "subject_id", "hadm_id", "age", "sex", "admission_type", "admission_location", "admit_hour",
    "arrived_via_ed", "arrival_transport", "ed_acuity_esi",
    "triage_temperature_f", "triage_heartrate", "triage_resprate", "triage_o2sat", "triage_sbp", "triage_dbp",
    "los_hours", "icu_flag", "icu_hours", "died_in_hospital",
)


def validate_raw(tables: RawTables) -> None:
    """Checks required columns, non-emptiness and primary-key uniqueness of each raw table."""
    problems: list[str] = []
    for name, columns in REQUIRED_RAW_COLUMNS.items():
        df: pd.DataFrame = getattr(tables, name)
        if df.empty and name in ("admissions", "patients"):
            problems.append(f"{name}: table is empty")
        missing = [c for c in columns if c not in df.columns]
        if missing:
            problems.append(f"{name}: missing columns {missing}")
            continue
        key = _UNIQUE_KEYS[name]
        duplicated = int(df[key].duplicated().sum())
        if duplicated:
            problems.append(f"{name}: {duplicated} duplicated values of key '{key}'")
    if not problems:
        orphans = int((~tables.admissions["subject_id"].isin(tables.patients["subject_id"])).sum())
        if orphans:
            problems.append(f"admissions: {orphans} rows reference a subject_id missing from patients")
    if problems:
        raise DataValidationError("Raw data validation failed: " + "; ".join(problems))


def validate_processed(df: pd.DataFrame) -> None:
    """Checks the cleaned, one-row-per-admission table the twin and the model consume."""
    problems: list[str] = []
    missing = [c for c in PROCESSED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Processed data is missing columns {missing}")
    if df.empty:
        problems.append("no admissions left after cleaning")
    if df["hadm_id"].duplicated().any():
        problems.append("duplicated hadm_id")
    if (df["los_hours"] <= 0).any():
        problems.append("non-positive length of stay")
    if ((df["age"] < 0) | (df["age"] > 120)).any():
        problems.append("age outside 0-120")
    if not df["admit_hour"].between(0, 23).all():
        problems.append("admit_hour outside 0-23")
    if (df["icu_hours"] > df["los_hours"] + 1e-6).any():
        problems.append("ICU hours exceed hospital length of stay")
    acuity = df["ed_acuity_esi"].dropna()
    if not acuity.between(1, 5).all():
        problems.append("ED acuity outside ESI 1-5")
    if (df["icu_flag"] != (df["icu_hours"] > 0)).any():
        problems.append("icu_flag inconsistent with icu_hours")
    if problems:
        raise DataValidationError("Processed data validation failed: " + "; ".join(problems))
