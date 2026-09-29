"""Steps 4-5 of the pipeline: cleaning and transformation into one row per admission.

Every rule is explicit and counted in a ``DataReport`` so the processed table can be
audited against the raw files. Only information available *at admission time* becomes
a model feature; outcomes (length of stay, ICU use, death) are kept as targets / for the
observed profile only. See ``ai_module/data/DATASET.md`` for the rationale."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai_module.digital_twin.ingest import RawTables
from ai_module.digital_twin.validation import PROCESSED_COLUMNS, validate_processed, validate_raw

# Plausible physiological ranges; values outside are recording errors and become missing.
VITAL_RANGES: dict[str, tuple[float, float]] = {
    "triage_temperature_f": (90.0, 110.0),
    "triage_heartrate": (20.0, 250.0),
    "triage_resprate": (4.0, 60.0),
    "triage_o2sat": (50.0, 100.0),
    "triage_sbp": (50.0, 300.0),
    "triage_dbp": (20.0, 200.0),
}
# Temperatures in this band were clearly typed in Celsius into a Fahrenheit field.
_CELSIUS_BAND = (30.0, 45.0)

NOT_ED = "NOT_ED"


@dataclass
class DataReport:
    """Audit trail of the cleaning step (persisted as ``data_report.json``)."""

    raw_rows: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    vitals_nulled_out_of_range: dict[str, int] = field(default_factory=dict)
    temperature_celsius_converted: int = 0
    icu_hours_clipped_to_los: int = 0
    processed_rows: int = 0
    processed_patients: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hours(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 3600.0


def _ed_features(tables: RawTables) -> pd.DataFrame:
    """One row per admission that came through the ED: the most urgent visit (lowest ESI;
    latest visit on ties) with its triage vitals, as recorded at triage time."""
    stays = tables.edstays.dropna(subset=["hadm_id"]).copy()
    stays["hadm_id"] = stays["hadm_id"].astype("int64")
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce", format="ISO8601")
    merged = stays.merge(tables.triage, on="stay_id", how="left", suffixes=("", "_triage"))
    merged["_acuity_sort"] = merged["acuity"].fillna(99)
    merged = merged.sort_values(["hadm_id", "_acuity_sort", "intime"], ascending=[True, True, False])
    first = merged.drop_duplicates("hadm_id", keep="first")
    return pd.DataFrame(
        {
            "hadm_id": first["hadm_id"].to_numpy(),
            "arrival_transport": first["arrival_transport"].fillna("UNKNOWN").astype(str).str.upper().to_numpy(),
            "ed_acuity_esi": first["acuity"].to_numpy(dtype="float64"),
            "triage_temperature_f": first["temperature"].to_numpy(dtype="float64"),
            "triage_heartrate": first["heartrate"].to_numpy(dtype="float64"),
            "triage_resprate": first["resprate"].to_numpy(dtype="float64"),
            "triage_o2sat": first["o2sat"].to_numpy(dtype="float64"),
            "triage_sbp": first["sbp"].to_numpy(dtype="float64"),
            "triage_dbp": first["dbp"].to_numpy(dtype="float64"),
        }
    )


def _clean_vitals(df: pd.DataFrame, report: DataReport) -> pd.DataFrame:
    df = df.copy()
    temp = df["triage_temperature_f"]
    celsius = temp.between(*_CELSIUS_BAND)
    report.temperature_celsius_converted = int(celsius.sum())
    df.loc[celsius, "triage_temperature_f"] = temp[celsius] * 9 / 5 + 32
    for column, (low, high) in VITAL_RANGES.items():
        out = df[column].notna() & ~df[column].between(low, high)
        report.vitals_nulled_out_of_range[column] = int(out.sum())
        df.loc[out, column] = np.nan
    return df


def preprocess(tables: RawTables) -> tuple[pd.DataFrame, DataReport]:
    """Validates, cleans and transforms the raw tables. Returns the processed admissions
    table (columns ``PROCESSED_COLUMNS``) and the audit report."""
    validate_raw(tables)
    report = DataReport(raw_rows={n: int(len(getattr(tables, n))) for n in ("admissions", "patients", "icustays", "edstays", "triage")})

    adm = tables.admissions.merge(tables.patients[["subject_id", "gender", "anchor_age", "anchor_year"]], on="subject_id", how="inner")
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce", format="ISO8601")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce", format="ISO8601")

    bad_times = adm["admittime"].isna() | adm["dischtime"].isna() | (adm["dischtime"] <= adm["admittime"])
    report.dropped["invalid_or_non_positive_stay"] = int(bad_times.sum())
    adm = adm[~bad_times]

    # MIMIC-IV method: age at admission = anchor_age + years elapsed since anchor_year
    # (dates are shifted per patient, consistently within each patient).
    adm = adm.assign(age=adm["anchor_age"] + (adm["admittime"].dt.year - adm["anchor_year"]))
    bad_age = ~adm["age"].between(0, 120)
    report.dropped["age_out_of_range"] = int(bad_age.sum())
    adm = adm[~bad_age]

    icu = tables.icustays.groupby("hadm_id")["los"].sum().mul(24.0).rename("icu_hours_raw")
    ed = _ed_features(tables)

    df = pd.DataFrame(
        {
            "subject_id": adm["subject_id"].astype("int64").to_numpy(),
            "hadm_id": adm["hadm_id"].astype("int64").to_numpy(),
            "age": adm["age"].astype("float64").to_numpy(),
            "sex": adm["gender"].astype(str).str.upper().str[0].to_numpy(),
            "admission_type": adm["admission_type"].fillna("UNKNOWN").astype(str).to_numpy(),
            "admission_location": adm["admission_location"].fillna("UNKNOWN").astype(str).to_numpy(),
            "admit_hour": adm["admittime"].dt.hour.astype("int64").to_numpy(),
            "los_hours": _hours(adm["dischtime"] - adm["admittime"]).to_numpy(dtype="float64"),
            "died_in_hospital": (adm["hospital_expire_flag"] == 1).to_numpy(),
        }
    )
    df = df.merge(ed, on="hadm_id", how="left")
    df["arrived_via_ed"] = df["arrival_transport"].notna()
    df["arrival_transport"] = df["arrival_transport"].fillna(NOT_ED)

    df["icu_hours"] = df["hadm_id"].map(icu).fillna(0.0).astype("float64")
    over = df["icu_hours"] > df["los_hours"]
    report.icu_hours_clipped_to_los = int(over.sum())
    df["icu_hours"] = np.minimum(df["icu_hours"], df["los_hours"])
    df["icu_flag"] = df["icu_hours"] > 0

    df = _clean_vitals(df, report)

    before = len(df)
    df = df.drop_duplicates("hadm_id")
    report.dropped["duplicated_admission"] = before - len(df)

    df = df[list(PROCESSED_COLUMNS)].sort_values("hadm_id").reset_index(drop=True)
    validate_processed(df)
    report.processed_rows = int(len(df))
    report.processed_patients = int(df["subject_id"].nunique())
    return df, report
