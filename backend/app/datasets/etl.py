"""ETL pipeline for public healthcare datasets (MIMIC-IV, PhysioNet, Kaggle
healthcare datasets, government open data).

Design: each source gets a thin `Loader` that knows how to read the raw
export format; `clean()` and `normalize()` are source-agnostic and operate on
a canonical pandas schema so downstream AI training code never needs to know
which dataset a row came from.

NOTE: full MIMIC-IV requires credentialed access (PhysioNet data use agreement)
and is NOT bundled; point `raw_path` at your own credentialed export. The
open-access MIMIC-IV *demo* needs no credentials and is downloaded and checksum-
verified by `app/datasets/public_datasets.py` (see `app/services/dataset_service.py`
for the full download -> ETL -> Parquet -> `datasets` row flow).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = [
    "patient_id", "age", "sex", "admission_time", "discharge_time",
    "diagnosis", "severity_level", "los_hours", "icu_flag", "outcome",
]


@dataclass
class DatasetLoadResult:
    dataframe: pd.DataFrame
    schema_hash: str
    row_count: int


class BaseLoader:
    source_name: str = "BASE"

    def load_raw(self, raw_path: Path) -> pd.DataFrame:
        raise NotImplementedError

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["age", "sex"])
        df = df[(df["age"] >= 0) & (df["age"] <= 120)]
        return df.drop_duplicates()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sex"] = df["sex"].str.upper().str[0]
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[CANONICAL_COLUMNS]

    def run(self, raw_path: Path) -> DatasetLoadResult:
        df = self.load_raw(raw_path)
        df = self.clean(df)
        df = self.normalize(df)
        schema_hash = hashlib.sha256(",".join(df.columns).encode()).hexdigest()
        return DatasetLoadResult(dataframe=df, schema_hash=schema_hash, row_count=len(df))


# MIMIC-IV ED triage acuity is the Emergency Severity Index (1 = most urgent,
# 5 = least). The rest of the system uses `SeverityLevel` (1 = minor .. 4 =
# critical), so the scale is inverted here rather than at every consumer.
ESI_TO_SEVERITY = {1: 4, 2: 3, 3: 2, 4: 1, 5: 1}


def _find_table(raw_path: Path, module: str, name: str) -> Path | None:
    """Locate a MIMIC table in either the official layout (`<module>/<name>.csv[.gz]`)
    or a flat folder of CSVs."""
    for base in (raw_path / module, raw_path):
        for suffix in (".csv", ".csv.gz"):
            candidate = base / f"{name}{suffix}"
            if candidate.exists():
                return candidate
    return None


class MimicIVLoader(BaseLoader):
    """Reads a MIMIC-IV export (full credentialed release or the open demo) and
    builds one canonical row per hospital admission.

    Required: `admissions` and `patients` (hosp module). Optional tables enrich the
    result and are skipped when absent: `icustays` (true ICU flag), `diagnoses_icd` +
    `d_icd_diagnoses` (primary diagnosis), and the ED module's `edstays` + `triage`
    (severity from triage acuity, only for admissions that came through the ED)."""

    source_name = "MIMIC_IV"

    def _read(self, raw_path: Path, module: str, name: str, required: bool = False, **kwargs) -> pd.DataFrame | None:
        path = _find_table(raw_path, module, name)
        if path is None:
            if required:
                raise FileNotFoundError(f"MIMIC-IV table '{name}' not found under {raw_path}")
            return None
        return pd.read_csv(path, **kwargs)

    def load_raw(self, raw_path: Path) -> pd.DataFrame:
        admissions = self._read(raw_path, "hosp", "admissions", required=True)
        patients = self._read(raw_path, "hosp", "patients", required=True)
        df = admissions.merge(patients, on="subject_id", how="left")

        admit = pd.to_datetime(df["admittime"])
        discharge = pd.to_datetime(df["dischtime"])

        # `anchor_age` is the age in `anchor_year`, and MIMIC shifts each patient's dates
        # consistently around that anchor. Age at admission is therefore anchor_age plus
        # the years elapsed since the anchor year (the documented MIMIC-IV method).
        age = df["anchor_age"] + (admit.dt.year - df["anchor_year"])

        return pd.DataFrame(
            {
                "patient_id": df["subject_id"],
                "age": age,
                "sex": df["gender"],
                "admission_time": admit,
                "discharge_time": discharge,
                "diagnosis": self._primary_diagnosis(raw_path, df),
                "severity_level": self._ed_severity(raw_path, df),
                "los_hours": (discharge - admit).dt.total_seconds() / 3600,
                "icu_flag": self._icu_flag(raw_path, df),
                "outcome": df["hospital_expire_flag"].map({1: "DECEASED", 0: "DISCHARGED"}),
            }
        )

    def _icu_flag(self, raw_path: Path, df: pd.DataFrame) -> pd.Series:
        icustays = self._read(raw_path, "icu", "icustays")
        if icustays is None:
            return df["admission_location"].astype(str).str.contains("ICU", case=False)
        return df["hadm_id"].isin(icustays["hadm_id"])

    def _primary_diagnosis(self, raw_path: Path, df: pd.DataFrame) -> pd.Series:
        diagnoses = self._read(raw_path, "hosp", "diagnoses_icd", dtype={"icd_code": str})
        dictionary = self._read(raw_path, "hosp", "d_icd_diagnoses", dtype={"icd_code": str})
        if diagnoses is None or dictionary is None:
            return pd.Series([None] * len(df), index=df.index, dtype="object")
        primary = diagnoses[diagnoses["seq_num"] == 1].merge(dictionary, on=["icd_code", "icd_version"], how="left")
        titles = primary.drop_duplicates("hadm_id").set_index("hadm_id")["long_title"]
        return df["hadm_id"].map(titles)

    def _ed_severity(self, raw_path: Path, df: pd.DataFrame) -> pd.Series:
        stays = self._read(raw_path, "ed", "edstays")
        triage = self._read(raw_path, "ed", "triage")
        if stays is None or triage is None:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
        acuity = stays.dropna(subset=["hadm_id"]).merge(triage[["stay_id", "acuity"]], on="stay_id")
        acuity = acuity.dropna(subset=["acuity"])
        # A hospital admission can follow several ED visits; keep the most urgent (lowest ESI).
        most_urgent = acuity.assign(hadm_id=acuity["hadm_id"].astype("int64")).groupby("hadm_id")["acuity"].min()
        severity = df["hadm_id"].map(most_urgent).map(ESI_TO_SEVERITY)
        return severity.astype("Int64")


class KaggleHospitalLoader(BaseLoader):
    """Generic loader for Kaggle 'hospital emergency admissions' style CSVs."""

    source_name = "KAGGLE"

    def load_raw(self, raw_path: Path) -> pd.DataFrame:
        return pd.read_csv(raw_path)


class GovOpenDataLoader(BaseLoader):
    source_name = "GOV_OPEN"

    def load_raw(self, raw_path: Path) -> pd.DataFrame:
        return pd.read_csv(raw_path)


LOADERS: dict[str, type[BaseLoader]] = {
    "MIMIC_IV": MimicIVLoader,
    "KAGGLE": KaggleHospitalLoader,
    "GOV_OPEN": GovOpenDataLoader,
}


def ingest(source: str, raw_path: Path) -> DatasetLoadResult:
    loader_cls = LOADERS.get(source)
    if loader_cls is None:
        raise ValueError(f"No loader registered for source={source}")
    return loader_cls().run(raw_path)


def summarize(df: pd.DataFrame) -> dict:
    """Descriptive statistics of a canonical frame, JSON-serializable, computed from
    the data itself (nothing is hard-coded or estimated)."""
    n = len(df)
    if n == 0:
        return {"rows": 0}

    def pct(mask_or_series) -> float:
        return round(100 * float(mask_or_series.mean()), 1)

    los = df["los_hours"].dropna()
    severity = df["severity_level"].dropna().astype(int)
    by_hour = df["admission_time"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    diagnoses = df["diagnosis"].dropna().value_counts().head(10)

    return {
        "rows": n,
        "patients": int(df["patient_id"].nunique()),
        # MIMIC shifts dates per patient for de-identification: the span is internally
        # consistent (durations, hour of day) but is not the real calendar date.
        "period": {"start": str(df["admission_time"].min()), "end": str(df["admission_time"].max())},
        "age": {
            "mean": round(float(df["age"].mean()), 1),
            "median": float(df["age"].median()),
            "min": int(df["age"].min()),
            "max": int(df["age"].max()),
        },
        "sex": {str(k): int(v) for k, v in df["sex"].value_counts().items()},
        "los_hours": (
            {
                "mean": round(float(los.mean()), 1),
                "median": round(float(los.median()), 1),
                "p90": round(float(los.quantile(0.9)), 1),
            }
            if len(los)
            else None
        ),
        "icu_rate_pct": pct(df["icu_flag"].dropna().astype(bool)),
        "mortality_rate_pct": pct(df["outcome"].dropna() == "DECEASED"),
        # 1 = minor .. 4 = critical (SeverityLevel); only admissions that came through the ED have one.
        "severity_distribution": {str(k): int(v) for k, v in severity.value_counts().sort_index().items()},
        "severity_coverage_pct": round(100 * len(severity) / n, 1),
        "admissions_by_hour": [int(v) for v in by_hour.tolist()],
        "top_diagnoses": [{"diagnosis": str(k), "count": int(v)} for k, v in diagnoses.items()],
        "missing_pct": {col: round(100 * float(df[col].isna().mean()), 1) for col in CANONICAL_COLUMNS},
    }


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
