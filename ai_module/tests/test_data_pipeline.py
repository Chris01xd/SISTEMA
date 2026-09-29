"""Ingestion, validation, cleaning and transformation of the public dataset."""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pandas as pd
import pytest

from ai_module.digital_twin import ingest
from ai_module.digital_twin.errors import DatasetDownloadError, DataValidationError
from ai_module.digital_twin.preprocess import NOT_ED, preprocess
from ai_module.digital_twin.sources import DatasetSpec, PackageSpec
from ai_module.digital_twin.validation import PROCESSED_COLUMNS, validate_processed


def test_processed_table_has_one_row_per_admission_and_valid_ranges(raw_tables):
    df, report = preprocess(raw_tables)
    assert list(df.columns) == list(PROCESSED_COLUMNS)
    assert len(df) == len(raw_tables.admissions) == report.processed_rows
    assert df["hadm_id"].is_unique
    assert (df["los_hours"] > 0).all()
    assert (df["icu_hours"] <= df["los_hours"]).all()
    assert (df["icu_flag"] == (df["icu_hours"] > 0)).all()


def test_age_follows_the_mimic_anchor_method(raw_tables):
    admissions = raw_tables.admissions.copy()
    admissions.loc[0, "admittime"] = "2153-06-01 10:00:00"
    admissions.loc[0, "dischtime"] = "2153-06-03 10:00:00"
    patients = raw_tables.patients
    tables = dataclasses.replace(raw_tables, admissions=admissions)
    df, _ = preprocess(tables)
    subject = admissions.loc[0, "subject_id"]
    anchor = patients.set_index("subject_id").loc[subject]
    row = df[df["hadm_id"] == admissions.loc[0, "hadm_id"]].iloc[0]
    assert row["age"] == anchor["anchor_age"] + (2153 - anchor["anchor_year"])
    assert row["admit_hour"] == 10


def test_vitals_cleaning_converts_celsius_and_nulls_impossible_values(raw_tables):
    triage = raw_tables.triage.copy()
    triage.loc[0, "temperature"] = 37.0  # typed in Celsius
    triage.loc[1, "dbp"] = 879.0  # recording error seen in the real data
    df, report = preprocess(dataclasses.replace(raw_tables, triage=triage))
    by_stay = raw_tables.edstays.set_index("stay_id")["hadm_id"].astype(int)
    first = df.set_index("hadm_id").loc[by_stay[triage.loc[0, "stay_id"]]]
    second = df.set_index("hadm_id").loc[by_stay[triage.loc[1, "stay_id"]]]
    assert first["triage_temperature_f"] == pytest.approx(98.6)
    assert np.isnan(second["triage_dbp"])
    assert report.temperature_celsius_converted == 1
    assert report.vitals_nulled_out_of_range["triage_dbp"] == 1


def test_most_urgent_ed_visit_is_kept_and_non_ed_admissions_are_marked(raw_tables):
    ed = raw_tables.edstays.copy()
    triage = raw_tables.triage.copy()
    hadm = ed.loc[0, "hadm_id"]
    extra_stay = 99999
    triage.loc[0, "acuity"] = 3.0  # the original visit is less urgent than the extra one
    ed = pd.concat([ed, pd.DataFrame([{**ed.loc[0].to_dict(), "stay_id": extra_stay, "arrival_transport": "WALK IN"}])])
    triage = pd.concat([triage, pd.DataFrame([{**triage.loc[0].to_dict(), "stay_id": extra_stay, "acuity": 1.0}])])
    df, _ = preprocess(dataclasses.replace(raw_tables, edstays=ed, triage=triage))
    row = df[df["hadm_id"] == int(hadm)].iloc[0]
    assert row["ed_acuity_esi"] == 1 and row["arrival_transport"] == "WALK IN"
    non_ed = df[~df["arrived_via_ed"]]
    assert (non_ed["arrival_transport"] == NOT_ED).all() and non_ed["ed_acuity_esi"].isna().all()


def test_icu_time_longer_than_the_stay_is_clipped(raw_tables):
    icu = raw_tables.icustays.copy()
    icu.loc[0, "los"] = 999.0
    df, report = preprocess(dataclasses.replace(raw_tables, icustays=icu))
    row = df[df["hadm_id"] == icu.loc[0, "hadm_id"]].iloc[0]
    assert row["icu_hours"] == pytest.approx(row["los_hours"])
    assert report.icu_hours_clipped_to_los >= 1


def test_invalid_stays_are_dropped_and_counted(raw_tables):
    admissions = raw_tables.admissions.copy()
    admissions.loc[0, "dischtime"] = admissions.loc[0, "admittime"]
    df, report = preprocess(dataclasses.replace(raw_tables, admissions=admissions))
    assert report.dropped["invalid_or_non_positive_stay"] == 1
    assert len(df) == len(admissions) - 1


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: dataclasses.replace(t, admissions=t.admissions.drop(columns=["admittime"])), "missing columns"),
        (lambda t: dataclasses.replace(t, patients=pd.concat([t.patients, t.patients.head(1)])), "duplicated"),
        (lambda t: dataclasses.replace(t, patients=t.patients.iloc[1:]), "missing from patients"),
    ],
)
def test_raw_validation_rejects_broken_inputs(raw_tables, mutate, message):
    with pytest.raises(DataValidationError, match=message):
        preprocess(mutate(raw_tables))


def test_processed_validation_catches_inconsistencies(raw_tables):
    df, _ = preprocess(raw_tables)
    broken = df.copy()
    broken.loc[0, "icu_flag"] = not broken.loc[0, "icu_flag"]
    with pytest.raises(DataValidationError, match="icu_flag"):
        validate_processed(broken)


# ---------------------------------------------------------------- download


def _spec() -> DatasetSpec:
    return DatasetSpec(key="demo", name="d", version="1", license="l", homepage=(), citation=(),
                       packages=(PackageSpec(slug="pkg", version="1", files=("hosp/a.csv.gz",)),))


def test_download_verifies_checksums_and_skips_valid_files(tmp_path, monkeypatch):
    payload = b"content"
    digest = hashlib.sha256(payload).hexdigest()
    calls: list[str] = []

    def fake_get(url: str) -> bytes:
        calls.append(url)
        return f"{digest}  hosp/a.csv.gz\n".encode() if url.endswith("SHA256SUMS.txt") else payload

    monkeypatch.setattr(ingest, "_http_get", fake_get)
    dest = ingest.download_dataset(_spec(), tmp_path)
    assert (dest / "hosp" / "a.csv.gz").read_bytes() == payload
    calls.clear()
    ingest.download_dataset(_spec(), tmp_path)
    assert calls == []  # checksum cached + file verified offline


def test_download_rejects_tampered_files(tmp_path, monkeypatch):
    def fake_get(url: str) -> bytes:
        return f"{'0' * 64}  hosp/a.csv.gz\n".encode() if url.endswith("SHA256SUMS.txt") else b"tampered"

    monkeypatch.setattr(ingest, "_http_get", fake_get)
    with pytest.raises(DatasetDownloadError, match="checksum mismatch"):
        ingest.download_dataset(_spec(), tmp_path)
    assert not (tmp_path / "demo-1" / "hosp" / "a.csv.gz").exists()


def test_parse_checksums_handles_binary_markers():
    assert ingest.parse_checksums("ABC *x/y.csv\n\nbad-line\n") == {"x/y.csv": "abc"}
