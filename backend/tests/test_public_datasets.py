"""Public-dataset integration: ETL semantics, download integrity, and the REST surface.

Everything runs offline against a tiny synthetic export that mimics the official
MIMIC-IV folder layout, so no network or PostgreSQL is needed."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.datasets import etl, public_datasets
from app.datasets.public_datasets import DatasetDownloadError, PackageSpec, PublicDataset
from app.models.dataset import Dataset, DatasetSource


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture
def mimic_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "mimic"
    _write(raw / "hosp" / "admissions.csv.gz", pd.DataFrame({
        "subject_id": [1, 1, 2, 3],
        "hadm_id": [11, 12, 13, 14],
        "admittime": ["2150-03-01 10:00:00", "2153-06-01 22:30:00", "2151-01-01 08:00:00", "2152-01-01 08:00:00"],
        "dischtime": ["2150-03-03 10:00:00", "2153-06-02 22:30:00", "2151-01-06 08:00:00", "2152-01-02 08:00:00"],
        "admission_location": ["EMERGENCY ROOM", "EMERGENCY ROOM", "EMERGENCY ROOM", "PHYSICIAN REFERRAL"],
        "hospital_expire_flag": [0, 0, 1, 0],
    }))
    _write(raw / "hosp" / "patients.csv.gz", pd.DataFrame({
        "subject_id": [1, 2, 3], "gender": ["F", "male", "F"],
        "anchor_age": [60, 40, 30], "anchor_year": [2150, 2150, 2150],
    }))
    _write(raw / "hosp" / "diagnoses_icd.csv.gz", pd.DataFrame({
        "subject_id": [1, 1, 1, 2], "hadm_id": [11, 11, 12, 13],
        "seq_num": [2, 1, 1, 1], "icd_code": ["V707", "0389", "4139", "I10"], "icd_version": [9, 9, 9, 10],
    }))
    _write(raw / "hosp" / "d_icd_diagnoses.csv.gz", pd.DataFrame({
        "icd_code": ["V707", "0389", "4139", "I10"], "icd_version": [9, 9, 9, 10],
        "long_title": ["Health exam", "Septicemia", "Angina", "Hypertension"],
    }))
    _write(raw / "icu" / "icustays.csv.gz", pd.DataFrame({"subject_id": [1], "hadm_id": [11], "stay_id": [900]}))
    _write(raw / "ed" / "edstays.csv.gz", pd.DataFrame({
        "subject_id": [1, 1, 1, 2], "hadm_id": [11, 11, 12, None], "stay_id": [500, 501, 502, 503],
    }))
    _write(raw / "ed" / "triage.csv.gz", pd.DataFrame({
        "subject_id": [1, 1, 1, 2], "stay_id": [500, 501, 502, 503], "acuity": [3, 1, 5, 2],
    }))
    return raw


def test_mimic_loader_enrichment_and_conventions(mimic_dir: Path):
    df = etl.ingest("MIMIC_IV", mimic_dir).dataframe.set_index("admission_time").sort_index()
    assert list(df.columns) == [c for c in etl.CANONICAL_COLUMNS if c != "admission_time"]

    # Age at admission = anchor_age + years since anchor_year (not the raw anchor_age).
    assert df["age"].tolist() == [60, 41, 32, 63]
    # Sex normalized to a single upper-case letter.
    assert set(df["sex"]) == {"F", "M"}
    # ICU flag comes from icustays (admission_location is not consulted when the table exists).
    assert df["icu_flag"].tolist() == [True, False, False, False]
    # Primary diagnosis = seq_num 1, resolved through the ICD dictionary (V-codes keep their letter).
    assert df.loc["2150-03-01 10:00:00", "diagnosis"] == "Septicemia"  # seq 1, not the V707 seq 2 row
    assert df.loc["2153-06-01 22:30:00", "diagnosis"] == "Angina"
    assert df.loc["2151-01-01 08:00:00", "diagnosis"] == "Hypertension"  # ICD-10 code resolved by (code, version)
    assert pd.isna(df.loc["2152-01-01 08:00:00", "diagnosis"])  # no diagnoses recorded
    assert df["outcome"].tolist() == ["DISCHARGED", "DECEASED", "DISCHARGED", "DISCHARGED"]
    assert df["los_hours"].iloc[0] == 48.0


def test_ed_acuity_is_inverted_to_system_severity_scale(mimic_dir: Path):
    df = etl.ingest("MIMIC_IV", mimic_dir).dataframe.set_index("admission_time").sort_index()
    # hadm 11 has ED stays with ESI 3 and 1 -> the most urgent (ESI 1) wins -> CRITICAL (4).
    assert df.loc["2150-03-01 10:00:00", "severity_level"] == 4
    # hadm 12: ESI 5 -> MINOR (1).
    assert df.loc["2153-06-01 22:30:00", "severity_level"] == 1
    # hadm 13's ED stay was not admitted (no hadm_id) and hadm 14 never visited the ED -> unknown.
    assert pd.isna(df.loc["2151-01-01 08:00:00", "severity_level"])
    assert pd.isna(df.loc["2152-01-01 08:00:00", "severity_level"])


def test_mimic_loader_works_with_only_required_tables_in_flat_layout(mimic_dir: Path, tmp_path: Path):
    flat = tmp_path / "flat"
    flat.mkdir()
    for name in ("admissions", "patients"):
        pd.read_csv(mimic_dir / "hosp" / f"{name}.csv.gz").to_csv(flat / f"{name}.csv", index=False)
    df = etl.ingest("MIMIC_IV", flat).dataframe
    assert len(df) == 4
    assert df["severity_level"].isna().all() and df["diagnosis"].isna().all()
    # Without icustays it falls back to admission_location.
    assert df["icu_flag"].sum() == 0


def test_mimic_loader_reports_missing_required_table(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="admissions"):
        etl.ingest("MIMIC_IV", tmp_path)


def test_summarize_is_computed_from_the_data(mimic_dir: Path):
    stats = etl.summarize(etl.ingest("MIMIC_IV", mimic_dir).dataframe)
    assert stats["rows"] == 4 and stats["patients"] == 3
    assert stats["mortality_rate_pct"] == 25.0
    assert stats["icu_rate_pct"] == 25.0
    assert stats["severity_distribution"] == {"1": 1, "4": 1}
    assert stats["severity_coverage_pct"] == 50.0
    assert len(stats["admissions_by_hour"]) == 24 and sum(stats["admissions_by_hour"]) == 4
    assert etl.summarize(pd.DataFrame(columns=etl.CANONICAL_COLUMNS)) == {"rows": 0}


# ------------------------------------------------------------------ download --
def _fake_source(monkeypatch, files: dict[str, bytes], sums_override: dict[str, str] | None = None):
    sums = {p: hashlib.sha256(b).hexdigest() for p, b in files.items()}
    sums.update(sums_override or {})
    sums_text = "\n".join(f"{h}  {p}" for p, h in sums.items())

    def fake_get(url: str) -> bytes:
        if url.endswith("SHA256SUMS.txt"):
            return sums_text.encode()
        return files[url.split("/2.2/", 1)[1]]

    monkeypatch.setattr(public_datasets, "_http_get", fake_get)


def _spec() -> PublicDataset:
    return PublicDataset(
        key="t", name="T", source=DatasetSource.SYNTHETIC, version="2.2", description="", license="", homepage="",
        citation="", packages=(PackageSpec(slug="t", version="2.2", files=("hosp/a.csv.gz", "ed/b.csv.gz")),),
    )


def test_download_verifies_and_keeps_official_layout(monkeypatch, tmp_path: Path):
    _fake_source(monkeypatch, {"hosp/a.csv.gz": b"aaa", "ed/b.csv.gz": b"bbb"})
    dest = public_datasets.download(_spec(), tmp_path)
    assert (dest / "hosp" / "a.csv.gz").read_bytes() == b"aaa"
    assert (dest / "ed" / "b.csv.gz").read_bytes() == b"bbb"

    # Second run is served from disk: no network call at all.
    monkeypatch.setattr(public_datasets, "_http_get", lambda url: pytest.fail("re-downloaded a verified file"))
    public_datasets.download(_spec(), tmp_path)


def test_download_rejects_tampered_file(monkeypatch, tmp_path: Path):
    _fake_source(monkeypatch, {"hosp/a.csv.gz": b"aaa", "ed/b.csv.gz": b"bbb"},
                 sums_override={"ed/b.csv.gz": "0" * 64})
    with pytest.raises(DatasetDownloadError, match="checksum mismatch"):
        public_datasets.download(_spec(), tmp_path)
    assert not (tmp_path / "t-2.2" / "ed" / "b.csv.gz").exists()


def test_download_redownloads_a_corrupted_local_file(monkeypatch, tmp_path: Path):
    _fake_source(monkeypatch, {"hosp/a.csv.gz": b"aaa", "ed/b.csv.gz": b"bbb"})
    dest = public_datasets.download(_spec(), tmp_path)
    (dest / "hosp" / "a.csv.gz").write_bytes(b"corrupt")
    public_datasets.download(_spec(), tmp_path)
    assert (dest / "hosp" / "a.csv.gz").read_bytes() == b"aaa"


def test_unknown_catalog_key():
    with pytest.raises(KeyError):
        public_datasets.get_public_dataset("nope")


# ----------------------------------------------------------------------- API --
@pytest.fixture
def client(monkeypatch, tmp_path: Path, mimic_dir: Path):
    from app.api.v1 import datasets as datasets_api  # noqa: F401  (ensures router import works)
    from app.authentication.dependencies import get_current_user
    from app.core.config import settings
    from app.database.session import get_db
    from app.main import app
    from app.services import dataset_service

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Dataset.__table__.create(engine)
    Session = sessionmaker(bind=engine)

    def override_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    admin = SimpleNamespace(role=SimpleNamespace(permissions={"*": True}), is_active=True)
    monkeypatch.setitem(app.dependency_overrides, get_db, override_db)
    monkeypatch.setitem(app.dependency_overrides, get_current_user, lambda: admin)
    monkeypatch.setattr(settings, "PUBLIC_DATASET_DIR", str(tmp_path / "canonical"))
    monkeypatch.setattr(dataset_service, "download", lambda entry, raw_root: mimic_dir)
    return TestClient(app)


def test_catalog_lists_mimic_demo_as_not_ingested(client):
    entries = client.get("/api/v1/datasets/catalog").json()
    mimic = next(e for e in entries if e["key"] == "mimic-iv-demo")
    assert mimic["ingested"] is False and mimic["license"].startswith("Open Database License")


def test_ingest_then_summary_and_preview(client):
    created = client.post("/api/v1/datasets/ingest", json={"key": "mimic-iv-demo"})
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["source"] == "MIMIC_IV" and body["row_count"] == 4

    assert [d["id"] for d in client.get("/api/v1/datasets").json()] == [body["id"]]
    catalog = client.get("/api/v1/datasets/catalog").json()
    assert next(e for e in catalog if e["key"] == "mimic-iv-demo")["dataset_id"] == body["id"]

    summary = client.get(f"/api/v1/datasets/{body['id']}/summary").json()
    assert summary["rows"] == 4 and summary["mortality_rate_pct"] == 25.0

    preview = client.get(f"/api/v1/datasets/{body['id']}/preview?limit=2").json()
    assert preview["total_rows"] == 4 and len(preview["rows"]) == 2
    assert preview["columns"] == etl.CANONICAL_COLUMNS


def test_reingest_is_idempotent(client):
    first = client.post("/api/v1/datasets/ingest", json={"key": "mimic-iv-demo"}).json()
    second = client.post("/api/v1/datasets/ingest", json={"key": "mimic-iv-demo"}).json()
    assert first["id"] == second["id"]
    assert len(client.get("/api/v1/datasets").json()) == 1


def test_ingest_unknown_key_is_404_and_bad_id_is_404(client):
    assert client.post("/api/v1/datasets/ingest", json={"key": "nope"}).status_code == 404
    assert client.get("/api/v1/datasets/00000000-0000-0000-0000-000000000000/summary").status_code == 404


def test_ingest_requires_permission(client):
    from app.authentication.dependencies import get_current_user
    from app.main import app

    viewer = SimpleNamespace(role=SimpleNamespace(permissions={"simulation:run": True}), is_active=True)
    app.dependency_overrides[get_current_user] = lambda: viewer
    assert client.post("/api/v1/datasets/ingest", json={"key": "mimic-iv-demo"}).status_code == 403
    assert client.get("/api/v1/datasets").status_code == 200  # read-only listing stays open to any user
