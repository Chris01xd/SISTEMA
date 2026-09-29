"""Registry + downloader for openly licensed public datasets.

Only datasets that can be fetched WITHOUT credentials belong here. Downloads
go exclusively to the fixed URLs declared in the registry (never a
caller-supplied URL) and every file is verified against the SHA-256 checksums
published by the source before it is trusted.

MIMIC-IV (full) and the other credentialed PhysioNet datasets are deliberately
not listed: they require a signed data use agreement and cannot be
redistributed or auto-downloaded. The open-access *demo* subsets below are
released under the ODbL and use the same schema, so the ETL in
`app/datasets/etl.py` works unchanged on the full data once a credentialed
user points `raw_path` at it.
"""
from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.models.dataset import DatasetSource

PHYSIONET_FILES_URL = "https://physionet.org/files"
_USER_AGENT = "hospital-digital-twin-etl/0.1"
_TIMEOUT_SECONDS = 60


class DatasetDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class PackageSpec:
    """One PhysioNet project (a versioned folder with its own SHA256SUMS.txt)."""

    slug: str
    version: str
    files: tuple[str, ...]  # paths relative to the package root

    def url(self, relative_path: str) -> str:
        return f"{PHYSIONET_FILES_URL}/{self.slug}/{self.version}/{relative_path}"


@dataclass(frozen=True)
class PublicDataset:
    key: str
    name: str
    source: DatasetSource
    version: str
    description: str
    license: str
    homepage: str
    citation: str
    packages: tuple[PackageSpec, ...]

    @property
    def local_dirname(self) -> str:
        return f"{self.key}-{self.version}"


MIMIC_IV_DEMO = PublicDataset(
    key="mimic-iv-demo",
    name="MIMIC-IV Clinical Database Demo",
    source=DatasetSource.MIMIC_IV,
    version="2.2",
    description=(
        "Real, de-identified hospital admissions of 100 patients from Beth Israel Deaconess "
        "Medical Center (admissions, ICU stays, diagnoses) joined with their emergency-department "
        "triage records (MIMIC-IV-ED demo)."
    ),
    license="Open Database License (ODbL) 1.0",
    homepage="https://physionet.org/content/mimic-iv-demo/2.2/",
    citation=(
        "Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). "
        "MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet. https://doi.org/10.13026/dp1f-ex47"
    ),
    packages=(
        PackageSpec(
            slug="mimic-iv-demo",
            version="2.2",
            files=(
                "LICENSE.txt",
                "hosp/admissions.csv.gz",
                "hosp/patients.csv.gz",
                "hosp/diagnoses_icd.csv.gz",
                "hosp/d_icd_diagnoses.csv.gz",
                "icu/icustays.csv.gz",
            ),
        ),
        PackageSpec(
            slug="mimic-iv-ed-demo",
            version="2.2",
            files=("ed/edstays.csv.gz", "ed/triage.csv.gz"),
        ),
    ),
)

CATALOG: dict[str, PublicDataset] = {MIMIC_IV_DEMO.key: MIMIC_IV_DEMO}


def get_public_dataset(key: str) -> PublicDataset:
    try:
        return CATALOG[key]
    except KeyError:
        raise KeyError(f"Unknown public dataset '{key}'. Available: {sorted(CATALOG)}") from None


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DatasetDownloadError(f"Could not download {url}: {exc}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_checksums(text: str) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            checksums[parts[1].lstrip("*")] = parts[0].lower()
    return checksums


def _load_checksums(package: PackageSpec, dest: Path) -> dict[str, str]:
    """Checksums are cached next to the data so a re-run with every file already
    present is verified offline instead of trusting whatever is on disk."""
    cache = dest / f"SHA256SUMS.{package.slug}.txt"
    if cache.exists():
        return _parse_checksums(cache.read_text(encoding="utf-8"))
    text = _http_get(package.url("SHA256SUMS.txt")).decode("utf-8")
    dest.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return _parse_checksums(text)


def download(dataset: PublicDataset, raw_root: Path) -> Path:
    """Fetch every file of `dataset` under `raw_root/<key>-<version>/`, keeping the
    official folder layout (hosp/, icu/, ed/). Returns that directory. Files that are
    already present and match their published checksum are not downloaded again."""
    dest = raw_root / dataset.local_dirname
    for package in dataset.packages:
        checksums = _load_checksums(package, dest)
        for relative in package.files:
            expected = checksums.get(relative)
            if expected is None:
                raise DatasetDownloadError(f"{package.slug}: no published checksum for {relative}")
            target = dest / relative
            if target.exists() and _sha256(target.read_bytes()) == expected:
                continue
            payload = _http_get(package.url(relative))
            if _sha256(payload) != expected:
                raise DatasetDownloadError(f"{package.slug}: checksum mismatch for {relative}; file discarded")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return dest
