"""Step 1-2 of the pipeline: download (checksum-verified) and read the raw public tables.

Files already present and matching their published checksum are not downloaded again,
so re-running the pipeline is idempotent and works offline once the data is cached."""
from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_module.config.logging import get_logger, timed
from ai_module.digital_twin.errors import DatasetDownloadError
from ai_module.digital_twin.sources import DatasetSpec, PackageSpec

logger = get_logger(__name__)

_USER_AGENT = "ai-module-digital-twin/1.0"
_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class RawTables:
    """The raw MIMIC tables the twin needs, exactly as published."""

    admissions: pd.DataFrame
    patients: pd.DataFrame
    icustays: pd.DataFrame
    edstays: pd.DataFrame
    triage: pd.DataFrame


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DatasetDownloadError(f"Could not download {url}: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def parse_checksums(text: str) -> dict[str, str]:
    """Parses a ``SHA256SUMS.txt`` (``<hex>  <relative path>`` per line)."""
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            checksums[parts[1].lstrip("*")] = parts[0].lower()
    return checksums


def _checksums(package: PackageSpec, dest: Path) -> dict[str, str]:
    # Cached next to the data so a later run verifies files offline.
    cache = dest / f"SHA256SUMS.{package.slug}.txt"
    if cache.exists():
        return parse_checksums(cache.read_text(encoding="utf-8"))
    text = _http_get(package.url("SHA256SUMS.txt")).decode("utf-8")
    dest.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return parse_checksums(text)


def download_dataset(spec: DatasetSpec, raw_root: Path) -> Path:
    """Downloads every file of ``spec`` into ``raw_root/<key>-<version>/`` keeping the
    official layout (hosp/, icu/, ed/) and returns that directory."""
    dest = raw_root / spec.local_dirname
    for package in spec.packages:
        checksums = _checksums(package, dest)
        # LICENSE.txt exists in both packages: keep each under its own name.
        for relative in package.files:
            expected = checksums.get(relative)
            if expected is None:
                raise DatasetDownloadError(f"{package.slug}: no published checksum for {relative}")
            target = dest / (f"LICENSE.{package.slug}.txt" if relative == "LICENSE.txt" else relative)
            if target.exists() and sha256_file(target) == expected:
                continue
            with timed(logger, "dataset_file_download", package=package.slug, file=relative) as info:
                payload = _http_get(package.url(relative))
                info["bytes"] = len(payload)
            if sha256_bytes(payload) != expected:
                raise DatasetDownloadError(f"{package.slug}: checksum mismatch for {relative}; file discarded")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return dest


def _read(raw_dir: Path, module: str, name: str) -> pd.DataFrame:
    for base in (raw_dir / module, raw_dir):
        for suffix in (".csv.gz", ".csv"):
            path = base / f"{name}{suffix}"
            if path.exists():
                return pd.read_csv(path, low_memory=False)
    raise DatasetDownloadError(f"Raw table '{module}/{name}' not found under {raw_dir}. Run the prepare step with download enabled.")


def load_raw_tables(raw_dir: Path) -> RawTables:
    """Reads the raw tables from a folder in the official layout (or a flat folder of CSVs)."""
    return RawTables(
        admissions=_read(raw_dir, "hosp", "admissions"),
        patients=_read(raw_dir, "hosp", "patients"),
        icustays=_read(raw_dir, "icu", "icustays"),
        edstays=_read(raw_dir, "ed", "edstays"),
        triage=_read(raw_dir, "ed", "triage"),
    )
