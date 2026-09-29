"""Step 7: persistence of the processed data and of the digital state (local files).

Files under ``<data_dir>/processed/``:
  admissions.parquet  cleaned one-row-per-admission table
  data_report.json    audit trail of the cleaning step
  provenance.json     dataset source, license, checksum of the processed table
  twin_state.json     last built TwinState snapshot

A file store is enough for a single-process prototype; the interface is small so it can
be replaced by a database/object store without touching the service."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ai_module.digital_twin.errors import TwinDataNotPreparedError
from ai_module.digital_twin.ingest import sha256_file
from ai_module.schemas.twin import Provenance, TwinState


class TwinDataStore:
    def __init__(self, processed_dir: Path) -> None:
        self.dir = processed_dir

    @property
    def admissions_path(self) -> Path:
        return self.dir / "admissions.parquet"

    @property
    def state_path(self) -> Path:
        return self.dir / "twin_state.json"

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_processed(self, df: pd.DataFrame, report: dict[str, Any]) -> str:
        """Writes the processed table and its report; returns the table's SHA-256."""
        self.dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.admissions_path, index=False)
        self._write_json("data_report.json", report)
        return self.processed_sha256()

    def save_provenance(self, provenance: Provenance) -> None:
        self._write_json("provenance.json", provenance.model_dump())

    def processed_sha256(self) -> str:
        if not self.admissions_path.exists():
            raise TwinDataNotPreparedError(f"{self.admissions_path.name} not found")
        return sha256_file(self.admissions_path)

    def load_processed(self) -> pd.DataFrame:
        if not self.admissions_path.exists():
            raise TwinDataNotPreparedError(f"{self.admissions_path.name} not found")
        return pd.read_parquet(self.admissions_path)

    def load_provenance(self) -> Provenance:
        path = self.dir / "provenance.json"
        if not path.exists():
            raise TwinDataNotPreparedError("provenance.json not found")
        return Provenance.model_validate_json(path.read_text(encoding="utf-8"))

    def save_state(self, state: TwinState) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def load_state(self) -> TwinState | None:
        if not self.state_path.exists():
            return None
        return TwinState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
