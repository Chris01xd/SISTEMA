"""Per-run storage: ``<runs_dir>/<run_id>/``.

Steps exchange small JSON reports; heavy objects (the dataset frame, the split, fitted
models) stay on disk under the run folder, so a Langflow node only passes ``run_id``
downstream and nothing large or non-serializable crosses node boundaries."""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ai_module.ml.errors import RunNotFoundError

_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


class RunStore:
    def __init__(self, runs_dir: Path, keep_runs: int = 10) -> None:
        self.root = runs_dir
        self.keep_runs = keep_runs

    def create(self) -> str:
        run_id = new_run_id()
        (self.root / run_id / "models").mkdir(parents=True, exist_ok=False)
        self._prune()
        return run_id

    def path(self, run_id: str) -> Path:
        # run_id arrives from API/Langflow payloads: never let it address anything outside root.
        if not _RUN_ID.match(run_id or ""):
            raise RunNotFoundError(f"Invalid run id '{run_id}'")
        path = self.root / run_id
        if not path.is_dir():
            raise RunNotFoundError(f"Run '{run_id}' not found (runs are kept only for the last {self.keep_runs} analyses)")
        return path

    def save_object(self, run_id: str, name: str, obj: Any) -> Path:
        target = self.path(run_id) / f"{name}.joblib"
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, target)
        return target

    def load_object(self, run_id: str, name: str) -> Any:
        target = self.path(run_id) / f"{name}.joblib"
        if not target.exists():
            raise RunNotFoundError(f"Run '{run_id}' has no '{name}' (did the previous step run?)")
        return joblib.load(target)

    def has_object(self, run_id: str, name: str) -> bool:
        return (self.path(run_id) / f"{name}.joblib").exists()

    def save_json(self, run_id: str, name: str, payload: dict[str, Any]) -> Path:
        target = self.path(run_id) / f"{name}.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return target

    def load_json(self, run_id: str, name: str) -> dict[str, Any]:
        target = self.path(run_id) / f"{name}.json"
        if not target.exists():
            raise RunNotFoundError(f"Run '{run_id}' has no '{name}.json'")
        return json.loads(target.read_text(encoding="utf-8"))

    def _prune(self) -> None:
        runs = sorted((p for p in self.root.iterdir() if p.is_dir() and _RUN_ID.match(p.name)), key=lambda p: p.name)
        for old in runs[: max(0, len(runs) - self.keep_runs)]:
            shutil.rmtree(old, ignore_errors=True)
