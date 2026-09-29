"""Versioned storage of trained models: ``<models_dir>/<name>/<version>/{model.joblib, metadata.json}``
plus ``latest.json`` pointing at the version inference should load."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from ai_module.digital_twin.errors import ModelArtifactError, ModelNotTrainedError

MODEL_NAME = "admission_outcome"


def library_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def new_version(dataset_sha256: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{dataset_sha256[:8]}"


@dataclass
class LoadedModel:
    models: dict[str, Any]
    metadata: dict[str, Any]
    path: Path


class ModelRegistry:
    def __init__(self, models_dir: Path, name: str = MODEL_NAME) -> None:
        self.root = models_dir / name

    def save(self, version: str, models: dict[str, Any], metadata: dict[str, Any]) -> Path:
        target = self.root / version
        target.mkdir(parents=True, exist_ok=False)
        joblib.dump(models, target / "model.joblib")
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.root / "latest.json").write_text(json.dumps({"version": version}), encoding="utf-8")
        return target

    def latest_version(self) -> str:
        pointer = self.root / "latest.json"
        if not pointer.exists():
            raise ModelNotTrainedError(f"{pointer} not found")
        return str(json.loads(pointer.read_text(encoding="utf-8"))["version"])

    def load(self, version: str | None = None) -> LoadedModel:
        version = version or self.latest_version()
        path = self.root / version
        if not (path / "model.joblib").exists():
            raise ModelNotTrainedError(f"version {version} has no model.joblib")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        trained_with = metadata.get("library_versions", {}).get("scikit-learn")
        # scikit-learn pickles are not portable across versions: an older runtime may unpickle
        # a newer model "successfully" and then fail (or silently misbehave) at predict time.
        if trained_with != sklearn.__version__:
            raise ModelArtifactError(
                f"Model {version} was trained with scikit-learn {trained_with} but this environment runs "
                f"{sklearn.__version__}. Retrain it here: python -m ai_module.digital_twin.train"
            )
        try:
            models = joblib.load(path / "model.joblib")
        except Exception as exc:  # corrupted or unreadable artifact
            raise ModelArtifactError(f"Model {version} could not be loaded ({type(exc).__name__}). Retrain it.") from exc
        return LoadedModel(models=models, metadata=metadata, path=path)

    def write_json(self, version: str, filename: str, payload: dict[str, Any]) -> Path:
        target = self.root / version / filename
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target
