"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# <repo>/langchain_module/hospital_assistant/config.py -> <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = _REPO_ROOT / "backend" / "data" / "public" / "canonical" / "mimic-iv-demo-2.2.parquet"
DEFAULT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class Settings:
    dataset_path: Path
    model: str
    temperature: float
    timeout: float = 60.0  # seconds per LLM request
    api_key: str | None = field(default=None, repr=False)  # None = let the provider SDK read its env var

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            dataset_path=Path(os.environ.get("HOSPITAL_DATASET_PATH", DEFAULT_DATASET_PATH)),
            # "provider:model" (e.g. "openai:gpt-4o") selects another provider if its
            # langchain integration package is installed; a bare name means Anthropic.
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0")),
            timeout=float(os.environ.get("LLM_TIMEOUT", "60")),
        )
