"""Runtime configuration of the AI module, read from environment variables (prefix ``AI_``).

Nothing here is hard-coded business data: every value is either a technical default
(timeouts, simulation run counts) or an explicit, documented *assumption* of the digital
twin (e.g. baseline daily admissions) that the caller can override per request.

Sources, highest priority first: explicit keyword arguments -> process environment ->
the env file named by ``AI_ENV_FILE`` (default ``ai_module/.env``)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _env_file() -> Path:
    return Path(os.environ.get("AI_ENV_FILE", MODULE_ROOT / ".env"))


class AISettings(BaseSettings):
    """All tunables of the module. See ``ai_module/.env.example`` for documentation."""

    model_config = SettingsConfigDict(env_prefix="AI_", env_file=str(_env_file()), extra="ignore")

    # --- Storage -------------------------------------------------------------------
    data_dir: Path = MODULE_ROOT / "data"
    artifacts_dir: Path = MODULE_ROOT / "artifacts"

    # --- LLM provider (LangChain) ----------------------------------------------------
    # "anthropic", "openai", "ollama", ... (any provider supported by
    # langchain.chat_models.init_chat_model whose integration package is installed),
    # or "none" to run without an LLM (deterministic template responses).
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_tokens: int = Field(default=1500, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_base_url: str | None = None  # e.g. an Ollama or OpenAI-compatible gateway
    # Optional: when empty the provider SDK reads its own variable (ANTHROPIC_API_KEY, OPENAI_API_KEY...).
    llm_api_key: SecretStr | None = Field(default=None, repr=False)

    # --- LangGraph -------------------------------------------------------------------
    graph_max_regenerations: int = Field(default=1, ge=0, le=3)
    max_message_chars: int = Field(default=2000, gt=0)

    # --- Digital twin assumptions (overridable per scenario) ----------------------------
    twin_baseline_daily_admissions: float = Field(default=20.0, gt=0)
    # None = sized from the observed length of stay with Little's law at `twin_target_occupancy`
    # (80%: below the 85% the backend 3D twin paints RED, so the baseline hospital is busy but functional).
    twin_ward_beds: int | None = Field(default=None, gt=0)
    twin_icu_beds: int | None = Field(default=None, gt=0)
    twin_target_occupancy: float = Field(default=0.80, gt=0, le=1)
    twin_sim_runs: int = Field(default=100, ge=1, le=1000)
    twin_sim_horizon_hours: int = Field(default=168, ge=1, le=720)
    twin_sim_warmup_hours: int = Field(default=720, ge=0, le=2160)
    twin_random_seed: int = 42

    # --- Observability ---------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("llm_provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("llm_api_key", "llm_base_url", "twin_ward_beds", "twin_icu_beds", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        # `AI_TWIN_ICU_BEDS=` in an env file means "not configured", not an invalid number.
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, SecretStr) and not value.get_secret_value():
            return None
        return value

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider not in ("", "none")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"


@lru_cache
def get_settings() -> AISettings:
    """Process-wide settings instance (tests should build ``AISettings(...)`` directly)."""
    return AISettings()
