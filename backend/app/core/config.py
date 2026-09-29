"""Application settings loaded from environment variables (.env)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "Hospital Digital Twin - MCE Resource Optimization"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"

    DATABASE_URL: str = (
        "postgresql+psycopg2://twin_user:twin_pass@localhost:5432/hospital_twin"
    )

    JWT_SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_ENV_FILE"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Simulation defaults
    DEFAULT_TIMESTEP_MINUTES: int = 1
    AI_INFERENCE_EVERY_N_TICKS: int = 15

    # Model artifact storage
    MODEL_ARTIFACT_DIR: str = "app/artificial_intelligence/artifacts"
    REPORT_OUTPUT_DIR: str = "generated_reports"

    # Public datasets (app/datasets/public_datasets.py): raw downloads as published,
    # and the canonical Parquet produced by the ETL.
    PUBLIC_DATASET_RAW_DIR: str = "data/public/raw"
    PUBLIC_DATASET_DIR: str = "data/public/canonical"

    # LangChain assistant (standalone ../langchain_module, see app/services/assistant_service.py).
    # A bare model name means Anthropic; "provider:model" selects another provider (its
    # langchain-<provider> package and API-key env var must then be set up separately).
    ASSISTANT_MODEL: str = "claude-sonnet-5"
    ASSISTANT_TIMEOUT_SECONDS: float = 60.0
    ASSISTANT_MODULE_PATH: str = ""  # empty = <repo>/langchain_module
    ANTHROPIC_API_KEY: str = ""

    # Standalone AI module (../ai_module: LangChain + LangGraph + public-data digital twin, see
    # app/services/ai_module_service.py). Directory that CONTAINS the ai_module package; empty =
    # the repo root. The module reads its own AI_* settings (ai_module/.env.example).
    AI_MODULE_ROOT: str = ""

    # ML model comparison (POST /api/v1/ml/analysis). "langflow": run the Langflow flow
    # through its REST API (the production path); "direct": run the same ai_module steps
    # in-process (no Langflow server, e.g. local development). See langflow_integration/README.md.
    ML_ANALYSIS_MODE: str = "langflow"
    LANGFLOW_URL: str = "http://localhost:7860"
    LANGFLOW_API_KEY: str = ""
    # Flow id (UUID) or the flow's endpoint name, as shown in Langflow's API panel.
    LANGFLOW_FLOW_ID: str = "ml-model-comparison"
    LANGFLOW_TIMEOUT_SECONDS: float = 600.0

    # CRISP-DM module: materialized dataset versions (phase 3) and trained
    # model artifacts (phase 4), kept separate from the hospital AI dir above.
    CRISP_DATASET_DIR: str = "app/crisp_data_artifacts/datasets"
    CRISP_MODEL_ARTIFACT_DIR: str = "app/crisp_data_artifacts/models"
    # Default PSI threshold above which data drift is flagged significant.
    DATA_DRIFT_PSI_THRESHOLD: float = 0.2
    # Default relative accuracy drop (vs. the model's test-set metric) above
    # which model drift is flagged and retraining is suggested.
    MODEL_DRIFT_ACCURACY_DROP_THRESHOLD: float = 0.15


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
