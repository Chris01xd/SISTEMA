"""Contract of the ML model-comparison endpoints. The response schema is also used to
check that whatever Langflow returned is complete before it reaches the frontend."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MLAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(default="mimic_icu", max_length=64)
    target: str | None = Field(default=None, max_length=64)
    problem_type: Literal["classification"] = "classification"
    features: list[str] | None = Field(default=None, max_length=50)
    cv_folds: int | None = Field(default=None, ge=3, le=10)
    test_size: float | None = Field(default=None, gt=0.05, lt=0.5)
    random_state: int | None = None
    models: list[str] | None = Field(default=None, max_length=20)
    explain_with_llm: bool | None = None
    language: Literal["es", "en"] = "es"


class RankingEntry(BaseModel):
    rank: int
    model: str
    key: str
    type: Literal["normal", "hybrid"]
    final_score: float
    score_breakdown: dict[str, float]


class BestModel(BaseModel):
    name: str
    key: str
    type: Literal["normal", "hybrid"]
    final_score: float
    rank: int
    metrics: dict[str, Any]
    artifact_path: str
    metadata_path: str


class MLAnalysisResult(BaseModel):
    """Final output of the flow (see ai_module/ml/service.py::build_output)."""

    status: Literal["success"]
    run_id: str
    models_evaluated: int = Field(ge=1)
    models_attempted: int
    normal_models: list[dict[str, Any]] = Field(min_length=1)
    hybrid_models: list[dict[str, Any]] = Field(min_length=1)
    comparison_table: list[dict[str, Any]] = Field(min_length=1)
    ranking: list[RankingEntry] = Field(min_length=1)
    best_model: BestModel
    best_model_metadata: dict[str, Any]
    cross_validation: dict[str, Any]
    cross_reference_analysis: dict[str, Any]
    statistical_tests: dict[str, Any]
    hyperparameters: dict[str, Any] = Field(min_length=1)
    curves: dict[str, Any]
    eda: dict[str, Any]
    # One interpretation per table/chart (shown on screen and in the reports).
    interpretations: dict[str, str] = Field(min_length=1)
    crisp_dm: list[dict[str, str]] = Field(min_length=6)
    language: str = "es"
    score_config: dict[str, Any]
    analysis: str = Field(min_length=1)
    analysis_source: Literal["llm", "deterministic"]
    selection_reason: str = Field(min_length=1)
    dataset: dict[str, Any]
    preprocessing: dict[str, Any]
    warnings: list[str]
    errors: list[dict[str, Any]]
    duration_s: float | None = None
    # Added by the backend: which engine ran the analysis and the Langflow run reference.
    engine: Literal["langflow", "direct"] = "langflow"
    langflow: dict[str, Any] | None = None
