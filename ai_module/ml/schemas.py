"""Contracts of the model-comparison analysis. Every step exchanges JSON-serializable
dicts validated by these models, so the same steps can run in-process or as Langflow
components passing ``Data`` objects between nodes."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelFamily = Literal["normal", "hybrid"]
ModelStatus = Literal["ok", "failed", "skipped"]


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(default="mimic_icu", description="Key of a registered dataset (see ai_module.ml.datasets).")
    target: str | None = Field(default=None, description="Default: the dataset's default target.")
    problem_type: Literal["classification"] = "classification"
    features: list[str] | None = Field(default=None, description="Subset of the dataset's features (default: all).")
    cv_folds: int | None = Field(default=None, ge=3, le=10)
    test_size: float | None = Field(default=None, gt=0.05, lt=0.5)
    random_state: int | None = None
    models: list[str] | None = Field(default=None, description="Subset of model keys (default: all).")
    explain_with_llm: bool | None = None


class SetMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None


class ModelTrainingResult(BaseModel):
    key: str
    name: str
    family: ModelFamily
    status: ModelStatus
    error: str | None = None
    description: str = ""
    components: list[str] = Field(default_factory=list)
    train_metrics: SetMetrics | None = None
    test_metrics: SetMetrics | None = None
    confusion_matrix: list[list[int]] | None = None
    training_time_s: float | None = None
    inference_time_s: float | None = None
    inference_ms_per_sample: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    tuning: dict[str, Any] = Field(default_factory=dict)
    curves: dict[str, Any] = Field(default_factory=dict)


class FoldResult(BaseModel):
    fold: int
    train_size: int
    validation_size: int
    validation: dict[str, float | None]
    train: dict[str, float | None]
    fit_time_s: float
    best_params: dict[str, Any] = Field(default_factory=dict)


class ModelCVResult(BaseModel):
    key: str
    name: str
    family: ModelFamily
    status: ModelStatus
    error: str | None = None
    folds: list[FoldResult] = Field(default_factory=list)
    mean: dict[str, float | None] = Field(default_factory=dict)
    std: dict[str, float | None] = Field(default_factory=dict)
    train_mean: dict[str, float | None] = Field(default_factory=dict)
    cv_scores: list[float] = Field(default_factory=list, description="Primary metric per validation fold.")
    cv_mean: float | None = None
    cv_std: float | None = None
    tuning_stability: float | None = Field(default=None, description="Share of outer folds that chose the most common hyper-parameters.")


class EvaluationRow(BaseModel):
    """One row of the unified comparison table (all models, same metrics)."""

    model: str
    key: str
    type: ModelFamily
    status: ModelStatus
    error: str | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    cv_mean: float | None = None
    cv_std: float | None = None
    cv_metrics: dict[str, float | None] = Field(default_factory=dict)
    cv_train_mean: float | None = None
    train_score: float | None = None
    test_score: float | None = None
    overfitting_gap: float | None = None
    cv_overfitting_gap: float | None = None
    training_time_s: float | None = None
    inference_time_s: float | None = None
    inference_ms_per_sample: float | None = None
    confusion_matrix: list[list[int]] | None = None


class RankingEntry(BaseModel):
    rank: int
    model: str
    key: str
    type: ModelFamily
    final_score: float
    score_breakdown: dict[str, float]


class BestModel(BaseModel):
    name: str
    key: str
    type: ModelFamily
    final_score: float
    rank: int
    metrics: dict[str, Any]
    artifact_path: str
    metadata_path: str
