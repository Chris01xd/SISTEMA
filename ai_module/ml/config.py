"""Every tunable of the model-comparison engine, in one place (env prefix ``AI_ML_``).

No weight, threshold or penalty used for ranking lives anywhere else in the code base:
the comparator and cross-reference modules read them from ``MLSettings``."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_module.config.settings import MODULE_ROOT


class ScoreWeights(BaseModel):
    """Weights of the CV metrics in ``final_score`` (they must add up to 1)."""

    roc_auc: float = Field(default=0.35, ge=0)
    f1: float = Field(default=0.30, ge=0)
    pr_auc: float = Field(default=0.15, ge=0)
    accuracy: float = Field(default=0.20, ge=0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "ScoreWeights":
        total = self.roc_auc + self.f1 + self.pr_auc + self.accuracy
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"score weights must add up to 1 (got {total:.3f})")
        return self


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_ML_", env_nested_delimiter="__",
        env_file=os.environ.get("AI_ENV_FILE", str(MODULE_ROOT / ".env")), extra="ignore",
    )

    runs_dir: Path = MODULE_ROOT / "artifacts" / "ml_runs"
    keep_runs: int = Field(default=10, ge=1, description="Older run folders are deleted.")

    # --- Protocol --------------------------------------------------------------------
    cv_folds: int = Field(default=5, ge=3, le=10)
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    random_state: int = 42
    min_rows: int = Field(default=60, ge=10)
    min_valid_models: int = Field(default=3, ge=2, description="Below this many trained models the analysis fails.")

    # --- Hyper-parameter tuning (GridSearchCV, nested inside every outer CV fold) -------
    tuning_enabled: bool = True
    tuning_inner_folds: int = Field(default=3, ge=2, le=5)
    n_jobs: int = Field(default=1, description="Parallel jobs for the grid search (-1 = all cores).")

    # --- Inferential statistics --------------------------------------------------------
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    bootstrap_resamples: int = Field(default=1000, ge=100, le=10000)

    # --- Imbalance -------------------------------------------------------------------
    # Minority-class share below which the problem is treated as imbalanced: class weights
    # are enabled, accuracy leaves the score and the primary metric becomes PR-AUC.
    imbalance_threshold: float = Field(default=0.35, gt=0, lt=0.5)

    # --- final_score -----------------------------------------------------------------
    # final_score = sum(weight_m * cv_mean_m) - instability_penalty * cv_std(primary)
    #               - overfitting_penalty * max(0, cv_train_mean(primary) - cv_mean(primary))
    weights_balanced: ScoreWeights = ScoreWeights()
    weights_imbalanced: ScoreWeights = ScoreWeights(roc_auc=0.30, f1=0.35, pr_auc=0.35, accuracy=0.0)
    instability_penalty: float = Field(default=1.0, ge=0)
    overfitting_penalty: float = Field(default=0.5, ge=0)

    # --- Cross-reference flags (differences of the primary metric) --------------------
    # Calibrated to the sampling noise of the data: with ~40-45 cases per validation fold the
    # standard error of a ROC-AUC is ~0.07 (Hanley-McNeil), so smaller spreads are noise.
    overfitting_gap_threshold: float = Field(default=0.15, ge=0)
    instability_std_threshold: float = Field(default=0.08, ge=0)
    cv_holdout_gap_threshold: float = Field(default=0.10, ge=0)

    # --- Explanation -----------------------------------------------------------------
    explain_with_llm: bool = True


@lru_cache
def get_ml_settings() -> MLSettings:
    return MLSettings()
