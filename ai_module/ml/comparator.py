"""Step 7: ModelComparator, computes ``final_score`` for every valid model and ranks them.

    final_score = sum(weight_m * cv_mean_m for m in {roc_auc, f1, pr_auc, accuracy})
                  - instability_penalty * cv_std(primary)
                  - overfitting_penalty * max(0, cv_overfitting_gap(primary))

Selection uses cross-validated metrics (5 validation folds) instead of the single
hold-out, which is too small to choose on without selection bias; the hold-out is used by
the cross-reference step to confirm generalization. Normal and hybrid models are scored by
exactly the same formula. Weights and penalties come only from ``MLSettings``.
Ties are broken by lower CV std, then shorter training time, then name."""
from __future__ import annotations

from typing import Any

from ai_module.ml.config import MLSettings
from ai_module.ml.errors import NotEnoughModelsError
from ai_module.ml.log import ml_log
from ai_module.ml.schemas import RankingEntry


class ModelComparator:
    def __init__(self, settings: MLSettings) -> None:
        self.settings = settings

    def weights(self, imbalanced: bool) -> dict[str, float]:
        return (self.settings.weights_imbalanced if imbalanced else self.settings.weights_balanced).model_dump()

    def score(self, row: dict[str, Any], imbalanced: bool) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {}
        total = 0.0
        for metric, weight in self.weights(imbalanced).items():
            value = row["cv_metrics"].get(metric)
            contribution = weight * (value or 0.0)  # a metric that cannot be computed contributes nothing
            breakdown[f"{metric}_contribution"] = round(contribution, 4)
            total += contribution
        stability = self.settings.instability_penalty * (row["cv_std"] or 0.0)
        overfitting = self.settings.overfitting_penalty * max(0.0, row["cv_overfitting_gap"] or 0.0)
        breakdown["instability_penalty"] = round(-stability, 4)
        breakdown["overfitting_penalty"] = round(-overfitting, 4)
        return round(total - stability - overfitting, 4), breakdown

    def compare(self, run_id: str, evaluation: dict[str, Any], imbalanced: bool) -> dict[str, Any]:
        ml_log("COMPARATOR", "Ranking models", run_id=run_id)
        valid = [r for r in evaluation["models"] if r["status"] == "ok"]
        if len(valid) < self.settings.min_valid_models:
            raise NotEnoughModelsError(
                f"Only {len(valid)} models trained and validated successfully; at least {self.settings.min_valid_models} are required. "
                f"Failures: {evaluation['failed_models']}"
            )
        scored = []
        for row in valid:
            final_score, breakdown = self.score(row, imbalanced)
            scored.append((final_score, row, breakdown))
        scored.sort(key=lambda item: (-item[0], item[1]["cv_std"], item[1]["training_time_s"], item[1]["model"]))
        ranking = [RankingEntry(rank=i + 1, model=row["model"], key=row["key"], type=row["type"], final_score=score, score_breakdown=bd)
                   for i, (score, row, bd) in enumerate(scored)]
        for entry in ranking:
            ml_log("COMPARATOR", f"{entry.rank}. {entry.model} ({entry.type}) final_score={entry.final_score}", run_id=run_id)
        return {
            "run_id": run_id,
            "ranking": [e.model_dump() for e in ranking],
            "models_evaluated": len(evaluation["models"]),
            "models_ranked": len(ranking),
            "normal_models_ranked": sum(e.type == "normal" for e in ranking),
            "hybrid_models_ranked": sum(e.type == "hybrid" for e in ranking),
            "score_config": {
                "formula": "sum(weight_m * cv_mean_m) - instability_penalty * cv_std - overfitting_penalty * max(0, cv_overfitting_gap)",
                "weights": self.weights(imbalanced), "instability_penalty": self.settings.instability_penalty,
                "overfitting_penalty": self.settings.overfitting_penalty, "imbalanced": imbalanced,
                "primary_metric": evaluation["primary_metric"],
            },
        }
