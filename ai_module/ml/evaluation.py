"""Step 5: ModelEvaluationService, one unified table with the same metrics for every model.

Column semantics:
  accuracy..pr_auc      hold-out (test partition) metrics at threshold 0.5
  cv_mean / cv_std      primary metric over the CV validation folds
  cv_metrics            mean of every metric over the CV validation folds
  train_score           primary metric of the final fit on its own training data
  test_score            primary metric on the hold-out
  overfitting_gap       train_score - test_score
  cv_overfitting_gap    mean CV train-fold score - mean CV validation-fold score
"""
from __future__ import annotations

from typing import Any

from ai_module.ml.log import ml_log
from ai_module.ml.schemas import EvaluationRow


def _gap(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(a - b, 4)


class ModelEvaluationService:
    def evaluate(self, run_id: str, training_reports: list[dict[str, Any]], cv_report: dict[str, Any]) -> dict[str, Any]:
        primary = cv_report["primary_metric"]
        ml_log("EVALUATION", "Evaluating models", run_id=run_id)
        rows: list[EvaluationRow] = []
        for report in training_reports:
            for model in report["models"]:
                cv = cv_report["models"].get(model["key"], {})
                status = model["status"] if model["status"] != "ok" else cv.get("status", "skipped")
                error = model.get("error") or cv.get("error")
                row = EvaluationRow(model=model["name"], key=model["key"], type=model["family"], status=status, error=error)
                if status == "ok":
                    test, train = model["test_metrics"], model["train_metrics"]
                    row = row.model_copy(update={
                        **{m: test[m] for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
                        "cv_mean": cv["cv_mean"], "cv_std": cv["cv_std"], "cv_metrics": cv["mean"],
                        "cv_train_mean": cv["train_mean"][primary],
                        "train_score": train[primary], "test_score": test[primary],
                        "overfitting_gap": _gap(train[primary], test[primary]),
                        "cv_overfitting_gap": _gap(cv["train_mean"][primary], cv["cv_mean"]),
                        "training_time_s": model["training_time_s"], "inference_time_s": model["inference_time_s"],
                        "inference_ms_per_sample": model["inference_ms_per_sample"], "confusion_matrix": model["confusion_matrix"],
                    })
                rows.append(row)
        return {
            "run_id": run_id, "primary_metric": primary,
            "models": [r.model_dump() for r in rows],
            "valid_models": sum(r.status == "ok" for r in rows),
            "failed_models": [{"model": r.model, "status": r.status, "error": r.error} for r in rows if r.status != "ok"],
        }
