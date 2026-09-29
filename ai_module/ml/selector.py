"""Step 8: BestModelSelector, takes rank 1 of the computed ranking and persists that fitted
model plus its metadata. There is no hard-coded winner anywhere: the name comes only from
``ranking[0]``, which the comparator derived from the real scores."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_module.digital_twin.registry import library_versions
from ai_module.ml.errors import IncompleteResultError
from ai_module.ml.log import ml_log
from ai_module.ml.run_store import RunStore
from ai_module.ml.schemas import BestModel

METRIC_FIELDS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cv_mean", "cv_std", "cv_metrics",
                 "train_score", "test_score", "overfitting_gap", "cv_overfitting_gap", "training_time_s",
                 "inference_time_s", "inference_ms_per_sample", "confusion_matrix")


class BestModelSelector:
    def __init__(self, store: RunStore, best_dir: Path) -> None:
        self.store = store
        self.best_dir = best_dir

    def select(self, run_id: str, comparison: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
        ranking = comparison.get("ranking") or []
        if not ranking:
            raise IncompleteResultError("The ranking is empty: no model can be selected")
        top = ranking[0]
        row = next((r for r in evaluation["models"] if r["key"] == top["key"]), None)
        if row is None or row["status"] != "ok":
            raise IncompleteResultError(f"The top-ranked model '{top['model']}' has no valid evaluation")

        cfg = self.store.load_json(run_id, "config")
        model = self.store.load_object(run_id, f"models/{top['key']}")
        target_dir = self.best_dir / run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        artifact = target_dir / "best_model.joblib"
        shutil.copyfile(self.store.path(run_id) / "models" / f"{top['key']}.joblib", artifact)
        metrics = {k: row[k] for k in METRIC_FIELDS}
        metadata = {
            "model_name": top["model"], "model_key": top["key"], "model_type": top["type"],
            "final_score": top["final_score"], "rank": top["rank"], "score_breakdown": top["score_breakdown"],
            "score_config": comparison["score_config"], "metrics": metrics,
            "features": [c for group in cfg["features"].values() for c in group], "feature_groups": cfg["features"],
            "target": cfg["target"], "dataset": cfg["dataset"], "dataset_source": cfg["source"],
            "problem_type": cfg["problem_type"], "primary_metric": cfg["primary_metric"],
            "cv_folds": cfg["cv_folds"], "test_size": cfg["test_size"], "random_state": cfg["random_state"],
            "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": run_id,
            "estimator": type(model.named_steps["model"]).__name__, "library_versions": library_versions(),
        }
        metadata_path = target_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (self.best_dir / "latest.json").write_text(json.dumps({"run_id": run_id, "model_name": top["model"]}), encoding="utf-8")

        best = BestModel(name=top["model"], key=top["key"], type=top["type"], final_score=top["final_score"], rank=1,
                         metrics=metrics, artifact_path=str(artifact), metadata_path=str(metadata_path))
        ml_log("BEST MODEL", f"{best.name} selected (final_score={best.final_score})", run_id=run_id, model=best.key)
        return {"run_id": run_id, "best_model": best.model_dump(), "metadata": metadata}
