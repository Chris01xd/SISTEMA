"""Step 4: real training of the normal models and of the hybrid ensembles.

Each model is tuned with GridSearchCV (inner grouped CV on the training partition) and the
best configuration is refitted on the whole training partition. It is then scored on the
training partition (for the train/test gap) and on the untouched hold-out, timed, and
saved to the run folder. Hold-out probabilities are saved for the statistical tests. A
model that fails is recorded with its error; the others continue."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from ai_module.ml.catalog import get_specs
from ai_module.ml.config import MLSettings
from ai_module.ml.log import ml_log
from ai_module.ml.metrics import compute_metrics, confusion
from ai_module.ml.preprocessing import load_partitions
from ai_module.ml.run_store import RunStore
from ai_module.ml.schemas import ModelFamily, ModelTrainingResult
from ai_module.ml.tuning import fit_tuned

_TAG = {"normal": "ML", "hybrid": "HYBRID"}
_CURVE_POINTS = 60


def _params(model: Any) -> dict[str, Any]:
    estimator = model.named_steps["model"]
    params = estimator.get_params(deep=False)
    return {k: repr(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
            for k, v in params.items() if k not in ("estimators", "final_estimator", "cv", "estimator")}


def _downsample(xs: np.ndarray, ys: np.ndarray) -> dict[str, list[float]]:
    idx = np.unique(np.linspace(0, len(xs) - 1, min(len(xs), _CURVE_POINTS)).astype(int))
    return {"x": [round(float(v), 4) for v in xs[idx]], "y": [round(float(v), 4) for v in ys[idx]]}


def curves(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    if len(np.unique(y_true)) < 2:
        return {}
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    return {"roc": _downsample(fpr, tpr), "pr": _downsample(recall[::-1], precision[::-1])}


def train_family(run_id: str, family: ModelFamily, store: RunStore, ml: MLSettings) -> dict[str, Any]:
    cfg, ctx, x_train, y_train, x_test, y_test, groups = load_partitions(run_id, store)
    if not store.has_object(run_id, "predictions/y_test"):
        store.save_object(run_id, "predictions/y_test", y_test.to_numpy())
    results: list[ModelTrainingResult] = []
    for spec in get_specs(family, cfg["models"]):
        base = {"key": spec.key, "name": spec.name, "family": spec.family, "description": spec.description, "components": list(spec.components)}
        if not spec.available():
            ml_log(_TAG[family], f"{spec.name} skipped: optional dependency '{spec.requires}' is not installed", run_id=run_id, model=spec.key)
            results.append(ModelTrainingResult(**base, status="skipped", error=f"optional dependency '{spec.requires}' is not installed"))
            continue
        grid_size = int(np.prod([len(v) for v in spec.param_grid.values()])) if (ml.tuning_enabled and spec.param_grid) else 1
        ml_log(_TAG[family], f"Training {spec.name} (grid search over {grid_size} configurations)", run_id=run_id, model=spec.key)
        try:
            start = time.perf_counter()
            tuned = fit_tuned(spec, ctx, x_train, y_train, groups, cfg["primary_metric"], ml, cfg["random_state"])
            training_time = time.perf_counter() - start
            model = tuned.model
            train_proba = model.predict_proba(x_train)[:, 1]
            start = time.perf_counter()
            test_proba = model.predict_proba(x_test)[:, 1]
            inference_time = time.perf_counter() - start
            store.save_object(run_id, f"models/{spec.key}", model)
            store.save_object(run_id, f"predictions/{spec.key}", test_proba)
            results.append(ModelTrainingResult(
                **base, status="ok",
                train_metrics=compute_metrics(y_train, train_proba), test_metrics=compute_metrics(y_test, test_proba),
                confusion_matrix=confusion(y_test, test_proba),
                training_time_s=round(training_time, 4), inference_time_s=round(inference_time, 5),
                inference_ms_per_sample=round(1000 * inference_time / max(len(x_test), 1), 4), params=_params(model),
                tuning={"enabled": grid_size > 1, "grid_size": grid_size, "best_params": tuned.best_params,
                        "best_inner_score": tuned.best_inner_score, "inner_folds": ml.tuning_inner_folds if grid_size > 1 else None,
                        "scoring": cfg["primary_metric"], "candidates": tuned.candidates},
                curves=curves(y_test.to_numpy(), test_proba),
            ))
            ml_log(_TAG[family], f"{spec.name} trained in {training_time:.2f}s, best params {tuned.best_params}", run_id=run_id, model=spec.key)
        except Exception as exc:  # one failing model must not cancel the analysis
            ml_log(_TAG[family], f"{spec.name} failed: {type(exc).__name__}", 30, run_id=run_id, model=spec.key)
            results.append(ModelTrainingResult(**base, status="failed", error=f"{type(exc).__name__}: {str(exc)[:200]}"))
    report = {"run_id": run_id, "family": family, "models": [r.model_dump() for r in results],
              "trained": sum(r.status == "ok" for r in results), "failed": sum(r.status == "failed" for r in results)}
    store.save_json(run_id, f"training_{family}", report)
    return report
