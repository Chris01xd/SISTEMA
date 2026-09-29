"""Hyper-parameter search shared by training and nested cross-validation.

``fit_tuned`` runs ``GridSearchCV`` over the model's grid with an *inner*
StratifiedGroupKFold on whatever training data it receives. Called once per outer CV fold,
it yields nested CV: the hyper-parameters are chosen without ever seeing the outer
validation fold, so the CV estimate is not optimistically biased by the tuning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ai_module.ml.catalog import BuildContext, ModelSpec
from ai_module.ml.config import MLSettings
from ai_module.ml.metrics import CV_SCORERS


@dataclass
class TuningResult:
    model: Pipeline
    best_params: dict[str, Any] = field(default_factory=dict)
    best_inner_score: float | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)


def _clean(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def short_params(params: dict[str, Any]) -> dict[str, Any]:
    """'model__estimator__C' -> 'C' for display (keys stay unique within one grid)."""
    return {k.split("__")[-1]: _clean(v) for k, v in params.items()}


def fit_tuned(spec: ModelSpec, ctx: BuildContext, x: pd.DataFrame, y: pd.Series, groups: pd.Series | None,
              primary_metric: str, ml: MLSettings, seed: int) -> TuningResult:
    estimator = spec.build(ctx)
    if not (ml.tuning_enabled and spec.param_grid):
        estimator.fit(x, y)
        return TuningResult(model=estimator)
    if groups is not None:
        inner = StratifiedGroupKFold(n_splits=ml.tuning_inner_folds, shuffle=True, random_state=seed)
    else:
        inner = StratifiedKFold(n_splits=ml.tuning_inner_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator, spec.param_grid, scoring=CV_SCORERS[primary_metric], cv=inner,
                          refit=True, n_jobs=ml.n_jobs, error_score="raise")
    search.fit(x, y, groups=groups)
    results = search.cv_results_
    candidates = [
        {"params": short_params(results["params"][i]), "mean_score": round(float(results["mean_test_score"][i]), 4),
         "std_score": round(float(results["std_test_score"][i]), 4), "rank": int(results["rank_test_score"][i])}
        for i in range(len(results["params"]))
    ]
    return TuningResult(model=search.best_estimator_, best_params=short_params(search.best_params_),
                        best_inner_score=round(float(search.best_score_), 4), candidates=sorted(candidates, key=lambda c: c["rank"]))
