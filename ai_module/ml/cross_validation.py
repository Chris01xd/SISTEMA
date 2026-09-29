"""Step 5: nested k-fold cross-validation of every successfully trained model.

Outer loop: StratifiedGroupKFold on the training partition (the hold-out is never touched),
class balance preserved per fold, no patient in both training and validation folds. Inner
loop: the grid search runs on each outer training fold only (``tuning.fit_tuned``), so the
validation-fold scores estimate the performance of the *whole* procedure, tuning included.
The Pipeline (imputation, scaling, encoding) is refitted in every fit: no leakage."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ai_module.ml.catalog import get_specs
from ai_module.ml.config import MLSettings
from ai_module.ml.log import ml_log
from ai_module.ml.metrics import METRIC_NAMES, compute_metrics
from ai_module.ml.preprocessing import load_partitions
from ai_module.ml.run_store import RunStore
from ai_module.ml.schemas import FoldResult, ModelCVResult
from ai_module.ml.tuning import fit_tuned


def _stat(values: list[float | None], fn) -> float | None:
    clean = [v for v in values if v is not None]
    return round(float(fn(clean)), 4) if clean else None


def run_cross_validation(run_id: str, training_reports: list[dict[str, Any]], store: RunStore, ml: MLSettings) -> dict[str, Any]:
    cfg, ctx, x_train, y_train, _, _, groups = load_partitions(run_id, store)
    folds, seed, primary = cfg["cv_folds"], cfg["random_state"], cfg["primary_metric"]
    if groups is not None:
        splitter, strategy = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed), "StratifiedGroupKFold (grouped by patient)"
    else:
        splitter, strategy = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), "StratifiedKFold"
    splits = list(splitter.split(x_train, y_train, groups=groups))
    nested = ml.tuning_enabled
    trained = {m["key"]: m for report in training_reports for m in report["models"]}
    ml_log("CV", f"Running {folds}-fold {'nested ' if nested else ''}CV ({strategy}) on {len(x_train)} training rows", run_id=run_id)

    results: dict[str, ModelCVResult] = {}
    for spec in get_specs(None, cfg["models"]):
        info = trained.get(spec.key)
        if info is None:
            continue
        base = {"key": spec.key, "name": spec.name, "family": spec.family}
        if info["status"] != "ok":
            results[spec.key] = ModelCVResult(**base, status="skipped", error=f"not cross-validated: training {info['status']}")
            continue
        try:
            fold_results, chosen = [], []
            for i, (tr, va) in enumerate(splits):
                start = time.perf_counter()
                tuned = fit_tuned(spec, ctx, x_train.iloc[tr], y_train.iloc[tr], groups.iloc[tr] if groups is not None else None,
                                  primary, ml, seed)
                fit_time = time.perf_counter() - start
                validation = compute_metrics(y_train.iloc[va], tuned.model.predict_proba(x_train.iloc[va])[:, 1])
                train = compute_metrics(y_train.iloc[tr], tuned.model.predict_proba(x_train.iloc[tr])[:, 1])
                chosen.append(tuned.best_params)
                fold_results.append(FoldResult(fold=i + 1, train_size=len(tr), validation_size=len(va), validation=validation,
                                               train=train, fit_time_s=round(fit_time, 4), best_params=tuned.best_params))
            primary_scores = [f.validation[primary] for f in fold_results]
            counts = Counter(repr(sorted(p.items())) for p in chosen)
            results[spec.key] = ModelCVResult(
                **base, status="ok", folds=fold_results,
                mean={m: _stat([f.validation[m] for f in fold_results], np.mean) for m in METRIC_NAMES},
                std={m: _stat([f.validation[m] for f in fold_results], np.std) for m in METRIC_NAMES},
                train_mean={m: _stat([f.train[m] for f in fold_results], np.mean) for m in METRIC_NAMES},
                cv_scores=[round(float(s), 4) for s in primary_scores],
                cv_mean=_stat(primary_scores, np.mean), cv_std=_stat(primary_scores, np.std),
                tuning_stability=round(counts.most_common(1)[0][1] / len(chosen), 4) if chosen and chosen[0] else None,
            )
            ml_log("CV", f"{spec.name}: {primary} {np.mean(primary_scores):.3f} ± {np.std(primary_scores):.3f}", run_id=run_id, model=spec.key)
        except Exception as exc:
            ml_log("CV", f"{spec.name} CV failed: {type(exc).__name__}", 30, run_id=run_id, model=spec.key)
            results[spec.key] = ModelCVResult(**base, status="failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")

    report = {"run_id": run_id, "strategy": strategy, "folds": folds, "nested": nested, "inner_folds": ml.tuning_inner_folds if nested else None,
              "primary_metric": primary, "shuffle_seed": seed, "models": {k: r.model_dump() for k, r in results.items()}}
    store.save_json(run_id, "cross_validation", report)
    return report
