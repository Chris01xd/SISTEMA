"""Step 1: request resolution, dataset loading and input validation."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ai_module.config.settings import AISettings
from ai_module.digital_twin.errors import TwinDataNotPreparedError
from ai_module.ml.catalog import MODEL_KEYS
from ai_module.ml.config import MLSettings
from ai_module.ml.datasets import get_dataset
from ai_module.ml.errors import (
    DatasetUnavailableError,
    EmptyDatasetError,
    InsufficientDataError,
    InvalidColumnsError,
    InvalidRequestError,
    TargetNotFoundError,
)
from ai_module.ml.log import ml_log
from ai_module.ml.run_store import RunStore
from ai_module.ml.schemas import AnalysisRequest


def validate_input(request: AnalysisRequest, ai_settings: AISettings, ml: MLSettings, store: RunStore) -> dict[str, Any]:
    """Validates the request against the dataset, creates the run and stores the frame.
    Raises a specific ``MLAnalysisError`` for every invalid input."""
    spec = get_dataset(request.dataset)
    target = request.target or spec.default_target
    if target not in spec.allowed_targets:
        raise TargetNotFoundError(f"Target '{target}' does not exist in dataset '{spec.key}'. Allowed: {list(spec.allowed_targets)}")
    features = request.features or list(spec.features)
    invalid = sorted(set(features) - set(spec.features))
    if invalid or not features:
        raise InvalidColumnsError(f"Invalid feature columns {invalid or '[]'}. Allowed: {list(spec.features)}")
    if target in features:
        raise InvalidColumnsError(f"The target '{target}' cannot also be a feature")
    unknown_models = sorted(set(request.models or []) - set(MODEL_KEYS))
    if unknown_models:
        raise InvalidRequestError(f"Unknown models {unknown_models}. Available: {list(MODEL_KEYS)}")

    cv_folds = request.cv_folds or ml.cv_folds
    test_size = request.test_size or ml.test_size
    random_state = ml.random_state if request.random_state is None else request.random_state

    try:
        frame = spec.loader(ai_settings)
    except TwinDataNotPreparedError as exc:
        raise DatasetUnavailableError(str(exc)) from exc
    if frame.empty:
        raise EmptyDatasetError(f"Dataset '{spec.key}' is empty")
    if target not in frame.columns:
        raise TargetNotFoundError(f"Target column '{target}' not found in the loaded data")
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise InvalidColumnsError(f"Columns missing from the loaded data: {missing}")

    frame = frame.dropna(subset=[target])
    y = frame[target]
    if not set(pd.unique(y.astype(int))) <= {0, 1}:
        raise InvalidRequestError(f"Target '{target}' is not binary; only binary classification is supported")
    counts = y.astype(int).value_counts().reindex([0, 1], fill_value=0)
    minority = int(counts.min())
    if len(frame) < ml.min_rows:
        raise InsufficientDataError(f"{len(frame)} rows available; at least {ml.min_rows} are required")
    # Every CV training fold and the hold-out need examples of both classes.
    if minority < 2 * cv_folds:
        raise InsufficientDataError(f"Only {minority} examples of the minority class; at least {2 * cv_folds} are needed for {cv_folds}-fold CV")
    if spec.group_column and frame[spec.group_column].nunique() < 2 * cv_folds:
        raise InsufficientDataError(f"Only {frame[spec.group_column].nunique()} distinct groups; at least {2 * cv_folds} are needed")

    minority_share = minority / len(frame)
    imbalanced = minority_share < ml.imbalance_threshold
    feature_set = set(features)
    resolved = {
        "dataset": spec.key, "dataset_description": spec.description, "source": spec.source, "target": target,
        "problem_type": request.problem_type, "group_column": spec.group_column,
        "features": {"numeric": [c for c in spec.numeric if c in feature_set], "categorical": [c for c in spec.categorical if c in feature_set],
                     "boolean": [c for c in spec.boolean if c in feature_set]},
        "cv_folds": cv_folds, "test_size": test_size, "random_state": random_state,
        "models": request.models, "imbalanced": imbalanced, "primary_metric": "pr_auc" if imbalanced else "roc_auc",
        "explain_with_llm": ml.explain_with_llm if request.explain_with_llm is None else request.explain_with_llm,
    }
    run_id = store.create()
    columns = features + [target] + ([spec.group_column] if spec.group_column else [])
    store.save_object(run_id, "frame", frame[columns].reset_index(drop=True))
    store.save_json(run_id, "config", resolved)

    warnings = []
    if imbalanced:
        warnings.append(f"Imbalanced target ({minority_share:.1%} minority): accuracy is excluded from the score, "
                        "class weights are enabled and PR-AUC is the primary metric.")
    report = {
        "run_id": run_id, "status": "valid", **resolved, "rows": int(len(frame)),
        "class_counts": {"0": int(counts[0]), "1": int(counts[1])}, "minority_share": round(minority_share, 4),
        "groups": int(frame[spec.group_column].nunique()) if spec.group_column else None, "warnings": warnings,
    }
    ml_log("VALIDATION", f"Dataset '{spec.key}' valid: {len(frame)} rows, target '{target}', minority share {minority_share:.1%}",
           run_id=run_id, imbalanced=imbalanced)
    return report
