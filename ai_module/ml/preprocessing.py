"""Step 2: preprocessing plan, feature engineering description and the hold-out split.

Row-wise, stateless feature engineering (hour of day -> sin/cos, booleans -> 0/1) is done
when the dataset is loaded. Everything that *learns* from data (median imputation,
missing-value indicators, scaling, one-hot categories) lives inside each model's Pipeline,
so it is fitted on training rows only, and on the training folds only during CV."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from ai_module.ml.catalog import BuildContext, FeatureSpec
from ai_module.ml.log import ml_log
from ai_module.ml.run_store import RunStore

LEAKAGE_CONTROLS = [
    "Imputation, missing indicators, scaling and one-hot encoding are Pipeline steps fitted on training data only.",
    "Cross-validation refits the whole Pipeline inside every fold (no preprocessing sees validation folds).",
    "Hold-out and CV folds are grouped by patient: no patient appears in two partitions.",
    "Only admission-time features are used; outcomes (diagnoses, discharge data, length of stay) are excluded.",
    "SMOTE/oversampling is not used; imbalance is handled with class weights inside the estimators.",
]
ENGINEERED_FEATURES = {
    "admit_hour_sin / admit_hour_cos": "cyclical encoding of admission hour (stateless, row-wise)",
    "*_missing indicators": "added by SimpleImputer(add_indicator=True) inside the Pipeline",
}


def make_split(frame: pd.DataFrame, target: str, group_column: str | None, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    y = frame[target].astype(int).to_numpy()
    if group_column:
        n_splits = max(2, round(1 / test_size))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(frame, y, groups=frame[group_column]))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(frame, y))
    return np.sort(train_idx), np.sort(test_idx)


def preprocess(run_id: str, store: RunStore) -> dict[str, Any]:
    cfg = store.load_json(run_id, "config")
    frame: pd.DataFrame = store.load_object(run_id, "frame")
    target, group = cfg["target"], cfg["group_column"]
    train_idx, test_idx = make_split(frame, target, group, cfg["test_size"], cfg["random_state"])
    store.save_object(run_id, "split", {"train": train_idx, "test": test_idx})

    y = frame[target].astype(int)
    overlap = len(set(frame.loc[train_idx, group]) & set(frame.loc[test_idx, group])) if group else 0
    report = {
        "run_id": run_id,
        "features": cfg["features"],
        "n_features_in": sum(len(v) for v in cfg["features"].values()),
        "engineered_features": ENGINEERED_FEATURES,
        "pipeline": {
            "numeric": "SimpleImputer(median, add_indicator) -> StandardScaler",
            "categorical": "OneHotEncoder(handle_unknown=infrequent_if_exist, min_frequency=5)",
            "boolean": "passthrough",
        },
        "split": {
            "method": "StratifiedGroupKFold hold-out (grouped by patient)" if group else "StratifiedShuffleSplit",
            "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
            "train_positive_rate": round(float(y.iloc[train_idx].mean()), 4),
            "test_positive_rate": round(float(y.iloc[test_idx].mean()), 4),
            "group_overlap": overlap,
        },
        "leakage_controls": LEAKAGE_CONTROLS,
    }
    ml_log("PREPROCESSING", f"Hold-out split {len(train_idx)} train / {len(test_idx)} test rows, group overlap {overlap}", run_id=run_id)
    return report


def load_partitions(run_id: str, store: RunStore) -> tuple[dict[str, Any], BuildContext, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series | None]:
    """Config, build context and (X_train, y_train, X_test, y_test, groups_train) of a run."""
    cfg = store.load_json(run_id, "config")
    frame: pd.DataFrame = store.load_object(run_id, "frame")
    split = store.load_object(run_id, "split")
    features = FeatureSpec(tuple(cfg["features"]["numeric"]), tuple(cfg["features"]["categorical"]), tuple(cfg["features"]["boolean"]))
    ctx = BuildContext(features=features, random_state=cfg["random_state"], imbalanced=cfg["imbalanced"])
    train, test = frame.iloc[split["train"]], frame.iloc[split["test"]]
    target, group = cfg["target"], cfg["group_column"]
    return (cfg, ctx, train[features.columns], train[target].astype(int), test[features.columns], test[target].astype(int),
            train[group] if group else None)
