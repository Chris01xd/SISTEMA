"""Training / evaluation logic (pure functions, no I/O). CLIs live in ``train.py`` / ``evaluate.py``.

Two supervised tasks on the same admission-time features:
  * ICU need (binary classification of ``icu_flag``);
  * hospital length of stay in hours (regression, fitted on log1p hours).

Protocol: patient-grouped split train/validation/test = 60/20/20 (an admission's patient
never appears in two splits); model selection on validation; the selected model is refit
on train+validation and scored once on test against a naive baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    median_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_module.digital_twin.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    GROUP_COLUMN,
    NUMERIC_FEATURES,
    TARGET_ICU,
    TARGET_LOS,
    build_features,
)

TEST_SIZE = 0.2
VALIDATION_SIZE_OF_REST = 0.25  # 0.25 x 0.8 = 0.2 of the data


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, dict[str, int]]:
        return {
            name: {"admissions": int(len(part)), "patients": int(part[GROUP_COLUMN].nunique())}
            for name, part in (("train", self.train), ("validation", self.validation), ("test", self.test))
        }


def split_by_patient(df: pd.DataFrame, seed: int) -> Split:
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, groups=df[GROUP_COLUMN]))
    rest, test = df.iloc[rest_idx], df.iloc[test_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE_OF_REST, random_state=seed)
    train_idx, val_idx = next(inner.split(rest, groups=rest[GROUP_COLUMN]))
    return Split(train=rest.iloc[train_idx], validation=rest.iloc[val_idx], test=test)


def split_from_test_subjects(df: pd.DataFrame, test_subjects: list[int]) -> pd.DataFrame:
    """The test partition recorded in a model's metadata (used by ``evaluate``)."""
    return df[df[GROUP_COLUMN].isin(test_subjects)]


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
            ("bool", "passthrough", list(BOOLEAN_FEATURES)),
        ]
    )


def classifier_candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_c0.1": Pipeline([("prep", _preprocessor()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))]),
        "logistic_c1": Pipeline([("prep", _preprocessor()), ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))]),
        "random_forest": Pipeline([
            ("prep", _preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=seed, n_jobs=1)),
        ]),
    }


def _log_target(regressor: Any) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def regressor_candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "ridge_a1": Pipeline([("prep", _preprocessor()), ("reg", _log_target(Ridge(alpha=1.0)))]),
        "ridge_a10": Pipeline([("prep", _preprocessor()), ("reg", _log_target(Ridge(alpha=10.0)))]),
        "random_forest": Pipeline([
            ("prep", _preprocessor()),
            ("reg", _log_target(RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=1))),
        ]),
    }


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, prevalence: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    metrics = {
        "n": float(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "brier_baseline": round(float(brier_score_loss(y_true, np.full(len(y_true), prevalence))), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, proba >= 0.5)), 4),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)
        metrics["average_precision"] = round(float(average_precision_score(y_true, proba)), 4)
    return metrics


def regression_metrics(y_true: np.ndarray, pred: np.ndarray, baseline_value: float) -> dict[str, float]:
    return {
        "n": float(len(y_true)),
        "mae_hours": round(float(mean_absolute_error(y_true, pred)), 2),
        "median_ae_hours": round(float(median_absolute_error(y_true, pred)), 2),
        "mae_hours_baseline_median": round(float(mean_absolute_error(y_true, np.full(len(y_true), baseline_value))), 2),
    }


@dataclass
class TrainingOutcome:
    icu_model: Pipeline
    los_model: Pipeline
    selection: dict[str, dict[str, dict[str, float]]]
    chosen: dict[str, str]
    test_metrics: dict[str, dict[str, float]]
    baselines: dict[str, float]
    split: Split


def train_models(df: pd.DataFrame, seed: int) -> TrainingOutcome:
    """Full protocol: split -> select on validation -> refit on train+val -> score on test."""
    split = split_by_patient(df, seed)
    x_train, x_val, x_test = (build_features(p) for p in (split.train, split.validation, split.test))
    prevalence = float(split.train[TARGET_ICU].mean())
    median_los = float(split.train[TARGET_LOS].median())

    selection: dict[str, dict[str, dict[str, float]]] = {"icu_classifier": {}, "los_regressor": {}}
    for name, model in classifier_candidates(seed).items():
        fitted = clone(model).fit(x_train, split.train[TARGET_ICU].astype(int))
        proba = fitted.predict_proba(x_val)[:, 1]
        selection["icu_classifier"][name] = classification_metrics(split.validation[TARGET_ICU], proba, prevalence)
    for name, model in regressor_candidates(seed).items():
        fitted = clone(model).fit(x_train, split.train[TARGET_LOS])
        selection["los_regressor"][name] = regression_metrics(split.validation[TARGET_LOS], fitted.predict(x_val), median_los)

    best_clf = max(selection["icu_classifier"], key=lambda n: selection["icu_classifier"][n].get("roc_auc", -selection["icu_classifier"][n]["brier"]))
    best_reg = min(selection["los_regressor"], key=lambda n: selection["los_regressor"][n]["mae_hours"])

    train_val = pd.concat([split.train, split.validation])
    x_train_val = build_features(train_val)
    prevalence_tv = float(train_val[TARGET_ICU].mean())
    median_tv = float(train_val[TARGET_LOS].median())
    icu_model = clone(classifier_candidates(seed)[best_clf]).fit(x_train_val, train_val[TARGET_ICU].astype(int))
    los_model = clone(regressor_candidates(seed)[best_reg]).fit(x_train_val, train_val[TARGET_LOS])

    test_metrics = {
        "icu_classifier": classification_metrics(split.test[TARGET_ICU], icu_model.predict_proba(x_test)[:, 1], prevalence_tv),
        "los_regressor": regression_metrics(split.test[TARGET_LOS], los_model.predict(x_test), median_tv),
    }
    return TrainingOutcome(
        icu_model=icu_model,
        los_model=los_model,
        selection=selection,
        chosen={"icu_classifier": best_clf, "los_regressor": best_reg},
        test_metrics=test_metrics,
        baselines={"icu_prevalence_train_val": round(prevalence_tv, 4), "los_median_hours_train_val": round(median_tv, 2)},
        split=split,
    )


def evaluate_models(icu_model: Pipeline, los_model: Pipeline, test: pd.DataFrame, baselines: dict[str, float],
                    bootstrap: int = 1000, seed: int = 0) -> dict[str, dict[str, float]]:
    """Test metrics plus bootstrap 95% intervals (the test set is small: intervals matter)."""
    x_test = build_features(test)
    proba = icu_model.predict_proba(x_test)[:, 1]
    pred = los_model.predict(x_test)
    y_icu = test[TARGET_ICU].astype(int).to_numpy()
    y_los = test[TARGET_LOS].to_numpy()
    result = {
        "icu_classifier": classification_metrics(y_icu, proba, baselines["icu_prevalence_train_val"]),
        "los_regressor": regression_metrics(y_los, pred, baselines["los_median_hours_train_val"]),
    }
    rng = np.random.default_rng(seed)
    aucs, maes = [], []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(test), len(test))
        if len(np.unique(y_icu[idx])) == 2:
            aucs.append(roc_auc_score(y_icu[idx], proba[idx]))
        maes.append(mean_absolute_error(y_los[idx], pred[idx]))
    if aucs:
        result["icu_classifier"]["roc_auc_ci95_low"] = round(float(np.quantile(aucs, 0.025)), 4)
        result["icu_classifier"]["roc_auc_ci95_high"] = round(float(np.quantile(aucs, 0.975)), 4)
    result["los_regressor"]["mae_hours_ci95_low"] = round(float(np.quantile(maes, 0.025)), 2)
    result["los_regressor"]["mae_hours_ci95_high"] = round(float(np.quantile(maes, 0.975)), 2)
    return result
