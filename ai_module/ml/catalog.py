"""Model catalog: 5 normal models + 2 hybrid ensembles for binary classification.

Every model is a ``Pipeline(preprocessor, estimator)`` so imputation, scaling and encoding
are fitted only on the data each fit sees (training partition, or training folds in CV):
no information from validation/test rows leaks into preprocessing.

Choice rationale (small tabular clinical data, mixed numeric/categorical, ~46% positives):
  normal  - Logistic Regression: calibrated linear baseline, interpretable.
          - SVM (RBF): non-linear margin method, strong on small scaled data.
          - Random Forest: bagged trees, robust to noise and interactions.
          - Gradient Boosting: sequential boosted trees (scikit-learn).
          - XGBoost: regularized boosting (optional dependency; skipped if absent).
  hybrid  - Soft Voting: averages probabilities of LR + RF + XGBoost (GB if no XGBoost),
            mixing a linear, a bagging and a boosting learner.
          - Stacking: RF + GB + SVM produce out-of-fold probabilities; a Logistic Regression
            meta-model learns how to combine them.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field

from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from ai_module.ml.schemas import ModelFamily

XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None


@dataclass(frozen=True)
class FeatureSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    boolean: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.boolean]


@dataclass(frozen=True)
class BuildContext:
    features: FeatureSpec
    random_state: int
    imbalanced: bool


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    family: ModelFamily
    description: str
    build: Callable[[BuildContext], ClassifierMixin]
    components: tuple[str, ...] = field(default_factory=tuple)
    requires: str | None = None  # optional import needed
    # Hyper-parameter grid searched with inner grouped CV (keys use Pipeline step names).
    param_grid: dict[str, list[Any]] = field(default_factory=dict)

    def available(self) -> bool:
        return self.requires is None or importlib.util.find_spec(self.requires) is not None


def build_preprocessor(features: FeatureSpec) -> ColumnTransformer:
    transformers = []
    if features.numeric:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                               ("scale", StandardScaler())]), list(features.numeric)))
    if features.categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False),
                             list(features.categorical)))
    if features.boolean:
        transformers.append(("bool", "passthrough", list(features.boolean)))
    return ColumnTransformer(transformers)


def _pipeline(ctx: BuildContext, estimator: ClassifierMixin) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(ctx.features)), ("model", estimator)])


def _class_weight(ctx: BuildContext) -> str | None:
    return "balanced" if ctx.imbalanced else None


# ------------------------------------------------------------------ base estimators


def _logistic(ctx: BuildContext) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=5000, class_weight=_class_weight(ctx))


def _svm(ctx: BuildContext) -> CalibratedClassifierCV:
    # Sigmoid (Platt) calibration gives the probabilities ROC/PR-AUC and soft voting need.
    return CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", class_weight=_class_weight(ctx), random_state=ctx.random_state),
                                  method="sigmoid", cv=5, ensemble=False)


def _random_forest(ctx: BuildContext) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features="sqrt",
                                  class_weight="balanced_subsample" if ctx.imbalanced else None,
                                  random_state=ctx.random_state, n_jobs=1)


def _gradient_boosting(ctx: BuildContext) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8,
                                      random_state=ctx.random_state)


def _xgboost(ctx: BuildContext) -> ClassifierMixin:
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=1.0, eval_metric="logloss", random_state=ctx.random_state, n_jobs=1, verbosity=0)


def _boosting_member(ctx: BuildContext) -> tuple[str, ClassifierMixin]:
    return ("xgboost", _xgboost(ctx)) if XGBOOST_AVAILABLE else ("gradient_boosting", _gradient_boosting(ctx))


def _voting(ctx: BuildContext) -> VotingClassifier:
    return VotingClassifier([("logistic_regression", _logistic(ctx)), ("random_forest", _random_forest(ctx)), _boosting_member(ctx)],
                            voting="soft")


def _stacking(ctx: BuildContext) -> StackingClassifier:
    return StackingClassifier(
        estimators=[("random_forest", _random_forest(ctx)), ("gradient_boosting", _gradient_boosting(ctx)), ("svm", _svm(ctx))],
        final_estimator=LogisticRegression(max_iter=5000, class_weight=_class_weight(ctx)),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=ctx.random_state),
        n_jobs=1,
    )


def _wrap(factory: Callable[[BuildContext], ClassifierMixin]) -> Callable[[BuildContext], Pipeline]:
    return lambda ctx: _pipeline(ctx, factory(ctx))


_BOOST = "XGBoost" if XGBOOST_AVAILABLE else "Gradient Boosting"

CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("logistic_regression", "Logistic Regression", "normal", "L2-regularized linear model.", _wrap(_logistic),
              param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]}),
    ModelSpec("svm", "SVM (RBF)", "normal", "Support vector machine with RBF kernel; Platt-calibrated probabilities.", _wrap(_svm),
              param_grid={"model__estimator__C": [0.3, 1.0, 3.0], "model__estimator__gamma": ["scale", 0.03]}),
    ModelSpec("random_forest", "Random Forest", "normal", "200 bagged decision trees.", _wrap(_random_forest),
              param_grid={"model__max_depth": [None, 5], "model__min_samples_leaf": [1, 5]}),
    ModelSpec("gradient_boosting", "Gradient Boosting", "normal", "150 shallow boosted trees (scikit-learn).", _wrap(_gradient_boosting),
              param_grid={"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}),
    ModelSpec("xgboost", "XGBoost", "normal", "Regularized gradient boosting (xgboost).", _wrap(_xgboost), requires="xgboost",
              param_grid={"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 4]}),
    ModelSpec("voting_ensemble", "Voting Ensemble", "hybrid", "Soft voting: average of the members' probabilities.", _wrap(_voting),
              components=("Logistic Regression", "Random Forest", _BOOST),
              param_grid={"model__weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]}),
    ModelSpec("stacking_ensemble", "Stacking Ensemble", "hybrid",
              "Out-of-fold member probabilities combined by a Logistic Regression meta-model.", _wrap(_stacking),
              components=("Random Forest", "Gradient Boosting", "SVM (RBF)", "meta: Logistic Regression"),
              param_grid={"model__final_estimator__C": [0.1, 1.0]}),
)

MODEL_KEYS = tuple(spec.key for spec in CATALOG)


def get_specs(family: ModelFamily | None = None, keys: list[str] | None = None) -> list[ModelSpec]:
    return [s for s in CATALOG if (family is None or s.family == family) and (keys is None or s.key in keys)]
