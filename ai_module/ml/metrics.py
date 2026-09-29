"""The single metric definition used for every model, fold and partition."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")

# scikit-learn scorer names for cross_validate (same definitions as `compute_metrics`).
CV_SCORERS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if both_classes else None,
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4) if both_classes else None,
    }


def confusion(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> list[list[int]]:
    return confusion_matrix(np.asarray(y_true).astype(int), (proba >= threshold).astype(int), labels=[0, 1]).tolist()
