"""Saturation / bed-crisis classifier.

Justification (docs/ARCHITECTURE.md section 7): RandomForest is the deployed
model because saturation is driven by threshold-like interactions between
occupancy, staffing and arrival rate that tree ensembles capture without
feature scaling and with robustness to MCE-regime outliers. SVM is kept only
as an ablation baseline for the paper's comparison table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

FEATURE_NAMES = [
    "occupancy_ratio",
    "queue_length_norm",
    "staff_effective_ratio",
    "arrival_rate_norm",
    "trend_saturation_delta",
]


class SaturationClassifier:
    """Binary classifier: will saturation cross into RED within `horizon_ticks`?"""

    def __init__(self, algorithm: str = "random_forest"):
        self.algorithm = algorithm
        if algorithm == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
            )
        elif algorithm == "svm":
            self.model = SVC(kernel="rbf", probability=True, class_weight="balanced")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> dict[str, float] | None:
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(FEATURE_NAMES, self.model.feature_importances_.tolist()))
        return None

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        import joblib

        self.model = joblib.load(path)


def features_from_sd_history(history: list[dict]) -> np.ndarray:
    """Builds classifier feature vectors from SystemDynamicsModel.state.history rows."""
    feats = []
    prev_sat = history[0]["saturation_index"] if history else 0.0
    for row in history:
        occupancy_ratio = row["beds_occupied"] / max(row["beds_capacity"], 1)
        queue_norm = min(1.0, row["patients_waiting"] / 30.0)
        staff_ratio = 1.0 - row["staff_fatigue"]
        arrival_norm = 0.0  # populated by caller when arrival series is available
        trend = row["saturation_index"] - prev_sat
        prev_sat = row["saturation_index"]
        feats.append([occupancy_ratio, queue_norm, staff_ratio, arrival_norm, trend])
    return np.array(feats)
