"""Step 6: cross-referencing of results, so no model wins on a single number.

Per model it crosses train score, CV mean/std, per-fold scores and hold-out score to judge
stability, generalization, overfitting and fold consistency. Across models it checks
whether CV and hold-out agree on the ordering (Spearman rank correlation) and which model
leads each metric."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ai_module.ml.config import MLSettings
from ai_module.ml.log import ml_log

_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def _per_model(row: dict[str, Any], folds: list[float], ml: MLSettings) -> dict[str, Any]:
    cv_mean, cv_std, test = row["cv_mean"], row["cv_std"], row["test_score"]
    holdout_gap, cv_gap = row["overfitting_gap"], row["cv_overfitting_gap"]
    cv_vs_test = round(test - cv_mean, 4)
    worst_gap = max(g for g in (holdout_gap, cv_gap) if g is not None)
    flags, codes = [], []
    overfitting = worst_gap > ml.overfitting_gap_threshold
    unstable = cv_std > ml.instability_std_threshold
    inconsistent = abs(cv_vs_test) > ml.cv_holdout_gap_threshold
    if overfitting:
        codes.append("overfitting")
        flags.append(f"overfitting: train-validation gap {worst_gap:.3f} > {ml.overfitting_gap_threshold}")
    if unstable:
        codes.append("unstable")
        flags.append(f"unstable across folds: std {cv_std:.3f} > {ml.instability_std_threshold}")
    if inconsistent:
        codes.append("inconsistent")
        flags.append(f"CV and hold-out disagree: difference {cv_vs_test:+.3f} (threshold ±{ml.cv_holdout_gap_threshold})")
    return {
        "model": row["model"], "type": row["type"],
        "train_score": row["train_score"], "cv_train_mean": row["cv_train_mean"], "cv_mean": cv_mean, "cv_std": cv_std,
        "test_score": test, "fold_scores": folds,
        "fold_range": round(max(folds) - min(folds), 4) if folds else None,
        "coefficient_of_variation": round(cv_std / cv_mean, 4) if cv_mean else None,
        "overfitting_gap": holdout_gap, "cv_overfitting_gap": cv_gap, "cv_vs_test_difference": cv_vs_test,
        "stability": "unstable" if unstable else "stable",
        "generalization": "inconsistent" if inconsistent else "consistent",
        "overfitting": overfitting, "flags": flags, "flag_codes": codes,
    }


def cross_reference(run_id: str, evaluation: dict[str, Any], cv_report: dict[str, Any], ml: MLSettings) -> dict[str, Any]:
    ml_log("CROSS-REFERENCE", "Comparing CV/Test/Train results", run_id=run_id)
    rows = [r for r in evaluation["models"] if r["status"] == "ok"]
    per_model = {r["key"]: _per_model(r, cv_report["models"][r["key"]]["cv_scores"], ml) for r in rows}

    frame = pd.DataFrame(rows).set_index("key") if rows else pd.DataFrame()
    agreement: dict[str, Any] = {}
    if len(rows) >= 2:
        # Spearman = Pearson on ranks (no SciPy dependency).
        rho = frame["cv_mean"].rank().corr(frame["test_score"].rank())
        cv_best, test_best = frame["cv_mean"].idxmax(), frame["test_score"].idxmax()
        agreement = {
            "cv_vs_holdout_rank_spearman": None if pd.isna(rho) else round(float(rho), 4),
            "cv_best": frame.loc[cv_best, "model"], "holdout_best": frame.loc[test_best, "model"],
            "cv_and_holdout_agree_on_best": bool(cv_best == test_best),
        }
    leaders = {}
    for metric in _METRICS:
        cv_values = frame["cv_metrics"].map(lambda m, k=metric: m.get(k)) if rows else pd.Series(dtype=float)
        test_values = frame[metric] if rows else pd.Series(dtype=float)
        leaders[metric] = {
            "cv_leader": frame.loc[cv_values.astype(float).idxmax(), "model"] if cv_values.notna().any() else None,
            "holdout_leader": frame.loc[test_values.astype(float).idxmax(), "model"] if test_values.notna().any() else None,
        }
    best_normal = max((r for r in rows if r["type"] == "normal"), key=lambda r: r["cv_mean"], default=None)
    best_hybrid = max((r for r in rows if r["type"] == "hybrid"), key=lambda r: r["cv_mean"], default=None)
    report = {
        "run_id": run_id, "primary_metric": evaluation["primary_metric"],
        "thresholds": {"overfitting_gap": ml.overfitting_gap_threshold, "instability_std": ml.instability_std_threshold,
                       "cv_holdout_gap": ml.cv_holdout_gap_threshold},
        "models": per_model,
        "cross_model": {
            **agreement, "metric_leaders": leaders,
            "best_normal_by_cv": best_normal and {"model": best_normal["model"], "cv_mean": best_normal["cv_mean"]},
            "best_hybrid_by_cv": best_hybrid and {"model": best_hybrid["model"], "cv_mean": best_hybrid["cv_mean"]},
            "hybrid_gain_over_best_normal": round(best_hybrid["cv_mean"] - best_normal["cv_mean"], 4) if best_normal and best_hybrid else None,
        },
    }
    return report
