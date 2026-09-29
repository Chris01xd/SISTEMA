"""Step 9: inferential statistics comparing the models (robust to the small sample).

On the outer CV folds:
  * Friedman test (all models, non-parametric) + Nemenyi critical difference of mean ranks;
  * corrected resampled paired t-test (Nadeau & Bengio, 2003), best vs each other model.
    It corrects the variance for the overlap between CV training sets, which the naive
    paired t-test ignores (and which makes it far too optimistic).
On the untouched hold-out:
  * DeLong test for the difference of two correlated ROC-AUCs (Sun & Xu fast algorithm);
  * exact McNemar test on the 0.5-threshold predictions;
  * percentile bootstrap 95% confidence intervals of ROC-AUC and F1 for every model.
Pairwise p-values are Holm-corrected within each test family. These tests *qualify* the
selection (how much evidence there is that the winner is really better); the winner itself
still comes from the configured final_score."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

from ai_module.ml.log import ml_log
from ai_module.ml.run_store import RunStore


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def _r(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    v = float(value)
    return None if np.isnan(v) or np.isinf(v) else round(v, digits)


def nadeau_bengio(a: list[float], b: list[float], n_train: float, n_test: float) -> dict[str, Any]:
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1) if k > 1 else 0.0
    if var == 0:
        return {"mean_difference": _r(d.mean()), "t": None, "p_value": 1.0 if d.mean() == 0 else None, "df": k - 1}
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_difference": _r(d.mean()), "t": _r(t), "p_value": _r(p, 6), "df": k - 1}


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, Any]:
    """Two-sided DeLong test for AUC(p1) - AUC(p2) on the same samples."""
    y_true = np.asarray(y_true).astype(int)
    pos, neg = y_true == 1, y_true == 0
    m, n = pos.sum(), neg.sum()
    if m < 2 or n < 2:
        return {"auc_a": None, "auc_b": None, "z": None, "p_value": None}
    aucs, v10, v01 = [], [], []
    for scores in (np.asarray(p1), np.asarray(p2)):
        x, yv = scores[pos], scores[neg]
        tx, ty, tz = _midrank(x), _midrank(yv), _midrank(np.concatenate([x, yv]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1.0 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    if var <= 0:
        return {"auc_a": _r(aucs[0]), "auc_b": _r(aucs[1]), "z": None, "p_value": 1.0}
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {"auc_a": _r(aucs[0]), "auc_b": _r(aucs[1]), "z": _r(z), "p_value": _r(2 * stats.norm.sf(abs(z)), 6)}


def mcnemar(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    c1, c2 = (np.asarray(p1) >= threshold).astype(int) == y, (np.asarray(p2) >= threshold).astype(int) == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    p = 1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": _r(p, 6)}


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, resamples: int, seed: int) -> dict[str, list[float | None]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    aucs, f1s = [], []
    for _ in range(resamples):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], proba[idx]))
        f1s.append(f1_score(y[idx], proba[idx] >= 0.5, zero_division=0))
    q = lambda v: [_r(np.quantile(v, 0.025)), _r(np.quantile(v, 0.975))] if v else [None, None]  # noqa: E731
    return {"roc_auc": q(aucs), "f1": q(f1s)}


def statistical_tests(run_id: str, evaluation: dict[str, Any], cv_report: dict[str, Any], comparison: dict[str, Any],
                      store: RunStore, alpha: float, resamples: int, seed: int) -> dict[str, Any]:
    ml_log("STATS", "Running inferential tests (Friedman/Nemenyi, Nadeau-Bengio, DeLong, McNemar, bootstrap)", run_id=run_id)
    ranking = comparison["ranking"]
    keys = [e["key"] for e in ranking]
    names = {e["key"]: e["model"] for e in ranking}
    best = keys[0]
    fold_scores = {k: cv_report["models"][k]["cv_scores"] for k in keys}
    folds = cv_report["models"][best]["folds"]
    n_train = float(np.mean([f["train_size"] for f in folds]))
    n_test = float(np.mean([f["validation_size"] for f in folds]))

    # --- Friedman + Nemenyi on the outer folds --------------------------------------
    matrix = np.array([fold_scores[k] for k in keys])  # models x folds
    k_models, n_folds = matrix.shape
    friedman: dict[str, Any] = {"statistic": None, "p_value": None, "significant": None}
    if k_models >= 3:
        stat, p = stats.friedmanchisquare(*matrix)
        friedman = {"statistic": _r(stat), "p_value": _r(p, 6), "significant": bool(p < alpha)}
    ranks = np.vstack([stats.rankdata(-matrix[:, j]) for j in range(n_folds)]).T  # rank 1 = best per fold
    mean_ranks = {names[k]: _r(r) for k, r in zip(keys, ranks.mean(axis=1))}
    q_alpha = stats.studentized_range.ppf(1 - alpha, k_models, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k_models * (k_models + 1) / (6 * n_folds))
    best_rank = ranks.mean(axis=1)[0]
    nemenyi = {
        "critical_difference": _r(cd), "mean_ranks": mean_ranks,
        "different_from_best": [names[k] for k, r in zip(keys[1:], ranks.mean(axis=1)[1:]) if r - best_rank > cd],
    }

    # --- Pairwise best vs each other ----------------------------------------------------
    y_test = store.load_object(run_id, "predictions/y_test")
    proba = {k: store.load_object(run_id, f"predictions/{k}") for k in keys}
    pairwise = []
    for other in keys[1:]:
        pairwise.append({
            "model": names[other],
            "nadeau_bengio": nadeau_bengio(fold_scores[best], fold_scores[other], n_train, n_test),
            "delong": delong(y_test, proba[best], proba[other]),
            "mcnemar": mcnemar(y_test, proba[best], proba[other]),
        })
    for test in ("nadeau_bengio", "delong", "mcnemar"):
        present = [(row, row[test]["p_value"]) for row in pairwise if row[test]["p_value"] is not None]
        for (row, _), adjusted in zip(present, holm([p for _, p in present])):
            row[test]["p_value_holm"] = _r(adjusted, 6)
            row[test]["significant"] = bool(adjusted < alpha)

    ci = {names[k]: bootstrap_ci(y_test, proba[k], resamples, seed) for k in keys}
    beaten = sorted({row["model"] for row in pairwise for t in ("nadeau_bengio", "delong", "mcnemar") if row[t].get("significant")})
    report = {
        "run_id": run_id, "alpha": alpha, "bootstrap_resamples": resamples, "best_model": names[best], "primary_metric": cv_report["primary_metric"],
        "cv_folds": n_folds, "holdout_size": int(len(y_test)),
        "friedman": friedman, "nemenyi": nemenyi, "pairwise_vs_best": pairwise, "bootstrap_ci_95": ci,
        "summary": {
            "models_significantly_worse_than_best": beaten,
            "evidence": "strong" if len(beaten) == len(pairwise) and pairwise else "partial" if beaten else "weak",
        },
    }
    store.save_json(run_id, "statistics", report)
    ml_log("STATS", f"Friedman p={friedman['p_value']}; best significantly better than {len(beaten)}/{len(pairwise)} models", run_id=run_id)
    return report
