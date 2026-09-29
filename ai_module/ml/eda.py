"""Step 2 (CRISP-DM "data understanding"): exploratory data analysis of the analysis dataset.

Descriptive only: nothing here is fitted or used by the models, so looking at the full
dataset does not leak into training. Univariate association with the target uses robust,
non-parametric tests (Mann-Whitney U with rank-biserial effect size for numeric features,
chi-square with Cramer's V for categorical ones), Holm-corrected for multiple testing."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ai_module.ml.log import ml_log
from ai_module.ml.run_store import RunStore
from ai_module.ml.statistics import holm

_BINS = 12


def _r(value: Any, digits: int = 4) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) or np.isinf(v) else round(v, digits)


def _numeric(frame: pd.DataFrame, column: str, y: pd.Series) -> dict[str, Any]:
    s = frame[column]
    values = s.dropna()
    q1, q3 = (values.quantile(0.25), values.quantile(0.75)) if len(values) else (np.nan, np.nan)
    iqr = q3 - q1
    outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()) if len(values) else 0
    pos, neg = s[y == 1].dropna(), s[y == 0].dropna()
    test: dict[str, Any] = {"test": "Mann-Whitney U", "statistic": None, "p_value": None, "effect_size_rank_biserial": None}
    if len(pos) >= 3 and len(neg) >= 3 and values.nunique() > 1:
        u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        test.update(statistic=_r(u), p_value=_r(p, 6), effect_size_rank_biserial=_r(2 * u / (len(pos) * len(neg)) - 1))
    edges = np.histogram_bin_edges(values, bins=_BINS) if values.nunique() > 1 else np.array([values.min() - 0.5, values.max() + 0.5]) if len(values) else np.array([0, 1])
    hist_pos, _ = np.histogram(pos, bins=edges)
    hist_neg, _ = np.histogram(neg, bins=edges)
    return {
        "feature": column, "count": int(values.size), "missing": int(s.isna().sum()), "missing_pct": _r(100 * s.isna().mean(), 2),
        "mean": _r(values.mean()), "std": _r(values.std()), "min": _r(values.min()), "q1": _r(q1), "median": _r(values.median()),
        "q3": _r(q3), "max": _r(values.max()), "skewness": _r(stats.skew(values)) if len(values) > 2 and values.nunique() > 1 else None,
        "outliers_iqr": outliers, "mean_positive": _r(pos.mean()), "mean_negative": _r(neg.mean()),
        "median_positive": _r(pos.median()), "median_negative": _r(neg.median()),
        "association": test,
        "histogram": {"edges": [_r(e) for e in edges], "positive": hist_pos.tolist(), "negative": hist_neg.tolist()},
    }


def _categorical(frame: pd.DataFrame, column: str, y: pd.Series) -> dict[str, Any]:
    s = frame[column].astype(str)
    table = pd.crosstab(s, y)
    levels = [
        {"level": str(level), "count": int(table.loc[level].sum()), "positive_rate": _r(table.loc[level].get(1, 0) / table.loc[level].sum())}
        for level in table.sum(axis=1).sort_values(ascending=False).index
    ]
    test: dict[str, Any] = {"test": "chi-square", "statistic": None, "p_value": None, "dof": None, "cramers_v": None, "low_expected_counts": None}
    if table.shape[0] > 1 and table.shape[1] > 1:
        chi2, p, dof, expected = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n else np.nan
        test.update(statistic=_r(chi2), p_value=_r(p, 6), dof=int(dof), cramers_v=_r(v),
                    low_expected_counts=bool((expected < 5).mean() > 0.2))
    return {"feature": column, "levels": levels[:15], "n_levels": int(table.shape[0]), "missing": int(frame[column].isna().sum()), "association": test}


def run_eda(run_id: str, store: RunStore, alpha: float) -> dict[str, Any]:
    cfg = store.load_json(run_id, "config")
    frame: pd.DataFrame = store.load_object(run_id, "frame")
    target = cfg["target"]
    y = frame[target].astype(int)
    feats = cfg["features"]
    numeric = [_numeric(frame, c, y) for c in feats["numeric"]] + [_numeric(frame, c, y) for c in feats["boolean"]]
    categorical = [_categorical(frame, c, y) for c in feats["categorical"]]

    # Holm correction across every univariate test of this EDA.
    tests = [(item, item["association"]["p_value"]) for item in numeric + categorical if item["association"]["p_value"] is not None]
    for (item, _), adjusted in zip(tests, holm([p for _, p in tests])):
        item["association"]["p_value_holm"] = _r(adjusted, 6)
        item["association"]["significant"] = bool(adjusted < alpha)

    num_cols = [c for c in feats["numeric"] if frame[c].nunique() > 1]
    corr = frame[num_cols + [target]].corr(method="spearman").round(3) if num_cols else pd.DataFrame()
    pairs = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            value = corr.loc[a, b]
            if not pd.isna(value):
                pairs.append({"a": a, "b": b, "spearman": float(value)})
    pairs.sort(key=lambda p: -abs(p["spearman"]))

    counts = y.value_counts().reindex([0, 1], fill_value=0)
    missing = [{"feature": c, "missing": int(frame[c].isna().sum()), "missing_pct": _r(100 * frame[c].isna().mean(), 2)}
               for group in feats.values() for c in group]
    report = {
        "run_id": run_id,
        "overview": {
            "rows": int(len(frame)), "features": sum(len(v) for v in feats.values()), "numeric": len(feats["numeric"]),
            "categorical": len(feats["categorical"]), "boolean": len(feats["boolean"]),
            "duplicated_rows": int(frame.duplicated().sum()), "groups": int(frame[cfg["group_column"]].nunique()) if cfg["group_column"] else None,
            "target": target, "class_counts": {"0": int(counts[0]), "1": int(counts[1])}, "positive_rate": _r(y.mean()),
            "complete_rows_pct": _r(100 * frame[[c for g in feats.values() for c in g]].notna().all(axis=1).mean(), 2),
        },
        "missing": sorted(missing, key=lambda m: -m["missing"]),
        "numeric": numeric,
        "categorical": categorical,
        "correlations": {
            "method": "spearman", "features": num_cols + [target],
            "matrix": corr.where(pd.notna(corr), None).values.tolist() if not corr.empty else [],
            "with_target": {c: _r(corr.loc[c, target], 3) for c in num_cols} if not corr.empty else {},
            "top_pairs": pairs[:8],
        },
        "alpha": alpha,
        "tests_note": "Mann-Whitney U (numeric/boolean) and chi-square (categorical), Holm-corrected; descriptive, not used for training.",
    }
    significant = sum(1 for item in numeric + categorical if item["association"].get("significant"))
    ml_log("EDA", f"{len(numeric) + len(categorical)} features explored, {significant} significantly associated with '{target}'", run_id=run_id)
    store.save_json(run_id, "eda", report)
    return report
