"""Pure pandas analytics over the canonical admissions table (no LangChain here).

Everything the LLM can ask for goes through this file, so the surface is small and
auditable: columns, operators, metrics and groupings are whitelisted, nothing is
evaluated as code, and only aggregates are returned (never individual patient rows)."""
from __future__ import annotations

import operator
from typing import Any

import pandas as pd

FILTERABLE_COLUMNS = ("age", "sex", "diagnosis", "severity_level", "los_hours", "icu_flag", "outcome", "admission_hour")
GROUP_BY = ("sex", "severity_level", "icu_flag", "outcome", "admission_hour", "age_band")
METRICS = ("count", "mean_age", "mean_los_hours", "median_los_hours", "icu_rate_pct", "mortality_rate_pct")

_OPS = {"eq": operator.eq, "ne": operator.ne, "gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le}
OPERATORS = tuple(_OPS) + ("contains", "in")

_AGE_BANDS = [0, 18, 40, 65, 80, 200]
_AGE_LABELS = ["0-17", "18-39", "40-64", "65-79", "80+"]


class QueryError(ValueError):
    """Invalid request from the model; the message is written to be fed back to it."""


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived columns the tools expose (kept out of the source file)."""
    out = df.copy()
    out["admission_hour"] = out["admission_time"].dt.hour
    out["age_band"] = pd.cut(out["age"], bins=_AGE_BANDS, labels=_AGE_LABELS, right=False).astype(str)
    return out


def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]] | None) -> pd.DataFrame:
    for f in filters or []:
        column, op, value = f.get("column"), f.get("op"), f.get("value")
        if column not in FILTERABLE_COLUMNS:
            raise QueryError(f"Cannot filter on '{column}'. Allowed columns: {list(FILTERABLE_COLUMNS)}")
        if op not in OPERATORS:
            raise QueryError(f"Unknown operator '{op}'. Allowed: {list(OPERATORS)}")
        series = df[column]
        try:
            if op == "contains":
                mask = series.astype(str).str.contains(str(value), case=False, regex=False, na=False)
            elif op == "in":
                if not isinstance(value, list):
                    raise QueryError("Operator 'in' needs a list value")
                mask = series.isin(value)
            else:
                mask = _OPS[op](series, value)
        except TypeError as exc:
            raise QueryError(f"Cannot compare column '{column}' with {value!r}: {exc}") from exc
        df = df[mask.fillna(False)]
    return df


def _metric(frame: pd.DataFrame, metric: str) -> float | int | None:
    if metric == "count":
        return int(len(frame))
    if frame.empty:
        return None
    if metric == "mean_age":
        return round(float(frame["age"].mean()), 1)
    if metric == "mean_los_hours":
        return round(float(frame["los_hours"].mean()), 1)
    if metric == "median_los_hours":
        return round(float(frame["los_hours"].median()), 1)
    if metric == "icu_rate_pct":
        return round(100 * float(frame["icu_flag"].astype(bool).mean()), 1)
    return round(100 * float((frame["outcome"] == "DECEASED").mean()), 1)  # mortality_rate_pct


def overview(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "admissions": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "mean_age": _metric(df, "mean_age"),
        "median_los_hours": _metric(df, "median_los_hours"),
        "icu_rate_pct": _metric(df, "icu_rate_pct"),
        "mortality_rate_pct": _metric(df, "mortality_rate_pct"),
        "severity_available_for_pct": round(100 * float(df["severity_level"].notna().mean()), 1),
        "columns_you_can_filter": list(FILTERABLE_COLUMNS),
        "groupings": list(GROUP_BY),
        "metrics": list(METRICS),
        "notes": [
            "severity_level: 1=minor, 2=delayed, 3=immediate, 4=critical; only known for admissions that came through the emergency department.",
            "Dates are shifted by the data provider for de-identification: durations and hour of day are valid, calendar dates are not.",
        ],
    }


def aggregate(
    df: pd.DataFrame, metric: str, group_by: str | None = None, filters: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if metric not in METRICS:
        raise QueryError(f"Unknown metric '{metric}'. Allowed: {list(METRICS)}")
    if group_by is not None and group_by not in GROUP_BY:
        raise QueryError(f"Cannot group by '{group_by}'. Allowed: {list(GROUP_BY)}")

    subset = apply_filters(df, filters)
    result: dict[str, Any] = {"metric": metric, "filters": filters or [], "admissions_matched": int(len(subset))}
    if group_by is None:
        result["value"] = _metric(subset, metric)
    else:
        # dropna=False keeps admissions with unknown severity visible as their own group.
        groups = subset.groupby(group_by, dropna=False)
        result["group_by"] = group_by
        result["groups"] = {str(k): {"value": _metric(g, metric), "admissions": int(len(g))} for k, g in groups}
    return result


def top_diagnoses(df: pd.DataFrame, n: int = 10, filters: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not 1 <= n <= 25:
        raise QueryError("n must be between 1 and 25")
    subset = apply_filters(df, filters)
    counts = subset["diagnosis"].dropna().value_counts().head(n)
    return {
        "admissions_matched": int(len(subset)),
        "top_diagnoses": [{"diagnosis": str(k), "admissions": int(v)} for k, v in counts.items()],
    }
