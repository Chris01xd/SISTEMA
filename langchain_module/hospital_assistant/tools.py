"""LangChain tools: a thin, validated wrapper over `analytics` bound to one DataFrame."""
from __future__ import annotations

from typing import Literal

import pandas as pd
from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field

from hospital_assistant import analytics

FilterValue = str | int | float | bool | list[str] | list[int | float]


class Filter(BaseModel):
    column: Literal[*analytics.FILTERABLE_COLUMNS]
    op: Literal[*analytics.OPERATORS] = Field(description="'contains' is a case-insensitive substring match; 'in' needs a list.")
    value: FilterValue


def build_tools(df: pd.DataFrame) -> list[BaseTool]:
    data = analytics.prepare(df)

    @tool
    def get_dataset_overview() -> dict:
        """Headline statistics of the admissions dataset plus the columns, groupings and metrics
        the other tools accept. Call this first when unsure what can be asked."""
        return analytics.overview(data)

    @tool
    def aggregate_admissions(
        metric: Literal[*analytics.METRICS],
        group_by: Literal[*analytics.GROUP_BY] | None = None,
        filters: list[Filter] | None = None,
    ) -> dict:
        """Compute one metric over hospital admissions, optionally filtered and split by a grouping.
        Example: metric='mortality_rate_pct', group_by='age_band'. Returns aggregates only."""
        try:
            return analytics.aggregate(data, metric, group_by, [f.model_dump() for f in filters or []])
        except analytics.QueryError as exc:
            return {"error": str(exc)}

    @tool
    def list_top_diagnoses(n: int = 10, filters: list[Filter] | None = None) -> dict:
        """Most frequent primary diagnoses (by admissions), optionally filtered. n is 1-25."""
        try:
            return analytics.top_diagnoses(data, n, [f.model_dump() for f in filters or []])
        except analytics.QueryError as exc:
            return {"error": str(exc)}

    return [get_dataset_overview, aggregate_admissions, list_top_diagnoses]
