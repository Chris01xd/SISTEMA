"""Glue shared by the Langflow custom components (``langflow_integration/components``).

Each component calls ``run_step`` with its upstream reports. If any upstream report is an
error, it is passed through unchanged. If the step raises, the exception becomes an error
report with the same contract as a success (``service.error_output``). The flow therefore
always reaches its Chat Output with a JSON the backend can interpret, instead of aborting
halfway with an opaque Langflow exception."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ai_module.ml.log import ml_log
from ai_module.ml.service import MLAnalysisService, error_output


def data_of(value: Any) -> dict[str, Any] | None:
    """Payload of a connected ``Data`` input; None when the input is not connected
    (Langflow then passes an empty string), so the step reports a missing connection."""
    data = getattr(value, "data", None)
    return data if isinstance(data, dict) else None


def upstream_error(*reports: dict[str, Any] | None) -> dict[str, Any] | None:
    for report in reports:
        if not report:
            return error_output(_Missing())
        if report.get("status") == "error":
            return report
    return None


class _Missing(Exception):
    pass


def run_step(step: str, fn: Callable[..., dict[str, Any]], *reports: dict[str, Any] | None, **kwargs: Any) -> dict[str, Any]:
    failed = upstream_error(*reports)
    if failed is not None:
        if failed["error"]["code"] == "internal_error":
            failed = {**failed, "error": {"code": "incomplete_result", "message": f"Step '{step}' received no input: check the flow connections"}}
        return failed
    try:
        return fn(*reports, **kwargs)
    except Exception as exc:
        ml_log("LANGFLOW", f"Step '{step}' failed: {type(exc).__name__}", 40)
        return error_output(exc)


def parse_message(text: str | None) -> dict[str, Any]:
    """Chat Input text -> request dict. Empty text means 'use the component's fields'."""
    text = (text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"_unparsable_message": text[:200]}
    return payload if isinstance(payload, dict) else {"_unparsable_message": text[:200]}


def service() -> MLAnalysisService:
    return MLAnalysisService()
