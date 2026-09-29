"""ML model comparison: Frontend -> Backend -> Langflow flow -> ai_module.ml -> Backend -> Frontend.

In "langflow" mode (default) the backend never trains anything itself: it sends the request
parameters to the Langflow flow through its run API, then parses and validates the flow's
JSON output. "direct" mode runs the same ai_module steps in-process for environments
without a Langflow server; the response says which engine was used."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.ml_analysis import MLAnalysisRequest, MLAnalysisResult
from app.services import ai_module_service
from app.services.langflow_client import LangflowClient, LangflowError, parse_flow_json

logger = logging.getLogger(__name__)

# ai_module.ml error codes -> HTTP status.
_CLIENT_ERRORS = {"invalid_request", "target_not_found", "invalid_columns", "empty_dataset", "insufficient_data"}
_UNAVAILABLE = {"dataset_unavailable"}


def client() -> LangflowClient:
    """Separate function so tests can inject a mock transport."""
    return LangflowClient(settings.LANGFLOW_URL, settings.LANGFLOW_API_KEY, settings.LANGFLOW_TIMEOUT_SECONDS)


def _flow_error(payload: dict[str, Any]) -> HTTPException:
    error = payload.get("error") or {}
    code, message = error.get("code", "analysis_error"), error.get("message", "The analysis failed")
    status = 422 if code in _CLIENT_ERRORS else 503 if code in _UNAVAILABLE else 502
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _validated(payload: dict[str, Any], engine: str, langflow: dict[str, Any] | None) -> dict[str, Any]:
    if payload.get("status") == "error":
        raise _flow_error(payload)
    try:
        result = MLAnalysisResult.model_validate({**payload, "engine": engine, "langflow": langflow})
    except ValidationError as exc:
        missing = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors(include_url=False, include_input=False)})[:10]
        raise HTTPException(status_code=502, detail={"code": "incomplete_result", "message": f"The analysis result is incomplete: {missing}"}) from None
    return result.model_dump(mode="json")


def _run_langflow(request: MLAnalysisRequest) -> dict[str, Any]:
    session_id = f"ml-{uuid.uuid4()}"
    params = request.model_dump(exclude_none=True)
    try:
        response = client().run_flow(settings.LANGFLOW_FLOW_ID, json.dumps(params), session_id=session_id)
        payload = parse_flow_json(response)
    except LangflowError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": "langflow_error", "message": str(exc)}) from None
    return _validated(payload, "langflow", {"flow": settings.LANGFLOW_FLOW_ID, "session_id": session_id, "url": settings.LANGFLOW_URL})


def _run_direct(request: MLAnalysisRequest) -> dict[str, Any]:
    ml = ai_module_service._import("ai_module.ml.service")
    params = request.model_dump(exclude_none=True)
    language = params.pop("language", "es")
    try:
        payload = ml.MLAnalysisService().run(params, language=language)
    except Exception as exc:  # MLAnalysisError subclasses carry a safe code/message
        payload = ml.error_output(exc)
    return _validated(payload, "direct", None)


def run_analysis(request: MLAnalysisRequest, user_id: Any) -> dict[str, Any]:
    mode = settings.ML_ANALYSIS_MODE
    logger.info("ml analysis requested: user=%s mode=%s dataset=%s", user_id, mode, request.dataset)
    result = _run_direct(request) if mode == "direct" else _run_langflow(request)
    record_engine(result["run_id"], result["engine"])
    logger.info("ml analysis done: user=%s run_id=%s best=%s engine=%s", user_id, result["run_id"], result["best_model"]["name"], result["engine"])
    return result


def status() -> dict[str, Any]:
    info: dict[str, Any] = {"mode": settings.ML_ANALYSIS_MODE, "flow": settings.LANGFLOW_FLOW_ID, "langflow_url": settings.LANGFLOW_URL}
    info["langflow_reachable"] = client().health() if settings.ML_ANALYSIS_MODE == "langflow" else None
    return info


def latest_best_model() -> dict[str, Any]:
    ml = ai_module_service._import("ai_module.ml.service")
    best_dir = ml.MLAnalysisService().best_dir
    pointer = best_dir / "latest.json"
    if not pointer.exists():
        raise HTTPException(status_code=404, detail="No analysis has been run yet")
    run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    return json.loads((best_dir / run_id / "metadata.json").read_text(encoding="utf-8"))


def _run_output(run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ml = ai_module_service._import("ai_module.ml.service")
    errors = ai_module_service._import("ai_module.ml.errors")
    store = ml.MLAnalysisService().store
    try:
        output = store.load_json(run_id, "output")
    except errors.RunNotFoundError as exc:  # also rejects malformed ids (no path traversal)
        raise HTTPException(status_code=404, detail=str(exc)) from None
    try:
        meta = store.load_json(run_id, "backend")
    except errors.RunNotFoundError:
        meta = {"engine": "unknown"}
    return output, meta


def record_engine(run_id: str, engine: str) -> None:
    ml = ai_module_service._import("ai_module.ml.service")
    try:
        ml.MLAnalysisService().store.save_json(run_id, "backend", {"engine": engine})
    except Exception:  # the run folder may live on another host (remote Langflow): report then says "unknown"
        logger.info("could not record engine for run %s", run_id)


def build_report(run_id: str, fmt: str) -> Path:
    from app.reports.ml_analysis_report import generate_ml_pdf, generate_ml_xlsx

    output, meta = _run_output(run_id)
    target = Path(settings.REPORT_OUTPUT_DIR) / f"ml_analysis_{run_id}.{fmt}"
    return generate_ml_pdf(output, target, meta.get("engine", "unknown")) if fmt == "pdf" else generate_ml_xlsx(output, target)
