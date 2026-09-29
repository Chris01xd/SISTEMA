"""Bridge to the standalone AI module (`ai_module/`: LangChain + LangGraph + digital twin).

Same convention as `assistant_service.py`: the module is imported lazily and is optional.
When it (or its dependencies) is missing the endpoints answer 501, and when its data has
not been prepared they answer 503; the rest of the API is unaffected.

Data egress: `/ai/invoke` sends the user's message and the digital-twin context (aggregate
statistics of the public MIMIC-IV demo + simulation results; never patient rows) to the
configured LLM provider. Direct twin endpoints never call an LLM."""
from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi import HTTPException
from pydantic import SecretStr

from app.core.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_instance: Any = None


def _module_root() -> Path:
    if settings.AI_MODULE_ROOT:
        return Path(settings.AI_MODULE_ROOT)
    # backend/app/services/<this file> -> <repo>  (in Docker: the mount point's parent, see docker-compose.yml).
    return Path(__file__).resolve().parents[3]


def _import(name: str) -> ModuleType:
    root = _module_root()
    # Appended (not prepended) so nothing in the module can shadow this backend's packages.
    if (root / "ai_module").is_dir() and str(root) not in sys.path:
        sys.path.append(str(root))
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"AI module is not available ({exc.name or 'import error'}). Install ai_module/requirements.txt "
            f"and make sure the ai_module folder exists under {root}.",
        ) from None


def get_module() -> Any:
    """Process-wide `AIModule` (model, data and graph are loaded once)."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            ai_service = _import("ai_module.services.ai_service")
            ai_settings = _import("ai_module.config.settings").AISettings()
            # Reuse the backend's Anthropic key when the module has none of its own.
            if (ai_settings.llm_provider == "anthropic" and ai_settings.llm_api_key is None
                    and settings.ANTHROPIC_API_KEY and not os.environ.get("ANTHROPIC_API_KEY")):
                ai_settings = ai_settings.model_copy(update={"llm_api_key": SecretStr(settings.ANTHROPIC_API_KEY)})
            _instance = ai_service.AIModule(ai_settings)
    return _instance


def reset() -> None:
    """Drops the cached module (tests, or after re-training)."""
    global _instance
    with _lock:
        _instance = None


def _twin_error(exc: Exception) -> HTTPException:
    errors = _import("ai_module.digital_twin.errors")
    if isinstance(exc, (errors.TwinDataNotPreparedError, errors.ModelNotTrainedError, errors.ModelArtifactError)):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


def _validation_error(exc: Any) -> HTTPException:
    # Field locations and messages only: no input echo, no internals.
    return HTTPException(status_code=422, detail=[{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors(include_url=False, include_input=False)])


def _run(fn, *args: Any) -> Any:
    from pydantic import ValidationError

    errors = _import("ai_module.digital_twin.errors")
    try:
        return fn(*args)
    except ValidationError as exc:
        raise _validation_error(exc) from None
    except errors.DigitalTwinError as exc:
        raise _twin_error(exc) from None


def status() -> dict[str, Any]:
    try:
        return {"available": True, **get_module().status()}
    except HTTPException as exc:
        return {"available": False, "detail": exc.detail}


def invoke(message: str, parameters: dict[str, Any], user_id: Any) -> dict[str, Any]:
    result = get_module().invoke(message, parameters)
    meta = result.metadata
    # Lengths only: the message may contain sensitive text.
    logger.info("ai invoke: user=%s request_id=%s status=%s message_chars=%d", user_id, result.request_id, meta["status"], len(message))
    if meta["status"] == "rejected":
        raise HTTPException(status_code=422, detail=[e["message"] for e in meta["errors"] if e["fatal"]])
    if meta["status"] == "failed":
        raise HTTPException(status_code=503, detail=result.response or "The AI module could not process the request.")
    return result.model_dump(mode="json")


def get_state() -> dict[str, Any]:
    return _run(get_module().get_state).model_dump(mode="json")


def simulate(scenario: dict[str, Any], include_timeline: bool) -> dict[str, Any]:
    result = _run(get_module().simulate, scenario).model_dump(mode="json")
    if not include_timeline:
        result.pop("timeline", None)
    return result


def predict(patient: dict[str, Any]) -> dict[str, Any]:
    return _run(get_module().predict, patient).model_dump(mode="json")
