"""Bridge to the standalone LangChain module (`langchain_module/hospital_assistant`).

The module is deliberately independent of this backend: it is imported lazily, and when it
(or LangChain) is not installed the endpoint degrades to a clear 501 instead of breaking the
API, the same convention as the optional LSTM/torch dependency.

Data egress: the model provider receives the user's question and the *aggregate* results of
the read-only tools (never patient rows). That is acceptable for the public, de-identified
MIMIC demo; pointing this at real hospital data would be a privacy decision to make first."""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
from fastapi import HTTPException

from app.core.config import settings
from app.models.dataset import Dataset
from app.schemas.assistant import ChatTurn
from app.services import dataset_service

logger = logging.getLogger(__name__)


def _module_dir() -> Path:
    if settings.ASSISTANT_MODULE_PATH:
        return Path(settings.ASSISTANT_MODULE_PATH)
    # backend/app/services/<this file> -> <repo>/langchain_module (in Docker: /langchain_module).
    return Path(__file__).resolve().parents[3] / "langchain_module"


def _import_module() -> ModuleType:
    directory = _module_dir()
    # Appended (not prepended) so the module's own `tests/` package can never shadow this backend's.
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.append(str(directory))
    try:
        return importlib.import_module("hospital_assistant")
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"AI assistant is not available ({exc}). Install backend/requirements-assistant.txt "
            f"and make sure the langchain_module folder exists at {directory}.",
        )


def _uses_anthropic(model: str) -> bool:
    return ":" not in model or model.startswith("anthropic:")


def _configuration_problem() -> str | None:
    if _uses_anthropic(settings.ASSISTANT_MODEL) and not settings.ANTHROPIC_API_KEY:
        return "ANTHROPIC_API_KEY is not configured on the server."
    return None


_CREDENTIAL_ERRORS = {"AuthenticationError", "PermissionDeniedError", "AnthropicAuthenticationError"}


def _is_credentials_error(exc: BaseException) -> bool:
    """Matches provider SDK auth failures by class name anywhere in the cause chain, so this
    module needs no import of any specific provider SDK."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ in _CREDENTIAL_ERRORS:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def status() -> dict[str, Any]:
    try:
        _import_module()
    except HTTPException as exc:
        return {"available": False, "model": settings.ASSISTANT_MODEL, "detail": exc.detail}
    problem = _configuration_problem()
    return {"available": problem is None, "model": settings.ASSISTANT_MODEL, "detail": problem}


def _build_agent(module: ModuleType, df: pd.DataFrame):
    """Separate function so tests can substitute a scripted model."""
    api_key = (settings.ANTHROPIC_API_KEY or None) if _uses_anthropic(settings.ASSISTANT_MODEL) else None
    module_settings = module.config.Settings(
        dataset_path=Path("."),  # unused: the frame is passed in directly
        model=settings.ASSISTANT_MODEL,
        temperature=0.0,
        timeout=settings.ASSISTANT_TIMEOUT_SECONDS,
        api_key=api_key,  # other providers read their own env vars
    )
    return module.build_agent(df, settings=module_settings)


def ask(dataset: Dataset, question: str, history: list[ChatTurn], user_id: Any) -> dict[str, Any]:
    module = _import_module()
    problem = _configuration_problem()
    if problem:
        raise HTTPException(status_code=503, detail=f"AI assistant is not configured: {problem}")

    df = dataset_service.load_frame(dataset)
    missing = [c for c in module.data.CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Dataset is not a canonical admissions table; missing columns: {missing}")

    from langchain_core.messages import AIMessage, HumanMessage

    past = [(HumanMessage if turn.role == "user" else AIMessage)(content=turn.content) for turn in history]
    # Lengths only: questions may contain sensitive text and must not end up in logs.
    logger.info("assistant ask: user=%s dataset=%s question_chars=%d history_turns=%d", user_id, dataset.id, len(question), len(history))
    try:
        answer, messages = module.ask(_build_agent(module, df), question, past)
    except Exception as exc:
        logger.exception("assistant request failed (user=%s dataset=%s)", user_id, dataset.id)
        if _is_credentials_error(exc):
            # Retrying cannot help: this is a server-side configuration problem, not a transient failure.
            raise HTTPException(status_code=503, detail="AI assistant is misconfigured: the model provider rejected the server's credentials.")
        raise HTTPException(status_code=502, detail="The language model request failed. Try again later.")

    if not answer.strip():
        raise HTTPException(status_code=502, detail="The assistant did not produce an answer.")
    return {
        "answer": answer,
        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in module.tool_calls_made(messages)],
        "model": settings.ASSISTANT_MODEL,
    }
