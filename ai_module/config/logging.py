"""Structured (JSON) logging with request correlation and secret redaction.

The module is a library: importing it never reconfigures the host's logging. Loggers
propagate to whatever the host set up (e.g. uvicorn); CLIs call ``configure_logging``.
Every event is emitted as ``log_event(logger, "event_name", **fields)`` so it is
machine-parseable, and the current ``request_id`` is attached automatically."""
from __future__ import annotations

import contextvars
import json
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("ai_request_id", default=None)

# Values that look like credentials are never written, whatever field they arrive in.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
)
_SECRET_KEYS = re.compile(r"(?i)(api[_-]?key|secret|token|password|authorization)")
REDACTED = "[REDACTED]"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("ai_module") else f"ai_module.{name}")


def set_request_id(request_id: str | None) -> contextvars.Token:
    return _request_id.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id.reset(token)


def current_request_id() -> str | None:
    return _request_id.get()


def redact(value: Any) -> Any:
    """Recursively removes secret-looking keys and values from a log payload."""
    if isinstance(value, dict):
        return {k: (REDACTED if _SECRET_KEYS.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(REDACTED, value)
        return value
    return value


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emits one structured event. Only pass metadata (ids, sizes, durations, names):
    user text and model output must not be logged."""
    payload = redact({"event": event, "request_id": current_request_id(), **fields})
    # Hosts with their own formatter (uvicorn, Langflow) print the record message: use the
    # human-readable message when there is one ("[ML] Training Random Forest").
    logger.log(level, payload.get("message") or event, extra={"structured": payload})


@contextmanager
def timed(logger: logging.Logger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Logs ``event`` with ``duration_ms`` and ``status``; callers may add fields to the yielded dict."""
    extra: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield extra
    except Exception as exc:
        log_event(logger, event, logging.WARNING, status="error", error_type=type(exc).__name__,
                  duration_ms=round((time.perf_counter() - start) * 1000, 1), **fields, **extra)
        raise
    log_event(logger, event, status="ok", duration_ms=round((time.perf_counter() - start) * 1000, 1), **fields, **extra)


class JsonFormatter(logging.Formatter):
    """One JSON object per line; structured fields from ``log_event`` are merged in."""

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
        }
        structured = getattr(record, "structured", None)
        if structured:
            base.update(structured)
        else:
            base["message"] = redact(record.getMessage())
        if record.exc_info:
            base["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(base, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable variant: ``LEVEL logger event key=value ...``."""

    def format(self, record: logging.LogRecord) -> str:
        structured = getattr(record, "structured", None)
        if not structured:
            return f"{record.levelname} {record.name} {redact(record.getMessage())}"
        fields = " ".join(f"{k}={v}" for k, v in structured.items() if k not in ("event", "message", "tag") and v is not None)
        headline = structured.get("message") or structured.get("event")
        return f"{record.levelname} {record.name} {headline} {fields}".rstrip()


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """For CLIs / standalone runs only. Configures the ``ai_module`` logger tree."""
    logger = logging.getLogger("ai_module")
    logger.setLevel(level.upper())
    if not any(getattr(h, "_ai_module", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler._ai_module = True  # type: ignore[attr-defined]
        handler.setFormatter(JsonFormatter() if json_format else TextFormatter())
        logger.addHandler(handler)
    logger.propagate = False
