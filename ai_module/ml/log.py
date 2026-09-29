"""Human-readable, tagged progress logs (``[ML] Training Random Forest``) that are also
structured events (``event=ml_step, tag=ML, run_id=...``) for log aggregation."""
from __future__ import annotations

import logging
from typing import Any

from ai_module.config.logging import get_logger, log_event

logger = get_logger("ai_module.ml")


def ml_log(tag: str, message: str, level: int = logging.INFO, **fields: Any) -> None:
    log_event(logger, "ml_step", level, tag=tag, message=f"[{tag}] {message}", **fields)
