"""Standalone AI module: LangChain + LangGraph over a digital twin built from public data.

Independent of the backend (imports nothing from ``backend/``). Public entry point::

    from ai_module import AIModule
    module = AIModule()                      # settings from AI_* env vars / ai_module/.env
    module.invoke("¿Qué pasa si llegan 60 víctimas en 4 horas?")

Imports are lazy so the data/training CLIs do not load LangChain/LangGraph."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

if TYPE_CHECKING:
    from ai_module.config.settings import AISettings
    from ai_module.digital_twin.service import DigitalTwinService
    from ai_module.services.ai_service import AIModule

_LAZY = {
    "AIModule": "ai_module.services.ai_service",
    "AISettings": "ai_module.config.settings",
    "DigitalTwinService": "ai_module.digital_twin.service",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(name)


__all__ = ["AIModule", "AISettings", "DigitalTwinService", "__version__"]
