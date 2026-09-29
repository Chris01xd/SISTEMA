"""Digital twin built from the public MIMIC-IV demo: ingestion -> validation -> cleaning ->
transformation -> state -> persistence -> query -> simulation / prediction.

Entry point: ``DigitalTwinService`` (imported lazily to keep the CLIs light)."""
from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "DigitalTwinService":
        from ai_module.digital_twin.service import DigitalTwinService

        return DigitalTwinService
    raise AttributeError(name)


__all__ = ["DigitalTwinService"]
