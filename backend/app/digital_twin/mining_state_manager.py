"""MiningTwinStateManager: canonical live state of each loading unit's digital
twin. Mirrors the hospital DigitalTwinStateManager pattern (in-memory mirror,
Postgres as source of truth) but for a single flat entity instead of a
floor/room/bed tree, since a loading unit has no internal spatial hierarchy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadingUnitTwinState:
    id: str
    code: str
    state: str = "NORMAL"  # NORMAL | WARNING | CRITICAL | OUT_OF_SERVICE
    risk_level: float = 0.0
    last_prediction: dict[str, Any] = field(default_factory=dict)


class MiningTwinStateManager:
    def __init__(self) -> None:
        self._twins: dict[str, LoadingUnitTwinState] = {}

    def register(self, twin: LoadingUnitTwinState) -> None:
        self._twins[twin.id] = twin

    def get(self, loading_unit_id: str) -> LoadingUnitTwinState | None:
        return self._twins.get(loading_unit_id)

    def apply_prediction(self, loading_unit_id: str, equipment_state: str, risk_level: float, prediction: dict) -> dict:
        twin = self._twins.get(loading_unit_id)
        if twin is None:
            twin = LoadingUnitTwinState(id=loading_unit_id, code=loading_unit_id)
            self._twins[loading_unit_id] = twin
        twin.state = equipment_state
        twin.risk_level = risk_level
        twin.last_prediction = prediction
        return {
            "type": "loading_unit_twin_update",
            "loading_unit_id": loading_unit_id,
            "state": twin.state,
            "risk_level": twin.risk_level,
            "prediction": twin.last_prediction,
        }


# Process-wide singleton (prototype scope, same caveat as twin_state_manager
# in app.digital_twin.state_manager for multi-worker deployment).
mining_twin_state_manager = MiningTwinStateManager()
