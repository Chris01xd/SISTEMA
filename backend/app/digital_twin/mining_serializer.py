"""Persists twin-state updates back to `loading_unit_digital_twins` and keeps
the in-memory MiningTwinStateManager mirror in sync."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.digital_twin.mining_state_manager import LoadingUnitTwinState, mining_twin_state_manager
from app.models.equipment import LoadingUnitDigitalTwin, TwinState


def apply_prediction_to_twin(
    db: Session, twin: LoadingUnitDigitalTwin, equipment_state: str, risk_level: float, prediction: dict
) -> dict:
    twin.current_state = TwinState(equipment_state)
    twin.risk_level = risk_level
    twin.last_prediction = prediction
    db.add(twin)
    db.commit()

    mining_twin_state_manager.register(
        LoadingUnitTwinState(
            id=str(twin.loading_unit_id), code="", state=equipment_state, risk_level=risk_level, last_prediction=prediction
        )
    )
    return mining_twin_state_manager.apply_prediction(str(twin.loading_unit_id), equipment_state, risk_level, prediction)
