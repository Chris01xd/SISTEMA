"""SimulationOrchestrator: creates simulation runs, drives the hybrid engine
tick-by-tick as an async background task, persists results, invokes the AI
layer periodically, and broadcasts state deltas over the WebSocket hub.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.artificial_intelligence.resource_optimizer import genetic_optimize
from app.core.config import settings
from app.database.session import SessionLocal
from app.digital_twin.serializer import tick_result_to_ws_payload
from app.models.clinical import MedicalStaff, StaffRole
from app.models.emergency import EmergencyEvent
from app.models.hospital import Bed, BedType, Floor, Room
from app.models.simulation import Simulation, SimulationResult, SimulationStatus
from app.schemas.simulation import SimulationCreate
from app.services.event_bus import event_bus
from app.simulation.abm.model import HospitalMCEModel
from app.simulation.hybrid_engine import HybridSimulationEngine
from app.simulation.system_dynamics.sd_model import SDParameters

# Used when a hospital has no seeded infrastructure yet, so a simulation still
# has something to allocate instead of running with zero capacity.
_FALLBACK_CAPACITY = {"er": 30, "icu": 10, "ward": 50, "or_tables": 4}


def create_simulation(db: Session, user_id: uuid.UUID, payload: SimulationCreate) -> Simulation:
    simulation = Simulation(
        event_id=payload.event_id,
        created_by=user_id,
        horizon_minutes=payload.horizon_minutes,
        timestep_minutes=payload.timestep_minutes,
        status=SimulationStatus.PENDING,
        config=payload.model_dump(mode="json"),
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)
    return simulation


def _hospital_bed_capacity(db: Session, hospital_id: uuid.UUID) -> dict[str, int]:
    """Real bed/table counts for this hospital, by type — so simulation
    capacity (and therefore report totals) matches what was actually seeded
    for it instead of a fixed number every hospital shared."""
    counts = (
        db.query(Bed.bed_type, func.count(Bed.id))
        .join(Room, Bed.room_id == Room.id)
        .join(Floor, Room.floor_id == Floor.id)
        .filter(Floor.hospital_id == hospital_id)
        .group_by(Bed.bed_type)
        .all()
    )
    by_type = {bed_type.value: count for bed_type, count in counts}
    capacity = {
        "er": by_type.get(BedType.ER.value, 0),
        "icu": by_type.get(BedType.ICU.value, 0),
        "ward": by_type.get(BedType.WARD.value, 0),
        "or_tables": by_type.get(BedType.OR_TABLE.value, 0),
    }
    if sum(capacity.values()) == 0:
        return dict(_FALLBACK_CAPACITY)
    return capacity


def _hospital_staff_counts(db: Session, hospital_id: uuid.UUID, total_beds: int) -> dict[str, int]:
    counts = (
        db.query(MedicalStaff.role, func.count(MedicalStaff.id))
        .filter(MedicalStaff.hospital_id == hospital_id, MedicalStaff.is_available.is_(True))
        .group_by(MedicalStaff.role)
        .all()
    )
    by_role = {role.value: count for role, count in counts}
    if by_role:
        return {
            "doctors": by_role.get(StaffRole.DOCTOR.value, 0),
            "surgeons": by_role.get(StaffRole.SURGEON.value, 0),
            "nurses": by_role.get(StaffRole.NURSE.value, 0),
        }
    # No staff records seeded for this hospital yet — scale the same
    # doctor/surgeon/nurse ratio the old fixed defaults used (15/6/30 for 94
    # beds) to the real bed count instead, so bigger/smaller hospitals aren't
    # over- or under-staffed relative to their actual capacity.
    scale = max(total_beds, 1) / 94
    return {
        "doctors": max(1, round(15 * scale)),
        "surgeons": max(1, round(6 * scale)),
        "nurses": max(1, round(30 * scale)),
    }


def _build_abm_from_event(db: Session, event: EmergencyEvent, sim: Simulation) -> HospitalMCEModel:
    capacity = _hospital_bed_capacity(db, event.hospital_id)
    total_beds = sum(capacity.values())
    staff = _hospital_staff_counts(db, event.hospital_id, total_beds)

    return HospitalMCEModel(
        n_beds_er=capacity["er"],
        n_beds_icu=capacity["icu"],
        n_beds_ward=capacity["ward"],
        n_operating_rooms=capacity["or_tables"],
        n_doctors=staff["doctors"],
        n_surgeons=staff["surgeons"],
        n_nurses=staff["nurses"],
        n_ambulances=6,
        severity_distribution=event.params.get(
            "severity_distribution",
            {"minor": 0.4, "delayed": 0.3, "immediate": 0.2, "critical": 0.08, "expectant": 0.02},
        ),
        expected_victims=event.expected_victims,
        arrival_curve=event.params.get("arrival_curve", "surge_decay"),
        arrival_window_minutes=event.params.get("arrival_window_minutes", 60),
        timestep_minutes=sim.timestep_minutes,
        assignment_strategy=sim.config.get("assignment_strategy", "greedy"),
    )


async def run_simulation(simulation_id: uuid.UUID) -> None:
    """Runs entirely tick-by-tick with a real-time-proportional sleep so the
    WebSocket stream is consumable live by the frontend; for pure batch/report
    generation this loop can be run without the `asyncio.sleep`."""
    db = SessionLocal()
    try:
        sim = db.get(Simulation, simulation_id)
        event = db.get(EmergencyEvent, sim.event_id)
        sim.status = SimulationStatus.RUNNING
        sim.started_at = datetime.now(timezone.utc)
        db.commit()

        abm = _build_abm_from_event(db, event, sim)
        engine = HybridSimulationEngine(abm, SDParameters())
        n_ticks = sim.horizon_minutes // max(sim.timestep_minutes, 1)

        recommendations: list[str] = []

        for _ in range(n_ticks):
            result = engine.step()

            db.add(
                SimulationResult(
                    simulation_id=sim.id,
                    tick=result.tick,
                    timestamp=datetime.now(timezone.utc),
                    metrics={
                        **result.census,
                        "saturation_index": result.saturation_index,
                        "status_band": result.status_band,
                        "staff_fatigue": result.staff_fatigue,
                    },
                )
            )
            if result.tick % 20 == 0:
                db.commit()

            await event_bus.publish(tick_result_to_ws_payload(str(sim.id), result))

            if result.tick % settings.AI_INFERENCE_EVERY_N_TICKS == 0 and result.status_band != "GREEN":
                candidate = genetic_optimize(
                    max_overflow_beds=10,
                    max_recallable_staff=8,
                    current_waiting=result.census["patients_waiting"],
                    current_saturation=result.saturation_index,
                )
                rec = (
                    f"[t={result.tick}min] Saturation={result.status_band}: open "
                    f"{candidate.overflow_beds_opened} overflow beds and recall "
                    f"{candidate.on_call_staff_recalled} on-call staff "
                    f"(est. wait reduction {candidate.expected_wait_reduction_minutes:.0f} min, "
                    f"overtime cost ${candidate.staffing_overtime_cost:.0f})."
                )
                recommendations.append(rec)
                await event_bus.publish(
                    {"type": "ai_recommendation", "simulation_id": str(sim.id), "message": rec}
                )

            await asyncio.sleep(0)  # yield control; swap for real-time pacing if desired

        sim.status = SimulationStatus.COMPLETED
        sim.finished_at = datetime.now(timezone.utc)
        sim.config = {**sim.config, "recommendations": recommendations}
        db.commit()
    except Exception:
        sim = db.get(Simulation, simulation_id)
        if sim:
            sim.status = SimulationStatus.FAILED
            db.commit()
        raise
    finally:
        db.close()
