"""System Dynamics layer: aggregate stocks/flows + feedback loops.

Implements the equations documented in docs/ARCHITECTURE.md section 6.2 using
explicit Euler integration (adequate at the 1-minute timestep this system
runs at; swap for RK4 in scipy.integrate if a coarser Δt is ever used).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SDParameters:
    turnover_delay_minutes: float = 20.0
    admission_time_constant_minutes: float = 5.0
    fatigue_accumulation_rate: float = 0.0008  # k1
    fatigue_recovery_rate: float = 0.02  # k2
    queue_reference: float = 30.0  # QueueRef, normalizes wait pressure
    staff_reference: float = 1.0  # StaffRef, normalizes effective staffing
    w_occupancy: float = 0.5
    w_queue: float = 0.3
    w_staff: float = 0.2


@dataclass
class SDState:
    patients_waiting: float = 0.0
    beds_occupied: float = 0.0
    beds_capacity: float = 1.0
    staff_active: float = 0.0
    staff_total: float = 1.0
    staff_fatigue: float = 0.0
    saturation_index: float = 0.0
    history: list[dict] = field(default_factory=list)


class SystemDynamicsModel:
    """Stock-flow model coupled bidirectionally with the ABM by HybridSimulationEngine."""

    def __init__(self, params: SDParameters | None = None):
        self.params = params or SDParameters()
        self.state = SDState()

    def sync_from_census(self, census: dict) -> None:
        """ABM -> SD (bottom-up): ground-truth the stocks from the agent population."""
        self.state.patients_waiting = float(census["patients_waiting"])
        self.state.beds_occupied = float(census["beds_occupied"])
        self.state.beds_capacity = float(max(census["beds_total"], 1))
        self.state.staff_active = float(census["staff_active"])
        self.state.staff_total = float(max(census["staff_total"], 1))

    def step(self, dt_minutes: float) -> None:
        p = self.params
        s = self.state

        # Staff fatigue: accumulates with active load, recovers otherwise (logistic-ish clamp)
        d_fatigue = p.fatigue_accumulation_rate * s.staff_active - p.fatigue_recovery_rate * (
            s.staff_total - s.staff_active
        )
        s.staff_fatigue = min(1.0, max(0.0, s.staff_fatigue + d_fatigue * dt_minutes))

        staff_effective = s.staff_total * (1 - s.staff_fatigue)

        # Saturation index: weighted combination of occupancy, queue pressure, staff shortfall
        occupancy_term = s.beds_occupied / s.beds_capacity
        queue_term = min(1.0, s.patients_waiting / max(p.queue_reference, 1e-6))
        staff_term = 1.0 - min(1.0, staff_effective / max(s.staff_total * p.staff_reference, 1e-6))

        s.saturation_index = (
            p.w_occupancy * occupancy_term + p.w_queue * queue_term + p.w_staff * staff_term
        )
        s.saturation_index = min(1.0, max(0.0, s.saturation_index))

        s.history.append(
            {
                "patients_waiting": s.patients_waiting,
                "beds_occupied": s.beds_occupied,
                "beds_capacity": s.beds_capacity,
                "staff_active": s.staff_active,
                "staff_fatigue": s.staff_fatigue,
                "saturation_index": s.saturation_index,
            }
        )

    @staticmethod
    def status_band(saturation_index: float) -> str:
        if saturation_index < 0.6:
            return "GREEN"
        if saturation_index < 0.85:
            return "YELLOW"
        return "RED"
