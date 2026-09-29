"""HospitalMCEModel: the Mesa model orchestrating patient arrivals, triage,
and the resource-assignment sub-problem for a mass casualty event.

This model is stepped by the HybridSimulationEngine (see
app/simulation/hybrid_engine.py), which also owns the System Dynamics layer
and the two-way coupling described in docs/ARCHITECTURE.md section 6.3.
"""
from __future__ import annotations

import random

from mesa import Model

from app.simulation.abm.agents import (
    AmbulanceAgent,
    BedAgent,
    ClinicianAgent,
    NurseAgent,
    ORAgent,
    PatientAgent,
    PatientState,
)


def sample_severity(distribution: dict[str, float]) -> int:
    """distribution keys: minor, delayed, immediate, critical, expectant -> severities 1..5"""
    keys = ["minor", "delayed", "immediate", "critical", "expectant"]
    weights = [distribution.get(k, 0.0) for k in keys]
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]
    return random.choices([1, 2, 3, 4, 5], weights=weights, k=1)[0]


def arrival_curve_rate(curve: str, tick: int, window_ticks: int, total_victims: int) -> float:
    """Expected number of arrivals at this tick, given the scenario's arrival curve shape."""
    if window_ticks <= 0:
        return 0.0
    if curve == "linear":
        return total_victims / window_ticks if tick < window_ticks else 0.0
    if curve == "poisson_steady":
        return total_victims / window_ticks if tick < window_ticks else 0.0
    # default: "surge_decay" -- fast initial surge (first 20% of window) then exponential decay
    surge_end = max(1, int(window_ticks * 0.2))
    if tick < surge_end:
        return (total_victims * 0.6) / surge_end
    decay_ticks = window_ticks - surge_end
    if tick < window_ticks and decay_ticks > 0:
        return (total_victims * 0.4) / decay_ticks
    return 0.0


class HospitalMCEModel(Model):
    def __init__(
        self,
        n_beds_er: int,
        n_beds_icu: int,
        n_beds_ward: int,
        n_operating_rooms: int,
        n_doctors: int,
        n_surgeons: int,
        n_nurses: int,
        n_ambulances: int,
        severity_distribution: dict[str, float],
        expected_victims: int,
        arrival_curve: str,
        arrival_window_minutes: int,
        timestep_minutes: int = 1,
        assignment_strategy: str = "greedy",
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        self.timestep_minutes = timestep_minutes
        self.current_tick = 0
        self.severity_distribution = severity_distribution
        self.expected_victims = expected_victims
        self.arrival_curve = arrival_curve
        self.arrival_window_ticks = max(1, arrival_window_minutes // max(timestep_minutes, 1))
        self.assignment_strategy = assignment_strategy

        # SD -> ABM coupling inputs, set externally by the hybrid engine each tick
        self.saturation_index = 0.0
        self.staff_fatigue = 0.0

        self.waiting_queue: set[PatientAgent] = set()
        self._arrivals_carry = 0.0

        self.beds: list[BedAgent] = []
        self.ors: list[ORAgent] = []
        self.doctors: list[ClinicianAgent] = []
        self.surgeons: list[ClinicianAgent] = []
        self.nurses: list[NurseAgent] = []
        self.ambulances: list[AmbulanceAgent] = []
        self.patients: list[PatientAgent] = []

        for _ in range(n_beds_er):
            b = BedAgent(self, bed_type="ER")
            self.beds.append(b)
        for _ in range(n_beds_icu):
            b = BedAgent(self, bed_type="ICU", turnover_ticks=45)
            self.beds.append(b)
        for _ in range(n_beds_ward):
            b = BedAgent(self, bed_type="WARD", turnover_ticks=20)
            self.beds.append(b)
        for _ in range(n_operating_rooms):
            self.ors.append(ORAgent(self))
        for _ in range(n_doctors):
            self.doctors.append(ClinicianAgent(self, specialty="emergency"))
        for _ in range(n_surgeons):
            self.surgeons.append(ClinicianAgent(self, specialty="surgery", is_surgeon=True))
        for _ in range(n_nurses):
            self.nurses.append(NurseAgent(self, assigned_area="general"))
        for _ in range(n_ambulances):
            self.ambulances.append(AmbulanceAgent(self))

    # ---- arrivals -------------------------------------------------------
    def _spawn_arrivals(self) -> int:
        rate = arrival_curve_rate(
            self.arrival_curve, self.current_tick, self.arrival_window_ticks, self.expected_victims
        )
        self._arrivals_carry += rate
        n_arrivals = int(self._arrivals_carry)
        self._arrivals_carry -= n_arrivals

        for _ in range(n_arrivals):
            severity = sample_severity(self.severity_distribution)
            age = random.randint(1, 95)
            sex = random.choice(["M", "F"])
            needs_surgery = severity >= 4 and random.random() < 0.5
            needs_icu = severity >= 3 and random.random() < 0.4
            patient = PatientAgent(
                self, age=age, sex=sex, severity=severity, needs_surgery=needs_surgery, needs_icu=needs_icu
            )
            self.patients.append(patient)
        return n_arrivals

    # ---- resource assignment sub-problem ---------------------------------
    def _assign_resources(self) -> None:
        """Baseline greedy assignment: highest triage_priority first.

        When assignment_strategy != "greedy", app.artificial_intelligence
        optimizers (GA/RL) may replace this with a smarter allocation by
        calling `apply_external_assignment()` before this method runs.
        """
        if not self.waiting_queue:
            return

        ordered = sorted(self.waiting_queue, key=lambda p: p.triage_priority, reverse=True)
        for patient in ordered:
            bed = self._find_bed_for(patient)
            if bed is None:
                continue
            clinician = self._find_clinician_for(patient)
            bed.occupy(patient)
            if clinician is not None:
                clinician.current_patients.add(patient)
            patient.begin_treatment(bed, clinician)
            self.waiting_queue.remove(patient)

    def _find_bed_for(self, patient: PatientAgent) -> BedAgent | None:
        if patient.needs_surgery:
            pool = [o for o in self.ors if o.status == "AVAILABLE"]
        elif patient.needs_icu:
            pool = [b for b in self.beds if b.bed_type == "ICU" and b.status == "AVAILABLE"]
        else:
            pool = [b for b in self.beds if b.bed_type == "ER" and b.status == "AVAILABLE"]
            if not pool:
                pool = [b for b in self.beds if b.bed_type == "WARD" and b.status == "AVAILABLE"]
        return pool[0] if pool else None

    def _find_clinician_for(self, patient: PatientAgent) -> ClinicianAgent | None:
        pool = self.surgeons if patient.needs_surgery else self.doctors
        available = [c for c in pool if c.has_slack()]
        return available[0] if available else None

    # ---- mesa step --------------------------------------------------------
    def step(self) -> None:
        # apply SD -> ABM feedback before behavior this tick
        for c in self.doctors + self.surgeons:
            c.fatigue = self.staff_fatigue

        self._spawn_arrivals()

        # Explicit random-order activation of active patient agents (Mesa 3.x
        # removed mesa.time.RandomActivation in favor of AgentSet-based
        # scheduling; patients here are tracked in our own list rather than
        # via model.agents so terminal-state patients can be skipped cheaply).
        active_patients = [
            p for p in self.patients if p.state not in (PatientState.DISCHARGED, PatientState.DECEASED)
        ]
        random.shuffle(active_patients)
        for patient in active_patients:
            patient.step()

        for b in self.beds:
            b.step()
        for o in self.ors:
            o.step()
        for c in self.doctors + self.surgeons:
            c.step()

        self._assign_resources()
        self.current_tick += 1

    # ---- census used by the SD coupler (ABM -> SD, ground truth) ----------
    def census(self) -> dict:
        beds_total = len(self.beds) + len(self.ors)
        beds_occupied = sum(1 for b in self.beds if b.status == "OCCUPIED") + sum(
            1 for o in self.ors if o.status == "OCCUPIED"
        )
        staff_total = len(self.doctors) + len(self.surgeons)
        staff_active = sum(1 for c in self.doctors + self.surgeons if c.current_patients)
        active_patients = [p for p in self.patients if p.state != PatientState.DISCHARGED and p.state != PatientState.DECEASED]
        return {
            "patients_waiting": len(self.waiting_queue),
            "patients_in_treatment": sum(1 for p in active_patients if p.state == PatientState.IN_TREATMENT),
            "patients_icu": sum(1 for p in active_patients if p.state == PatientState.ICU),
            "patients_ward": sum(1 for p in active_patients if p.state == PatientState.WARD),
            "patients_deceased": sum(1 for p in self.patients if p.state == PatientState.DECEASED),
            "patients_discharged": sum(1 for p in self.patients if p.state == PatientState.DISCHARGED),
            "beds_total": beds_total,
            "beds_occupied": beds_occupied,
            "staff_total": staff_total,
            "staff_active": staff_active,
            "total_patients_seen": len(self.patients),
        }
