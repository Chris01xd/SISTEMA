"""Agent definitions for the Mass-Casualty-Event Agent-Based Model (Mesa).

Design notes (see docs/ARCHITECTURE.md section 6.1):
- PatientAgent carries a START-triage-inspired severity and a state machine.
- ClinicianAgent/NurseAgent carry fatigue, which the SD layer feeds back into
  as `effective_capacity = base_capacity * (1 - fatigue)`.
- BedAgent/ORAgent/AmbulanceAgent are discrete resource agents with turnover delay.
"""
from __future__ import annotations

import enum
import random
import uuid

from mesa import Agent


class PatientState(str, enum.Enum):
    ARRIVED = "ARRIVED"
    TRIAGED = "TRIAGED"
    WAITING_BED = "WAITING_BED"
    IN_TREATMENT = "IN_TREATMENT"
    ICU = "ICU"
    WARD = "WARD"
    DISCHARGED = "DISCHARGED"
    DECEASED = "DECEASED"


SEVERITY_BASE_PRIORITY = {1: 10, 2: 30, 3: 60, 4: 90, 5: 0}  # expectant (5) deprioritized per triage doctrine
SEVERITY_TREATMENT_MINUTES = {1: 30, 2: 60, 3: 120, 4: 240, 5: 0}
SEVERITY_DEATH_RISK_PER_TICK_IF_UNTREATED = {1: 0.0, 2: 0.0005, 3: 0.004, 4: 0.02, 5: 0.05}


class PatientAgent(Agent):
    def __init__(self, model, age: int, sex: str, severity: int, needs_surgery: bool, needs_icu: bool):
        super().__init__(model)
        self.uid = uuid.uuid4()
        self.age = age
        self.sex = sex
        self.severity = severity
        self.needs_surgery = needs_surgery
        self.needs_icu = needs_icu
        self.state = PatientState.ARRIVED
        self.arrival_tick = model.current_tick
        self.wait_ticks = 0
        self.max_wait_ticks = max(5, int(180 / max(severity, 1)))
        self.triage_priority = SEVERITY_BASE_PRIORITY[severity]
        self.assigned_bed = None
        self.assigned_clinician = None
        self.treatment_ticks_remaining = 0

    def _reevaluate_priority(self, saturation_index: float) -> None:
        """SD -> ABM feedback: higher system saturation re-triages more aggressively."""
        escalation = 1.0 + saturation_index * 0.5
        self.triage_priority = SEVERITY_BASE_PRIORITY[self.severity] * escalation - self.wait_ticks * 0.1

    def step(self) -> None:
        model = self.model

        if self.state == PatientState.ARRIVED:
            self.state = PatientState.TRIAGED
            self._reevaluate_priority(model.saturation_index)
            self.state = PatientState.WAITING_BED
            model.waiting_queue.add(self)
            return

        if self.state == PatientState.WAITING_BED:
            self.wait_ticks += 1
            self._reevaluate_priority(model.saturation_index)
            risk = SEVERITY_DEATH_RISK_PER_TICK_IF_UNTREATED[self.severity]
            if random.random() < risk:
                self.state = PatientState.DECEASED
                model.waiting_queue.discard(self)
            return

        if self.state == PatientState.IN_TREATMENT:
            self.treatment_ticks_remaining -= 1
            if self.treatment_ticks_remaining <= 0:
                self.state = PatientState.ICU if self.needs_icu else PatientState.WARD
                if self.assigned_bed is not None:
                    self.assigned_bed.release()
                    self.assigned_bed = None
            return

        if self.state in (PatientState.ICU, PatientState.WARD):
            # simple stay duration model; discharge probability increases with severity inverse
            discharge_prob = 0.01 if self.state == PatientState.ICU else 0.03
            if random.random() < discharge_prob:
                self.state = PatientState.DISCHARGED
                if self.assigned_bed is not None:
                    self.assigned_bed.release()
                    self.assigned_bed = None

    def begin_treatment(self, bed: "BedAgent", clinician: "ClinicianAgent | None") -> None:
        self.assigned_bed = bed
        self.assigned_clinician = clinician
        self.state = PatientState.IN_TREATMENT
        self.treatment_ticks_remaining = SEVERITY_TREATMENT_MINUTES[self.severity] // max(
            self.model.timestep_minutes, 1
        )


class ClinicianAgent(Agent):
    def __init__(self, model, specialty: str, base_capacity: int = 4, is_surgeon: bool = False):
        super().__init__(model)
        self.specialty = specialty
        self.base_capacity = base_capacity
        self.is_surgeon = is_surgeon
        self.active_minutes = 0
        self.fatigue = 0.0  # [0,1], set by SD coupler each tick
        self.current_patients: set[PatientAgent] = set()

    @property
    def effective_capacity(self) -> float:
        return self.base_capacity * (1 - self.fatigue)

    def has_slack(self) -> bool:
        return len(self.current_patients) < self.effective_capacity

    def step(self) -> None:
        if self.current_patients:
            self.active_minutes += self.model.timestep_minutes


class NurseAgent(Agent):
    def __init__(self, model, assigned_area: str, capacity: int = 6):
        super().__init__(model)
        self.assigned_area = assigned_area
        self.capacity = capacity
        self.current_patients: set[PatientAgent] = set()

    def has_slack(self) -> bool:
        return len(self.current_patients) < self.capacity


class BedAgent(Agent):
    def __init__(self, model, bed_type: str, turnover_ticks: int = 15):
        super().__init__(model)
        self.bed_type = bed_type
        self.turnover_ticks = turnover_ticks
        self.status = "AVAILABLE"  # AVAILABLE | OCCUPIED | CLEANING | OUT_OF_SERVICE
        self._cleaning_remaining = 0
        self.occupant: PatientAgent | None = None

    def occupy(self, patient: PatientAgent) -> None:
        self.status = "OCCUPIED"
        self.occupant = patient

    def release(self) -> None:
        self.status = "CLEANING"
        self._cleaning_remaining = self.turnover_ticks
        self.occupant = None

    def step(self) -> None:
        if self.status == "CLEANING":
            self._cleaning_remaining -= 1
            if self._cleaning_remaining <= 0:
                self.status = "AVAILABLE"


class ORAgent(BedAgent):
    def __init__(self, model, turnover_ticks: int = 30):
        super().__init__(model, bed_type="OR_TABLE", turnover_ticks=turnover_ticks)


class AmbulanceAgent(Agent):
    def __init__(self, model, capacity_patients: int = 2):
        super().__init__(model)
        self.capacity_patients = capacity_patients
        self.status = "AVAILABLE"  # AVAILABLE | EN_ROUTE | RETURNING
