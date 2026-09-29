"""HybridSimulationEngine: couples the ABM (Mesa) and System Dynamics layers.

Two-way coupling per docs/ARCHITECTURE.md section 6.3:
  ABM -> SD (bottom-up): census() feeds SD stocks every tick (ground truth).
  SD -> ABM (top-down): staff_fatigue and saturation_index feed back into
  agent behavior (clinician effective capacity, patient re-triage pressure).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.simulation.abm.model import HospitalMCEModel
from app.simulation.system_dynamics.sd_model import SDParameters, SystemDynamicsModel


@dataclass
class HybridTickResult:
    tick: int
    census: dict
    saturation_index: float
    status_band: str
    staff_fatigue: float


class HybridSimulationEngine:
    def __init__(
        self,
        abm: HospitalMCEModel,
        sd_params: SDParameters | None = None,
        on_tick: Callable[[HybridTickResult], None] | None = None,
    ):
        self.abm = abm
        self.sd = SystemDynamicsModel(sd_params)
        self.on_tick = on_tick

    def run(self, n_ticks: int) -> list[HybridTickResult]:
        results: list[HybridTickResult] = []
        for _ in range(n_ticks):
            results.append(self.step())
        return results

    def step(self) -> HybridTickResult:
        # SD -> ABM feedback from the *previous* tick's computed values
        self.abm.saturation_index = self.sd.state.saturation_index
        self.abm.staff_fatigue = self.sd.state.staff_fatigue

        self.abm.step()

        census = self.abm.census()
        # ABM -> SD ground-truthing
        self.sd.sync_from_census(census)
        self.sd.step(dt_minutes=self.abm.timestep_minutes)

        result = HybridTickResult(
            tick=self.abm.current_tick,
            census=census,
            saturation_index=self.sd.state.saturation_index,
            status_band=SystemDynamicsModel.status_band(self.sd.state.saturation_index),
            staff_fatigue=self.sd.state.staff_fatigue,
        )
        if self.on_tick:
            self.on_tick(result)
        return result
