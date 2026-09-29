"""Step 9: Monte Carlo simulation of ward + ICU beds under a baseline flow and an optional surge.

The model is deliberately simple and fully driven by the observed data:

* Baseline arrivals per hour ~ Poisson(daily_admissions x observed share of admissions in that hour of day).
* Each patient is a bootstrap draw of a *real* admission (ICU hours, ward hours, ESI);
  surge patients are drawn from real ED admissions of the requested ESI level.
* ICU patients occupy an ICU bed first, then move to the ward for the rest of their stay.
* Beds are assigned every hour by priority (ESI 1 first; ICU step-downs before new ward
  admissions; non-ED admissions last), then first-come first-served. Patients without a
  free bed wait (boarding); the stay clock starts when a bed is assigned.
* Not modeled: deaths while waiting, transfers out, staff limits, discharges timed by day.

A warm-up period is simulated and discarded so the horizon starts from a loaded hospital."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_module.schemas.twin import (
    CapacityAssumptions,
    QueueStats,
    SimulationResult,
    StatusColor,
    TimelinePoint,
    UnitStats,
)

MAX_TIMELINE_POINTS = 168
_STEPDOWN_PRIORITY = 10  # ICU -> ward transfers are served before new ward admissions
_NO_ESI_PRIORITY = 0

ASSUMPTIONS = [
    "Patients are bootstrap samples of real MIMIC-IV demo admissions (length of stay and ICU time as observed).",
    "Baseline arrivals follow a Poisson process shaped by the observed admissions per hour of day.",
    "ICU patients use the ICU first and then a ward bed for the remainder of their stay.",
    "Beds are assigned by ESI priority, then arrival order; the stay starts when a bed is assigned.",
    "Not modeled: deaths while waiting, transfers to other hospitals, staffing limits.",
]


def status_color(utilization: float) -> StatusColor:
    """Same thresholds as the backend's 3D twin (``saturation_to_color``)."""
    if utilization < 0.6:
        return "GREEN"
    if utilization < 0.85:
        return "YELLOW"
    return "RED"


@dataclass(frozen=True)
class PatientPool:
    """Arrays sampled by the simulator, built once from the processed admissions."""

    icu_hours: np.ndarray
    ward_hours: np.ndarray
    priority: np.ndarray  # 6 - ESI for ED admissions, 0 otherwise
    esi: np.ndarray  # float, NaN when unknown
    hour_weights: np.ndarray  # 24 probabilities

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "PatientPool":
        esi = df["ed_acuity_esi"].to_numpy(dtype="float64")
        counts = df["admit_hour"].value_counts().reindex(range(24), fill_value=0).to_numpy(dtype="float64")
        return cls(
            icu_hours=df["icu_hours"].to_numpy(dtype="float64"),
            ward_hours=(df["los_hours"] - df["icu_hours"]).clip(lower=0).to_numpy(dtype="float64"),
            priority=np.where(np.isnan(esi), _NO_ESI_PRIORITY, 6 - np.nan_to_num(esi, nan=0)).astype("int64"),
            esi=esi,
            hour_weights=counts / counts.sum(),
        )

    def observed_esi_mix(self) -> dict[int, float]:
        known = self.esi[~np.isnan(self.esi)].astype(int)
        levels, counts = np.unique(known, return_counts=True)
        return {int(level): float(c) / float(counts.sum()) for level, c in zip(levels, counts)}

    def indices_for_esi(self, level: int) -> np.ndarray:
        return np.flatnonzero(self.esi == level)

    def mean_ward_hours(self) -> float:
        return float(self.ward_hours.mean())

    def mean_icu_hours(self) -> float:
        return float(self.icu_hours.mean())


def size_capacity(pool: PatientPool, daily_admissions: float, target_occupancy: float) -> tuple[int, int]:
    """Little's law: beds = arrival rate x mean time in bed / target occupancy."""
    per_hour = daily_admissions / 24.0
    ward = math.ceil(per_hour * pool.mean_ward_hours() / target_occupancy)
    icu = math.ceil(per_hour * pool.mean_icu_hours() / target_occupancy)
    return max(ward, 1), max(icu, 1)


@dataclass(frozen=True)
class SimulationConfig:
    capacity: CapacityAssumptions
    horizon_hours: int
    warmup_hours: int
    runs: int
    seed: int
    surge_patients: int = 0
    surge_start_hour: int = 0
    surge_duration_hours: int = 6
    surge_esi_mix: dict[int, float] | None = None


@dataclass
class _RunTrace:
    ward: np.ndarray
    icu: np.ndarray
    waiting: np.ndarray
    wait_hours: list[float]
    baseline_arrivals: int
    surge_arrivals: int


def _surge_sampler(pool: PatientPool, mix: dict[int, float]) -> tuple[list[np.ndarray], np.ndarray]:
    """Pools and probabilities per ESI level; levels with no observed patient fall back to all ED admissions."""
    all_ed = np.flatnonzero(~np.isnan(pool.esi))
    levels = sorted(mix)
    groups = [pool.indices_for_esi(level) for level in levels]
    groups = [g if len(g) else all_ed for g in groups]
    shares = np.array([mix[level] for level in levels], dtype="float64")
    return groups, shares / shares.sum()


def _sample_arrivals(pool: PatientPool, cfg: SimulationConfig, mix: dict[int, float], rng: np.random.Generator,
                     total: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Per-hour baseline and surge counts, and the pool index of every arriving patient in
    arrival order (sampled up front: one vectorized draw per run instead of one per hour)."""
    hourly_rate = cfg.capacity.baseline_daily_admissions * pool.hour_weights
    baseline_counts = rng.poisson(hourly_rate[np.arange(total) % 24])
    surge_counts = np.zeros(total, dtype="int64")
    if cfg.surge_patients:
        start = cfg.warmup_hours + min(cfg.surge_start_hour, cfg.horizon_hours - 1)
        duration = max(1, min(cfg.surge_duration_hours, total - start))
        surge_counts[start:start + duration] = rng.multinomial(cfg.surge_patients, np.full(duration, 1.0 / duration))
    baseline_idx = rng.integers(0, len(pool.ward_hours), size=int(baseline_counts.sum()))
    groups, shares = _surge_sampler(pool, mix)
    surge_group = rng.choice(len(groups), size=int(surge_counts.sum()), p=shares)
    surge_idx = np.array([groups[g][rng.integers(0, len(groups[g]))] for g in surge_group], dtype="int64")
    # Interleave per hour: baseline arrivals of hour t, then surge arrivals of hour t.
    order: list[int] = []
    b = s = 0
    for t in range(total):
        nb, ns = int(baseline_counts[t]), int(surge_counts[t])
        order.extend(baseline_idx[b:b + nb].tolist())
        order.extend(surge_idx[s:s + ns].tolist())
        b, s = b + nb, s + ns
    return baseline_counts, surge_counts, order


def _run_once(pool: PatientPool, cfg: SimulationConfig, mix: dict[int, float], rng: np.random.Generator) -> _RunTrace:
    """One replication. Beds are identical, so each unit is a heap of discharge times:
    a bed assigned at hour t for h hours is free again at hour t + ceil(h)."""
    total = cfg.warmup_hours + cfg.horizon_hours
    n_ward, n_icu = cfg.capacity.ward_beds, cfg.capacity.icu_beds
    icu_hours, ward_hours, priority = pool.icu_hours.tolist(), pool.ward_hours.tolist(), pool.priority.tolist()
    ward_busy: list[float] = []  # discharge times
    icu_busy: list[tuple[float, float]] = []  # (discharge time, ward hours after the ICU)
    ward_q: list[tuple[int, int, int, float]] = []  # (-priority, arrival_hour, seq, ward_hours)
    icu_q: list[tuple[int, int, int, float, float]] = []  # (-priority, arrival_hour, seq, icu_hours, ward_hours)
    admitted_at: dict[int, int] = {}  # seq -> arrival hour, for patients arriving after warm-up
    wait_hours: list[float] = []
    ward_trace, icu_trace, waiting_trace = [], [], []
    seq = 0

    baseline_counts, surge_counts, order = _sample_arrivals(pool, cfg, mix, rng, total)
    cursor = 0
    heappush, heappop = heapq.heappush, heapq.heappop

    for t in range(total):
        # 1. Discharges; finished ICU stays step down to the ward (ahead of new ward admissions).
        while icu_busy and icu_busy[0][0] <= t:
            _, ward_after = heappop(icu_busy)
            if ward_after > 0:
                seq += 1
                heappush(ward_q, (-_STEPDOWN_PRIORITY, t, seq, ward_after))
        while ward_busy and ward_busy[0] <= t:
            heappop(ward_busy)

        # 2. Arrivals of this hour.
        arrivals = int(baseline_counts[t] + surge_counts[t])
        for idx in order[cursor:cursor + arrivals]:
            seq += 1
            if t >= cfg.warmup_hours:
                admitted_at[seq] = t
            if icu_hours[idx] > 0:
                heappush(icu_q, (-priority[idx], t, seq, icu_hours[idx], ward_hours[idx]))
            else:
                heappush(ward_q, (-priority[idx], t, seq, max(ward_hours[idx], 1.0)))
        cursor += arrivals

        # 3. Bed assignment by priority, then arrival order.
        while icu_q and len(icu_busy) < n_icu:
            _, _, s, icu_h, ward_h = heappop(icu_q)
            heappush(icu_busy, (t + icu_h, ward_h))
            arrived = admitted_at.pop(s, None)
            if arrived is not None:
                wait_hours.append(float(t - arrived))
        while ward_q and len(ward_busy) < n_ward:
            _, _, s, ward_h = heappop(ward_q)
            heappush(ward_busy, t + ward_h)
            arrived = admitted_at.pop(s, None)
            if arrived is not None:
                wait_hours.append(float(t - arrived))

        if t >= cfg.warmup_hours:
            ward_trace.append(len(ward_busy))
            icu_trace.append(len(icu_busy))
            waiting_trace.append(len(ward_q) + len(icu_q))

    # Patients still waiting at the end count with their (censored) wait so far.
    wait_hours.extend(float(total - 1 - arrived) for arrived in admitted_at.values())
    horizon = slice(cfg.warmup_hours, total)
    return _RunTrace(
        ward=np.array(ward_trace, dtype="float64"),
        icu=np.array(icu_trace, dtype="float64"),
        waiting=np.array(waiting_trace, dtype="float64"),
        wait_hours=wait_hours,
        baseline_arrivals=int(baseline_counts[horizon].sum()),
        surge_arrivals=int(surge_counts[horizon].sum()),
    )


def _unit_stats(series: np.ndarray, capacity: int) -> UnitStats:
    peaks = series.max(axis=1)
    at_capacity = (series >= capacity).sum(axis=1)
    return UnitStats(
        capacity=capacity,
        mean_occupancy=round(float(series.mean()), 1),
        mean_utilization_pct=round(100.0 * float(series.mean()) / capacity, 1),
        peak_occupancy_mean=round(float(peaks.mean()), 1),
        peak_occupancy_p90=round(float(np.quantile(peaks, 0.9)), 1),
        hours_at_capacity_mean=round(float(at_capacity.mean()), 1),
        probability_reach_capacity_pct=round(100.0 * float((at_capacity > 0).mean()), 1),
    )


def _timeline(ward: np.ndarray, icu: np.ndarray, waiting: np.ndarray) -> list[TimelinePoint]:
    step = max(1, math.ceil(ward.shape[1] / MAX_TIMELINE_POINTS))
    return [
        TimelinePoint(
            hour=h,
            ward_mean=round(float(ward[:, h].mean()), 1),
            ward_p90=round(float(np.quantile(ward[:, h], 0.9)), 1),
            icu_mean=round(float(icu[:, h].mean()), 1),
            icu_p90=round(float(np.quantile(icu[:, h], 0.9)), 1),
            waiting_mean=round(float(waiting[:, h].mean()), 1),
        )
        for h in range(0, ward.shape[1], step)
    ]


def simulate(pool: PatientPool, cfg: SimulationConfig) -> SimulationResult:
    """Runs ``cfg.runs`` seeded replications and aggregates them. Deterministic for a given seed."""
    mix = cfg.surge_esi_mix or pool.observed_esi_mix()
    traces = [_run_once(pool, cfg, mix, np.random.default_rng([cfg.seed, run])) for run in range(cfg.runs)]
    ward = np.stack([t.ward for t in traces])
    icu = np.stack([t.icu for t in traces])
    waiting = np.stack([t.waiting for t in traces])
    waits = np.array([w for t in traces for w in t.wait_hours]) if any(t.wait_hours for t in traces) else np.zeros(1)

    ward_stats = _unit_stats(ward, cfg.capacity.ward_beds)
    icu_stats = _unit_stats(icu, cfg.capacity.icu_beds)
    # Sustained load of the most loaded unit (peaks of a small unit always touch capacity).
    worst = max(ward_stats.mean_utilization_pct, icu_stats.mean_utilization_pct) / 100.0
    max_waiting = waiting.max(axis=1)
    return SimulationResult(
        scenario={
            "surge_patients": cfg.surge_patients,
            "surge_start_hour": cfg.surge_start_hour,
            "surge_duration_hours": cfg.surge_duration_hours,
        },
        capacity=cfg.capacity,
        runs=cfg.runs,
        horizon_hours=cfg.horizon_hours,
        warmup_hours=cfg.warmup_hours,
        seed=cfg.seed,
        surge_esi_mix={str(k): round(v / sum(mix.values()), 3) for k, v in sorted(mix.items())},
        arrivals_mean={
            "baseline": round(float(np.mean([t.baseline_arrivals for t in traces])), 1),
            "surge": round(float(np.mean([t.surge_arrivals for t in traces])), 1),
        },
        ward=ward_stats,
        icu=icu_stats,
        waiting=QueueStats(
            max_waiting_mean=round(float(max_waiting.mean()), 1),
            max_waiting_p90=round(float(np.quantile(max_waiting, 0.9)), 1),
            mean_wait_hours_per_patient=round(float(waits.mean()), 2),
            patients_waited_pct=round(100.0 * float((waits > 0).mean()), 1),
        ),
        status_color=status_color(worst),
        timeline=_timeline(ward, icu, waiting),
        assumptions=ASSUMPTIONS,
    )
