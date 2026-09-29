"""DigitalTwinService: the single entry point to the digital twin.

Pipeline (offline, run by ``python -m ai_module.digital_twin.prepare``):
    load_data -> preprocess_data -> build_state
Runtime (used by LangChain tools, LangGraph nodes and the API):
    get_current_state, simulate, predict, get_context

Training is NOT done here: ``predict`` loads the latest model produced by
``python -m ai_module.digital_twin.train``."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ai_module.config.logging import get_logger, log_event, timed
from ai_module.config.settings import AISettings
from ai_module.digital_twin.errors import DigitalTwinError, ModelNotTrainedError
from ai_module.digital_twin.features import CATEGORICAL_FEATURES, build_features
from ai_module.digital_twin.ingest import RawTables, download_dataset, load_raw_tables
from ai_module.digital_twin.preprocess import preprocess
from ai_module.digital_twin.profile import build_observed_profile
from ai_module.digital_twin.registry import LoadedModel, ModelRegistry
from ai_module.digital_twin.simulator import ASSUMPTIONS, PatientPool, SimulationConfig, simulate, size_capacity
from ai_module.digital_twin.sources import MIMIC_IV_DEMO, DatasetSpec
from ai_module.digital_twin.store import TwinDataStore
from ai_module.schemas.twin import (
    CapacityAssumptions,
    ModelInfo,
    PatientProfile,
    PredictionResult,
    Provenance,
    ScenarioInput,
    SimulationResult,
    TwinContext,
    TwinState,
)

logger = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DigitalTwinService:
    """Thread-safe for concurrent reads: lazily loaded data/model are guarded by a lock
    and every simulation uses its own seeded random generator."""

    def __init__(self, settings: AISettings, dataset: DatasetSpec = MIMIC_IV_DEMO,
                 store: TwinDataStore | None = None, registry: ModelRegistry | None = None) -> None:
        self.settings = settings
        self.dataset = dataset
        self.store = store or TwinDataStore(settings.processed_dir)
        self.registry = registry or ModelRegistry(settings.models_dir)
        self._lock = threading.RLock()
        self._frame: pd.DataFrame | None = None
        self._pool: PatientPool | None = None
        self._state: TwinState | None = None
        self._model: LoadedModel | None = None

    # ------------------------------------------------------------------ pipeline

    def load_data(self, download: bool = True) -> RawTables:
        """Ingests the public dataset (download + checksum verification) and reads the raw tables."""
        raw_dir = self.settings.raw_dir / self.dataset.local_dirname
        with timed(logger, "twin_pipeline", step="load_data", download=download):
            if download:
                raw_dir = download_dataset(self.dataset, self.settings.raw_dir)
            return load_raw_tables(raw_dir)

    def preprocess_data(self, raw: RawTables) -> pd.DataFrame:
        """Validates, cleans and transforms the raw tables and persists the result."""
        with timed(logger, "twin_pipeline", step="preprocess_data") as info:
            df, report = preprocess(raw)
            sha256 = self.store.save_processed(df, report.to_dict())
            self.store.save_provenance(Provenance(
                dataset=self.dataset.as_provenance(),
                processed_sha256=sha256,
                processed_rows=report.processed_rows,
                processed_patients=report.processed_patients,
                prepared_at=_utcnow(),
            ))
            info["rows"] = report.processed_rows
        with self._lock:
            self._frame, self._pool, self._state = df, None, None
        return df

    def _config_fingerprint(self) -> str:
        s = self.settings
        relevant = [s.twin_baseline_daily_admissions, s.twin_ward_beds, s.twin_icu_beds, s.twin_target_occupancy,
                    s.twin_sim_runs, s.twin_sim_horizon_hours, s.twin_sim_warmup_hours, s.twin_random_seed]
        return hashlib.sha256(json.dumps(relevant).encode()).hexdigest()[:16]

    def build_state(self) -> TwinState:
        """Builds the digital state (observed profile + modeled hospital + baseline run) and persists it."""
        with timed(logger, "twin_pipeline", step="build_state"):
            frame, pool = self._data()
            provenance = self.store.load_provenance()
            capacity = self._capacity(pool, self.settings.twin_baseline_daily_admissions, None, None)
            baseline = self._run(pool, ScenarioInput(), capacity)
            state = TwinState(
                built_at=_utcnow(),
                config_fingerprint=self._config_fingerprint(),
                provenance=provenance,
                observed=build_observed_profile(frame),
                capacity=capacity,
                baseline=baseline,
                status_color=baseline.status_color,
            )
            self.store.save_state(state)
        with self._lock:
            self._state = state
        return state

    # ------------------------------------------------------------------ runtime

    def _data(self) -> tuple[pd.DataFrame, PatientPool]:
        with self._lock:
            if self._frame is None:
                self._frame = self.store.load_processed()
            if self._pool is None:
                self._pool = PatientPool.from_frame(self._frame)
            return self._frame, self._pool

    def get_current_state(self) -> TwinState:
        """The persisted state; rebuilt when missing, built from other data, or built with other settings."""
        with self._lock:
            if self._state is not None:
                return self._state
            snapshot = self.store.load_state()
            if (snapshot is not None and snapshot.config_fingerprint == self._config_fingerprint()
                    and snapshot.provenance.processed_sha256 == self.store.processed_sha256()):
                self._state = snapshot
                return snapshot
        log_event(logger, "twin_state_rebuild", reason="missing_or_stale_snapshot")
        return self.build_state()

    def _capacity(self, pool: PatientPool, daily: float, ward: int | None, icu: int | None) -> CapacityAssumptions:
        s = self.settings
        ward = ward or s.twin_ward_beds
        icu = icu or s.twin_icu_beds
        sized_ward, sized_icu = size_capacity(pool, daily, s.twin_target_occupancy)
        return CapacityAssumptions(
            baseline_daily_admissions=daily,
            ward_beds=ward or sized_ward,
            icu_beds=icu or sized_icu,
            sizing_method="configured" if (ward and icu) else "littles_law",
            target_occupancy=s.twin_target_occupancy,
        )

    def _run(self, pool: PatientPool, scenario: ScenarioInput, capacity: CapacityAssumptions) -> SimulationResult:
        s = self.settings
        cfg = SimulationConfig(
            capacity=capacity,
            horizon_hours=scenario.horizon_hours or s.twin_sim_horizon_hours,
            warmup_hours=s.twin_sim_warmup_hours,
            runs=scenario.runs or s.twin_sim_runs,
            seed=s.twin_random_seed if scenario.seed is None else scenario.seed,
            surge_patients=scenario.surge_patients,
            surge_start_hour=scenario.surge_start_hour,
            surge_duration_hours=scenario.surge_duration_hours,
            surge_esi_mix=scenario.surge_esi_mix,
        )
        if cfg.surge_patients and cfg.surge_start_hour >= cfg.horizon_hours:
            raise DigitalTwinError("surge_start_hour must be inside the simulation horizon")
        return simulate(pool, cfg)

    def simulate(self, scenario: ScenarioInput) -> SimulationResult:
        """What-if simulation. Omitted capacity fields keep the modeled hospital's beds."""
        state = self.get_current_state()
        _, pool = self._data()
        with timed(logger, "twin_query", operation="simulate", surge_patients=scenario.surge_patients):
            capacity = CapacityAssumptions(
                baseline_daily_admissions=scenario.baseline_daily_admissions or state.capacity.baseline_daily_admissions,
                ward_beds=scenario.ward_beds or state.capacity.ward_beds,
                icu_beds=scenario.icu_beds or state.capacity.icu_beds,
                sizing_method="configured" if (scenario.ward_beds or scenario.icu_beds) else state.capacity.sizing_method,
                target_occupancy=state.capacity.target_occupancy,
            )
            return self._run(pool, scenario, capacity)

    def _loaded_model(self) -> LoadedModel:
        with self._lock:
            if self._model is None:
                self._model = self.registry.load()
                log_event(logger, "model_loaded", version=self._model.metadata["version"])
            return self._model

    def predict(self, patient: PatientProfile) -> PredictionResult:
        """ICU need and length of stay for one admission profile, from the trained model."""
        with timed(logger, "twin_query", operation="predict"):
            loaded = self._loaded_model()
            meta = loaded.metadata
            row = pd.DataFrame([patient.model_dump()])
            features = build_features(row)
            known: dict[str, list[str]] = meta.get("categories", {})
            unknown = {c: str(features.iloc[0][c]) for c in CATEGORICAL_FEATURES if known.get(c) and str(features.iloc[0][c]) not in known[c]}
            proba = float(loaded.models["icu_classifier"].predict_proba(features)[0, 1])
            los = max(float(loaded.models["los_regressor"].predict(features)[0]), 0.0)
            notes = [
                f"Model trained on {meta['split']['train_validation_admissions']} real admissions and scored on "
                f"{meta['split']['test_admissions']} held-out admissions of unseen patients.",
                "Indicative decision support only; not validated for clinical use.",
            ]
            if unknown:
                notes.append("Some categories were never seen in training and were treated as 'other': " + ", ".join(unknown))
            return PredictionResult(
                icu_probability=round(proba, 3),
                predicted_los_hours=round(los, 1),
                predicted_los_days=round(los / 24.0, 1),
                model=ModelInfo(
                    name=meta["name"], version=meta["version"], trained_at=meta["trained_at"],
                    dataset_sha256=meta["dataset_sha256"], test_metrics=meta["test_metrics"],
                ),
                unknown_categories=unknown,
                notes=notes,
            )

    def model_available(self) -> bool:
        try:
            self._loaded_model()
            return True
        except DigitalTwinError:
            return False

    def get_context(self, scenario: ScenarioInput | None = None, patient: PatientProfile | None = None,
                    include_timeline: bool = False) -> TwinContext:
        """Everything the reasoning layer may cite, grouped by origin (observed/simulated/predicted/assumption)."""
        with timed(logger, "twin_query", operation="get_context", scenario=scenario is not None, patient=patient is not None):
            state = self.get_current_state()
            warnings: list[str] = []
            timeline_excluded = None if include_timeline else {"timeline"}
            simulated: dict[str, Any] = {"baseline": state.baseline.model_dump(exclude=timeline_excluded)}
            if scenario is not None:
                simulated["scenario"] = self.simulate(scenario).model_dump(exclude=timeline_excluded)
            predicted = None
            if patient is not None:
                try:
                    predicted = self.predict(patient).model_dump()
                except (ModelNotTrainedError, DigitalTwinError) as exc:
                    warnings.append(str(exc))
            return TwinContext(
                observed=state.observed.model_dump(),
                simulated=simulated,
                predicted=predicted,
                assumptions={"capacity": state.capacity.model_dump(), "simulation": ASSUMPTIONS},
                provenance={**state.provenance.model_dump(), "state_built_at": state.built_at, "status_color": state.status_color},
                warnings=warnings,
            )
