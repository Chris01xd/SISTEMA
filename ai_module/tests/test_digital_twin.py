"""Simulator, DigitalTwinService, training/evaluation and the model registry."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ai_module.digital_twin.errors import ModelNotTrainedError, TwinDataNotPreparedError
from ai_module.digital_twin.modeling import split_by_patient
from ai_module.digital_twin.preprocess import preprocess
from ai_module.digital_twin.service import DigitalTwinService
from ai_module.digital_twin.simulator import PatientPool, SimulationConfig, simulate, size_capacity, status_color
from ai_module.schemas.twin import CapacityAssumptions, PatientProfile, ScenarioInput

# ---------------------------------------------------------------- simulator


def _pool(raw_tables) -> PatientPool:
    df, _ = preprocess(raw_tables)
    return PatientPool.from_frame(df)


def _cfg(**overrides) -> SimulationConfig:
    base = dict(capacity=CapacityAssumptions(baseline_daily_admissions=10, ward_beds=80, icu_beds=30, sizing_method="configured",
                                             target_occupancy=0.8),
                horizon_hours=72, warmup_hours=200, runs=6, seed=7)
    return SimulationConfig(**{**base, **overrides})


def test_simulation_is_deterministic_for_a_seed(raw_tables):
    pool = _pool(raw_tables)
    assert simulate(pool, _cfg()).model_dump() == simulate(pool, _cfg()).model_dump()
    assert simulate(pool, _cfg()).model_dump() != simulate(pool, _cfg(seed=8)).model_dump()


def test_occupancy_never_exceeds_capacity_and_surge_adds_load(raw_tables):
    pool = _pool(raw_tables)
    base = simulate(pool, _cfg())
    surge = simulate(pool, _cfg(surge_patients=60, surge_duration_hours=3, surge_start_hour=5))
    for run in (base, surge):
        assert run.ward.peak_occupancy_p90 <= run.ward.capacity and run.icu.peak_occupancy_p90 <= run.icu.capacity
        assert all(p.icu_mean <= run.icu.capacity for p in run.timeline)
    assert surge.arrivals_mean["surge"] == 60
    assert surge.icu.mean_occupancy > base.icu.mean_occupancy
    assert surge.waiting.mean_wait_hours_per_patient >= base.waiting.mean_wait_hours_per_patient


def test_surge_mix_is_respected_and_defaults_to_the_observed_mix(raw_tables):
    pool = _pool(raw_tables)
    observed = pool.observed_esi_mix()
    assert sum(observed.values()) == pytest.approx(1.0)
    run = simulate(pool, _cfg(surge_patients=10, surge_esi_mix={1: 3, 2: 1}))
    assert run.surge_esi_mix == {"1": 0.75, "2": 0.25}


def test_littles_law_sizing(raw_tables):
    pool = _pool(raw_tables)
    ward, icu = size_capacity(pool, daily_admissions=24, target_occupancy=0.5)
    assert ward == int(np.ceil(pool.mean_ward_hours() / 0.5))
    assert icu == int(np.ceil(pool.mean_icu_hours() / 0.5))


@pytest.mark.parametrize("utilization, color", [(0.1, "GREEN"), (0.7, "YELLOW"), (0.85, "RED")])
def test_status_color_thresholds_match_the_backend(utilization, color):
    assert status_color(utilization) == color


# ---------------------------------------------------------------- service


def test_state_is_built_persisted_and_reloaded(prepared_twin, settings):
    state = prepared_twin.get_current_state()
    assert state.observed.admissions == 60 and state.observed.patients == 30
    assert state.provenance.dataset["license"].startswith("Open Database License")
    assert state.capacity.sizing_method == "littles_law"
    reloaded = DigitalTwinService(settings).get_current_state()
    assert reloaded.built_at == state.built_at  # snapshot reused, not rebuilt


def test_state_is_rebuilt_when_settings_change(prepared_twin, settings):
    before = prepared_twin.get_current_state()
    changed = settings.model_copy(update={"twin_baseline_daily_admissions": 30})
    after = DigitalTwinService(changed).get_current_state()
    assert after.config_fingerprint != before.config_fingerprint
    assert after.capacity.baseline_daily_admissions == 30


def test_unprepared_twin_raises_an_actionable_error(settings):
    with pytest.raises(TwinDataNotPreparedError, match="ai_module.digital_twin.prepare"):
        DigitalTwinService(settings).get_current_state()


def test_scenario_keeps_hospital_beds_unless_overridden(prepared_twin):
    state = prepared_twin.get_current_state()
    run = prepared_twin.simulate(ScenarioInput(surge_patients=20, baseline_daily_admissions=15))
    assert run.capacity.ward_beds == state.capacity.ward_beds
    assert run.capacity.baseline_daily_admissions == 15
    assert prepared_twin.simulate(ScenarioInput(icu_beds=3)).capacity.icu_beds == 3


def test_context_separates_sources(prepared_twin):
    ctx = prepared_twin.get_context(scenario=ScenarioInput(surge_patients=5))
    assert ctx.observed["source"] == "observed"
    assert ctx.simulated["baseline"]["source"] == ctx.simulated["scenario"]["source"] == "simulated"
    assert ctx.assumptions["capacity"]["source"] == "assumption"
    assert "timeline" not in ctx.simulated["scenario"]
    assert prepared_twin.get_context(include_timeline=True).simulated["baseline"]["timeline"]


def test_prediction_without_a_trained_model_is_a_warning_in_context(prepared_twin):
    patient = PatientProfile(age=70, sex="M", ed_acuity_esi=1)
    with pytest.raises(ModelNotTrainedError):
        prepared_twin.predict(patient)
    ctx = prepared_twin.get_context(patient=patient)
    assert ctx.predicted is None and "train" in ctx.warnings[0]


# ---------------------------------------------------------------- training / evaluation


def test_split_never_shares_patients(raw_tables):
    df, _ = preprocess(raw_tables)
    split = split_by_patient(df, seed=3)
    sets = [set(part["subject_id"]) for part in (split.train, split.validation, split.test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(p) for p in (split.train, split.validation, split.test)) == len(df)


def test_train_saves_versioned_artifact_with_metadata(trained_twin, settings):
    root = settings.models_dir / "admission_outcome"
    version = json.loads((root / "latest.json").read_text())["version"]
    meta = json.loads((root / version / "metadata.json").read_text())
    for key in ("version", "trained_at", "dataset_sha256", "params", "test_metrics", "split", "library_versions", "dataset"):
        assert key in meta
    assert meta["dataset_sha256"] == trained_twin.store.processed_sha256()
    assert "diagnosis" not in meta["features"] and "los_hours" not in meta["features"]  # no leakage


def test_prediction_uses_the_trained_model(trained_twin):
    result = trained_twin.predict(PatientProfile(age=70, sex="M", ed_acuity_esi=1, admission_type="NEW TYPE"))
    assert 0 <= result.icu_probability <= 1 and result.predicted_los_hours >= 0
    assert result.predicted_los_days == pytest.approx(result.predicted_los_hours / 24, abs=0.05)
    assert result.unknown_categories == {"admission_type": "NEW TYPE"}


def test_artifact_from_another_sklearn_version_is_refused_with_an_actionable_error(trained_twin, settings):
    from ai_module.digital_twin.errors import ModelArtifactError

    root = settings.models_dir / "admission_outcome"
    version = json.loads((root / "latest.json").read_text())["version"]
    meta_path = root / version / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["library_versions"]["scikit-learn"] = "0.0.1"
    meta_path.write_text(json.dumps(meta))
    fresh = DigitalTwinService(settings)
    with pytest.raises(ModelArtifactError, match="Retrain"):
        fresh.predict(PatientProfile(age=70, sex="M"))
    assert fresh.get_context(patient=PatientProfile(age=70, sex="M")).warnings  # graph degrades, does not crash
    assert fresh.model_available() is False


def test_evaluate_reproduces_training_metrics(trained_twin, settings, capsys):
    from ai_module.digital_twin import evaluate

    assert evaluate.main(["--bootstrap", "50"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["reproduces_training_metrics"] is True and report["warnings"] == []


def test_train_fails_cleanly_without_data(settings, capsys):
    from ai_module.digital_twin import train

    assert train.main([]) == 1
    assert "prepare" in capsys.readouterr().err


def test_prepare_cli_runs_offline_from_cached_raw_files(settings, raw_tables, capsys):
    from ai_module.digital_twin import prepare

    raw_dir = settings.raw_dir / "mimic-iv-demo-2.2"
    for module, name in (("hosp", "admissions"), ("hosp", "patients"), ("icu", "icustays"), ("ed", "edstays"), ("ed", "triage")):
        (raw_dir / module).mkdir(parents=True, exist_ok=True)
        getattr(raw_tables, name).to_csv(raw_dir / module / f"{name}.csv.gz", index=False)
    assert prepare.main(["--offline"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["processed_rows"] == 60
    assert pd.read_parquet(settings.processed_dir / "admissions.parquet").shape[0] == 60
