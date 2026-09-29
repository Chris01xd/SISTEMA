"""Shared fixtures. Everything runs offline: raw tables are synthetic but follow the MIMIC
schema, and LLMs are replaced by a scripted chat model (no API calls, no cost)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ai_module.config.settings import AISettings
from ai_module.digital_twin.ingest import RawTables
from ai_module.digital_twin.service import DigitalTwinService


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Tests must not depend on the developer's AI_* variables or real API keys."""
    import os

    for key in list(os.environ):
        if key.startswith("AI_") or key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)


def make_raw_tables(n_patients: int = 30, seed: int = 0) -> RawTables:
    rng = np.random.default_rng(seed)
    subjects = np.arange(1000, 1000 + n_patients)
    patients = pd.DataFrame({
        "subject_id": subjects,
        "gender": np.where(np.arange(n_patients) % 2 == 0, "F", "M"),
        "anchor_age": rng.integers(20, 85, n_patients),
        "anchor_year": 2150,
        "anchor_year_group": "2011 - 2013",
        "dod": None,
    })
    rows, icu, ed, triage = [], [], [], []
    hadm, stay = 20000, 30000
    for subject in subjects:
        for _ in range(2):
            hadm += 1
            admit = pd.Timestamp("2150-01-01") + pd.Timedelta(hours=int(rng.integers(0, 24 * 300)))
            acuity = int(rng.integers(1, 4))
            los = float(rng.uniform(12, 300)) * (1.6 if acuity == 1 else 1.0)
            via_ed = rng.random() < 0.7
            rows.append({
                "subject_id": subject, "hadm_id": hadm, "admittime": str(admit), "dischtime": str(admit + pd.Timedelta(hours=los)),
                "deathtime": None, "admission_type": "EW EMER." if via_ed else rng.choice(["ELECTIVE", "URGENT"]),
                "admission_location": "EMERGENCY ROOM" if via_ed else "PHYSICIAN REFERRAL",
                "hospital_expire_flag": int(rng.random() < 0.08),
            })
            needs_icu = rng.random() < (0.8 if acuity == 1 else 0.45 if acuity == 2 else 0.15) if via_ed else rng.random() < 0.3
            if needs_icu:
                stay += 1
                icu.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "intime": str(admit), "outtime": str(admit),
                            "los": min(los / 24.0 * 0.5, 10.0)})
            if via_ed:
                stay += 1
                ed.append({"subject_id": subject, "hadm_id": float(hadm), "stay_id": stay, "intime": str(admit - pd.Timedelta(hours=3)),
                           "outtime": str(admit), "arrival_transport": rng.choice(["AMBULANCE", "WALK IN"])})
                triage.append({"subject_id": subject, "stay_id": stay, "temperature": float(rng.normal(98.2, 0.8)),
                               "heartrate": float(rng.normal(90, 15)), "resprate": 18.0, "o2sat": 97.0,
                               "sbp": float(rng.normal(130, 20)), "dbp": 75.0, "pain": "0", "acuity": float(acuity),
                               "chiefcomplaint": "x"})
    return RawTables(
        admissions=pd.DataFrame(rows), patients=patients, icustays=pd.DataFrame(icu),
        edstays=pd.DataFrame(ed), triage=pd.DataFrame(triage),
    )


@pytest.fixture
def raw_tables() -> RawTables:
    return make_raw_tables()


@pytest.fixture
def settings(tmp_path: Path) -> AISettings:
    return AISettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        llm_provider="none",
        twin_sim_runs=8,
        twin_sim_horizon_hours=72,
        twin_sim_warmup_hours=240,
        twin_baseline_daily_admissions=10,
    )


@pytest.fixture
def prepared_twin(settings: AISettings, raw_tables: RawTables) -> DigitalTwinService:
    service = DigitalTwinService(settings)
    service.preprocess_data(raw_tables)
    service.build_state()
    return service


@pytest.fixture
def trained_twin(prepared_twin: DigitalTwinService) -> DigitalTwinService:
    from ai_module.digital_twin import train

    assert train.main([]) == 0  # uses get_settings(): patched below via the settings env
    return prepared_twin


@pytest.fixture(autouse=True)
def _cli_settings(monkeypatch, settings: AISettings):
    """CLIs call get_settings(); point it at the test settings."""
    import ai_module.config.settings as settings_module
    from ai_module.digital_twin import evaluate, prepare, train

    for module in (settings_module, train, evaluate, prepare):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    for module in (train, evaluate, prepare):  # keep log records flowing to pytest's caplog
        monkeypatch.setattr(module, "configure_logging", lambda *a, **k: None)


class ScriptedChatModel(GenericFakeChatModel):
    """Plays back scripted AIMessages and records every prompt it receives."""

    seen: list = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class FailingChatModel(ScriptedChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("provider exploded sk-ant-SECRET-VALUE-123")


def structured(schema_name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    """What a provider returns for ``with_structured_output(schema)`` (a forced tool call)."""
    return AIMessage(content="", tool_calls=[{"name": schema_name, "args": args, "id": call_id, "type": "tool_call"}])


def scripted(*messages: AIMessage) -> ScriptedChatModel:
    model = ScriptedChatModel(messages=iter(messages))
    model.seen = []
    return model


@pytest.fixture(scope="session")
def ml_analysis(tmp_path_factory):
    """One real end-to-end model-comparison run (tuning + nested CV + tests), shared by the
    read-only ML tests. Returns (service, output)."""
    from ai_module.ml.config import MLSettings
    from ai_module.ml.service import MLAnalysisService

    root = tmp_path_factory.mktemp("ml_session")
    ai = AISettings(_env_file=None, data_dir=root / "data", artifacts_dir=root / "artifacts", llm_provider="none",
                    twin_sim_runs=3, twin_sim_horizon_hours=24, twin_sim_warmup_hours=24)
    ml = MLSettings(_env_file=None, runs_dir=root / "artifacts" / "ml_runs", explain_with_llm=False, bootstrap_resamples=200)
    DigitalTwinService(ai).preprocess_data(make_raw_tables(n_patients=60))
    service = MLAnalysisService(ai, ml)
    return service, service.run({"dataset": "mimic_icu"})
