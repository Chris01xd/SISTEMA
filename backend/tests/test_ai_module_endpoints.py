"""/ai/* and /digital-twin/* endpoints: wiring to the standalone ai_module.

Offline: the twin is prepared from synthetic MIMIC-shaped tables in a temp dir, and no real
LLM is called (deterministic mode, or a scripted chat model)."""
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("langgraph", reason="optional: pip install -r ../ai_module/requirements.txt")
pytest.importorskip("sklearn")

from app.services import ai_module_service  # noqa: E402

ai_module_service._import("ai_module")  # puts the repo root on sys.path (the module lives outside backend/)

from ai_module.config.settings import AISettings  # noqa: E402
from ai_module.digital_twin import train  # noqa: E402
from ai_module.digital_twin.service import DigitalTwinService  # noqa: E402
from ai_module.services.ai_service import AIModule  # noqa: E402
from ai_module.tests.conftest import FailingChatModel, make_raw_tables  # noqa: E402


def _settings(tmp_path) -> AISettings:
    return AISettings(_env_file=None, data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts", llm_provider="none",
                      twin_sim_runs=5, twin_sim_horizon_hours=48, twin_sim_warmup_hours=120, twin_baseline_daily_admissions=10)


@pytest.fixture
def env(monkeypatch, tmp_path):
    from app.authentication.dependencies import get_current_user
    from app.main import app

    settings = _settings(tmp_path)
    twin = DigitalTwinService(settings)
    twin.preprocess_data(make_raw_tables())
    twin.build_state()

    admin = SimpleNamespace(id=uuid.uuid4(), role=SimpleNamespace(permissions={"*": True}), is_active=True)
    monkeypatch.setitem(app.dependency_overrides, get_current_user, lambda: admin)
    monkeypatch.setattr(ai_module_service, "_instance", AIModule(settings, twin=twin))
    return SimpleNamespace(client=TestClient(app), app=app, settings=settings, twin=twin)


def _train(monkeypatch, settings):
    monkeypatch.setattr(train, "get_settings", lambda: settings)
    monkeypatch.setattr(train, "configure_logging", lambda *a, **k: None)
    assert train.main([]) == 0


def test_status(env):
    body = env.client.get("/api/v1/ai/status").json()
    assert body["available"] is True and body["digital_twin"]["ready"] is True
    assert body["llm"]["ready"] is False and "disabled" in body["llm"]["detail"]


def test_invoke_runs_the_graph_end_to_end(env):
    r = env.client.post("/api/v1/ai/invoke", json={"message": "¿Qué pasa si llegan 30 víctimas?",
                                                   "parameters": {"scenario": {"surge_patients": 30, "surge_duration_hours": 3}}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"request_id", "response", "digital_twin", "analysis", "metadata"}
    assert body["metadata"]["status"] == "completed" and body["metadata"]["intent"] == "simulate"
    assert body["digital_twin"]["simulated"]["scenario"]["arrivals_mean"]["surge"] == 30
    assert body["digital_twin"]["observed"]["source"] == "observed"
    assert "## Datos observados" in body["response"]


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 2001},
        {"message": "ok", "parameters": {"scenario": {"surge_patients": -1}}},
        {"message": "ok", "parameters": {"unknown": 1}},
    ],
)
def test_invoke_validation(env, payload):
    assert env.client.post("/api/v1/ai/invoke", json=payload).status_code == 422


def test_llm_failure_is_not_leaked_to_the_client(env, monkeypatch):
    from ai_module.chains import LLMAnalyst, RulePlanner

    module = AIModule(env.settings, twin=env.twin, analyst=LLMAnalyst(FailingChatModel(messages=iter([])), "fake"), planner=RulePlanner())
    monkeypatch.setattr(ai_module_service, "_instance", module)
    r = env.client.post("/api/v1/ai/invoke", json={"message": "Estado"})
    assert r.status_code == 200 and r.json()["metadata"]["llm"]["fallback"] is True
    assert "SECRET" not in r.text and "Traceback" not in r.text


def test_state_and_simulate(env):
    state = env.client.get("/api/v1/digital-twin/state")
    assert state.status_code == 200 and state.json()["observed"]["admissions"] == 60
    sim = env.client.post("/api/v1/digital-twin/simulate", json={"surge_patients": 20, "icu_beds": 12})
    assert sim.status_code == 200, sim.text
    assert sim.json()["capacity"]["icu_beds"] == 12 and "timeline" not in sim.json()
    with_timeline = env.client.post("/api/v1/digital-twin/simulate?include_timeline=true", json={})
    assert with_timeline.json()["timeline"]


@pytest.mark.parametrize("scenario", [{"surge_patients": -1}, {"surge_esi_mix": {"9": 1}}, {"bogus": True}])
def test_simulate_validation_is_422_with_field_errors(env, scenario):
    r = env.client.post("/api/v1/digital-twin/simulate", json=scenario)
    assert r.status_code == 422 and isinstance(r.json()["detail"], list)


def test_predict_requires_a_trained_model(env, monkeypatch):
    patient = {"age": 72, "sex": "M", "ed_acuity_esi": 2}
    r = env.client.post("/api/v1/digital-twin/predict", json=patient)
    assert r.status_code == 503 and "ai_module.digital_twin.train" in r.json()["detail"]
    _train(monkeypatch, env.settings)
    r = env.client.post("/api/v1/digital-twin/predict", json=patient)
    assert r.status_code == 200 and r.json()["source"] == "predicted"
    assert env.client.post("/api/v1/digital-twin/predict", json={"age": 500, "sex": "M"}).status_code == 422


def test_unprepared_twin_is_503(env, monkeypatch, tmp_path):
    empty = _settings(tmp_path / "empty")
    monkeypatch.setattr(ai_module_service, "_instance", AIModule(empty))
    assert env.client.get("/api/v1/digital-twin/state").status_code == 503
    r = env.client.post("/api/v1/ai/invoke", json={"message": "Estado"})
    assert r.status_code == 503 and "prepare" in r.json()["detail"]


def test_module_not_installed_is_501(env, monkeypatch):
    def missing():
        raise ai_module_service.HTTPException(status_code=501, detail="AI module is not available (langgraph)")

    monkeypatch.setattr(ai_module_service, "get_module", missing)
    assert env.client.get("/api/v1/digital-twin/state").status_code == 501
    assert env.client.post("/api/v1/ai/invoke", json={"message": "x"}).status_code == 501
    assert env.client.get("/api/v1/ai/status").json()["available"] is False


def test_permissions(env):
    from app.authentication.dependencies import get_current_user

    viewer = SimpleNamespace(id=uuid.uuid4(), role=SimpleNamespace(permissions={}), is_active=True)
    env.app.dependency_overrides[get_current_user] = lambda: viewer
    assert env.client.post("/api/v1/ai/invoke", json={"message": "x"}).status_code == 403
    assert env.client.post("/api/v1/digital-twin/simulate", json={}).status_code == 403
    assert env.client.get("/api/v1/digital-twin/state").status_code == 200  # login is enough to read


def test_backend_key_is_forwarded_only_when_the_module_has_none(monkeypatch, tmp_path):
    from app.core.config import settings as backend_settings
    from ai_module.config import settings as ai_settings_module

    monkeypatch.setattr(ai_module_service, "_instance", None)
    # This test exercises the Anthropic fallback, independent of local provider
    # settings and credentials in ai_module/.env or the shell environment.
    monkeypatch.setattr(ai_settings_module, "AISettings", lambda: AISettings(
        _env_file=None, data_dir=tmp_path / "data", llm_provider="anthropic",
        llm_model="claude-sonnet-5", llm_api_key=None,
    ))
    monkeypatch.setattr(backend_settings, "ANTHROPIC_API_KEY", "sk-ant-backend-key")
    monkeypatch.setenv("AI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Environment variables win over ai_module/.env: isolate from the developer's local LLM config.
    monkeypatch.setenv("AI_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("AI_LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("AI_LLM_API_KEY", "")
    module = ai_module_service.get_module()
    assert module.settings.llm_api_key.get_secret_value() == "sk-ant-backend-key"
    assert module.analyst.uses_llm  # ChatAnthropic built offline from settings only
    ai_module_service.reset()
