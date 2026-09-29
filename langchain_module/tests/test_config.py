from hospital_assistant.agent import build_model
from hospital_assistant.config import Settings


def test_build_model_applies_timeout_temperature_and_explicit_key(tmp_path):
    settings = Settings(dataset_path=tmp_path, model="claude-sonnet-5", temperature=0.0, timeout=12.0, api_key="sk-test")
    model = build_model(settings)
    assert type(model).__name__ == "ChatAnthropic"
    assert model.default_request_timeout == 12.0
    assert model.anthropic_api_key.get_secret_value() == "sk-test"


def test_api_key_never_appears_in_settings_repr(tmp_path):
    assert "sk-secret" not in repr(Settings(dataset_path=tmp_path, model="m", temperature=0, api_key="sk-secret"))


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "5")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    s = Settings.from_env()
    assert s.timeout == 5.0 and s.model == "claude-haiku-4-5-20251001" and s.api_key is None
