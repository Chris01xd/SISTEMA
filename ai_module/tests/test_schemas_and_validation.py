"""Schemas, numeric grounding, language detection and log redaction."""
from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from ai_module.config.logging import log_event, redact, reset_request_id, set_request_id
from ai_module.schemas.api import InvokeRequest
from ai_module.schemas.twin import PatientProfile, ScenarioInput
from ai_module.services.grounding import ungrounded_numbers
from ai_module.services.language import detect_language


@pytest.mark.parametrize(
    "payload",
    [
        {"surge_patients": -1},
        {"surge_patients": 5000},
        {"surge_esi_mix": {7: 1.0}},
        {"surge_esi_mix": {1: 0.0}},
        {"icu_beds": 0},
        {"unknown_field": 1},
    ],
)
def test_scenario_rejects_invalid_values(payload):
    with pytest.raises(ValidationError):
        ScenarioInput.model_validate(payload)


def test_scenario_accepts_string_esi_keys_from_json():
    assert ScenarioInput.model_validate({"surge_esi_mix": {"1": 0.5, "2": 0.5}}).surge_esi_mix == {1: 0.5, 2: 0.5}


def test_patient_profile_validation():
    with pytest.raises(ValidationError):
        PatientProfile(age=150, sex="M")
    with pytest.raises(ValidationError, match="arrived_via_ed"):
        PatientProfile(age=50, sex="F", arrived_via_ed=False, ed_acuity_esi=2)
    with pytest.raises(ValidationError):
        PatientProfile(age=50, sex="X")


def test_invoke_request_forbids_unknown_parameters():
    with pytest.raises(ValidationError):
        InvokeRequest.model_validate({"message": "hi", "parameters": {"scenaro": {}}})


# ---------------------------------------------------------------- grounding


CONTEXT = {"icu": {"mean_utilization_pct": 86.4, "capacity": 47}, "icu_probability": 0.412, "los_hours": 48.0,
           "esi_distribution": {"1": 17}}


@pytest.mark.parametrize(
    "text",
    [
        "La UCI llega a 86.4% con 47 camas.",
        "La UCI llega al 86% (redondeado).",
        "Probabilidad de UCI 41.2%.",  # fraction shown as a percent
        "Estancia de 2 días.",  # 48 hours as days
        "86,4 % en notación española.",
        "ESI 1 representa 17 admisiones.",
    ],
)
def test_grounded_numbers_pass(text):
    assert ungrounded_numbers([text], CONTEXT) == []


def test_invented_numbers_are_detected():
    assert ungrounded_numbers(["La UCI tendrá 63 camas y 12.5% de mortalidad."], CONTEXT) == ["12.5", "63"]


def test_user_input_numbers_are_allowed():
    assert ungrounded_numbers(["Con 80 víctimas en 4 horas..."], CONTEXT, {"message": "¿80 víctimas en 4 horas?"}) == []


def test_language_detection():
    assert detect_language("¿Cuál es el estado de la UCI?") == "es"
    assert detect_language("What is the ICU status of the hospital?") == "en"


# ---------------------------------------------------------------- logging


def test_redaction_of_secrets():
    payload = {"api_key": "abc", "nested": {"note": "key sk-ant-abcdefghijkl leaked", "Authorization": "Bearer x"}}
    assert redact(payload) == {"api_key": "[REDACTED]", "nested": {"note": "key [REDACTED] leaked", "Authorization": "[REDACTED]"}}


def test_log_events_carry_request_id(caplog):
    logger = logging.getLogger("ai_module.test")
    token = set_request_id("req-123")
    try:
        with caplog.at_level(logging.INFO, logger="ai_module"):
            log_event(logger, "something", size=3, token="secret-token")
    finally:
        reset_request_id(token)
    record = caplog.records[-1]
    assert record.structured == {"event": "something", "request_id": "req-123", "size": 3, "token": "[REDACTED]"}
