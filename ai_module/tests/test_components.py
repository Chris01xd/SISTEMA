"""The framework-agnostic component registry (future Langflow seam)."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai_module.components import COMPONENTS, describe_components, get_component
from ai_module.services.ai_service import AIModule


def test_every_component_is_described_with_json_schemas():
    described = describe_components()
    assert {c["name"] for c in described} == {c.name for c in COMPONENTS}
    assert {c["kind"] for c in described} == {"service", "tool", "chain", "graph", "prompt"}
    json.dumps(described)  # serializable, can be used to generate UI components


def test_components_run_through_the_same_services(prepared_twin, settings):
    module = AIModule(settings, twin=prepared_twin)
    state = get_component("digital_twin_state").build(module)({})
    assert state["observed"]["admissions"] == 60
    sim = get_component("digital_twin_simulate").build(module)({"surge_patients": 5})
    assert sim["source"] == "simulated"
    graph = get_component("twin_graph").build(module)({"message": "Estado"})
    assert graph["metadata"]["status"] == "completed"
    prompt = get_component("prompt").build(module)({"name": "analyst_system"})
    assert "digital twin" in prompt["text"]


def test_component_inputs_are_validated(prepared_twin, settings):
    with pytest.raises(ValidationError):
        get_component("digital_twin_simulate").build(AIModule(settings, twin=prepared_twin))({"surge_patients": -1})
    with pytest.raises(KeyError):
        get_component("nope")
