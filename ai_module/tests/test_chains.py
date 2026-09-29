"""LangChain layer: provider factory, chains (with a scripted model), tools and agent."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from ai_module.agents import ask_agent, build_twin_agent, build_twin_tools
from ai_module.chains import ChainError, LLMAnalyst, LLMPlanner, RulePlanner, TemplateAnalyst
from ai_module.config.settings import AISettings
from ai_module.services.llm_factory import LLMConfigurationError, build_chat_model, configuration_problem

from .conftest import FailingChatModel, scripted, structured

ANALYSIS = {"answer": "La UCI está al 80%.", "findings": [{"statement": "UCI 80%", "source": "simulated"}],
            "recommendations": ["Revisar altas"], "caveats": []}


def _input(context=None) -> dict:
    return {"question": "¿Estado?", "context": context or {}, "context_json": "{}", "correction_notes": "", "language": "es"}


def test_analysis_chain_returns_structured_output_and_uses_prompt_files():
    model = scripted(structured("AnalysisOutput", ANALYSIS))
    result = LLMAnalyst(model, "fake").analyze({**_input(), "context_json": '{"x": 1}'})
    assert result.answer == "La UCI está al 80%." and result.findings[0].source == "simulated"
    system, human = model.seen[0]
    assert "digital twin" in system.content and '{"x": 1}' in human.content and "¿Estado?" in human.content


def test_analysis_chain_errors_are_wrapped_without_leaking_details():
    with pytest.raises(ChainError) as info:
        LLMAnalyst(FailingChatModel(messages=iter([])), "fake").analyze(_input())
    assert "SECRET" not in str(info.value) and "RuntimeError" in str(info.value)


def test_planner_chain_extracts_only_validated_parameters():
    plan = LLMPlanner(scripted(structured("TwinQueryPlan", {"intent": "simulate", "scenario": {"surge_patients": 60, "surge_duration_hours": 4}})), "fake").plan("60 víctimas en 4 horas")
    assert plan.intent == "simulate" and plan.scenario.surge_patients == 60


def test_planner_rejects_out_of_range_values_from_the_llm():
    with pytest.raises(ChainError):
        LLMPlanner(scripted(structured("TwinQueryPlan", {"intent": "simulate", "scenario": {"surge_patients": -5}})), "fake").plan("x")


@pytest.mark.parametrize("text, intent", [("¿Qué pasa si llegan 50 víctimas?", "simulate"), ("Estado actual de la UCI", "state")])
def test_rule_planner(text, intent):
    assert RulePlanner().plan(text).intent == intent


def test_template_analyst_only_uses_context_values(prepared_twin):
    ctx = prepared_twin.get_context().model_dump()
    out = TemplateAnalyst().analyze({**_input(ctx)})
    assert str(ctx["observed"]["admissions"]) in out.findings[0].statement


# ---------------------------------------------------------------- provider factory


def test_provider_none_is_reported_not_built():
    settings = AISettings(_env_file=None, llm_provider="none")
    assert "disabled" in configuration_problem(settings)
    with pytest.raises(LLMConfigurationError):
        build_chat_model(settings)


def test_missing_key_is_a_configuration_problem_without_secrets():
    problem = configuration_problem(AISettings(_env_file=None, llm_provider="anthropic"))
    assert "ANTHROPIC_API_KEY" in problem


def test_unknown_provider_package_gives_install_hint(monkeypatch):
    monkeypatch.setenv("FAKEPROV_API_KEY", "x")
    with pytest.raises(LLMConfigurationError, match="Unsupported|langchain-"):
        build_chat_model(AISettings(_env_file=None, llm_provider="fakeprov", llm_model="m"))


def test_anthropic_model_is_built_from_settings_only():
    pytest.importorskip("langchain_anthropic")
    model = build_chat_model(AISettings(_env_file=None, llm_provider="anthropic", llm_model="claude-sonnet-5",
                                        llm_api_key="sk-ant-test", llm_timeout_seconds=17, llm_max_tokens=321))
    assert model.model == "claude-sonnet-5" and model.default_request_timeout == 17 and model.max_tokens == 321
    assert "sk-ant-test" not in repr(AISettings(_env_file=None, llm_api_key="sk-ant-test"))


# ---------------------------------------------------------------- tools / agent


def test_tools_return_service_values_with_source_labels(prepared_twin):
    tools = {t.name: t for t in build_twin_tools(prepared_twin)}
    assert set(tools) == {"get_digital_twin_state", "simulate_scenario", "predict_admission"}
    state = tools["get_digital_twin_state"].invoke({})
    assert state["observed"]["admissions"] == prepared_twin.get_current_state().observed.admissions
    sim = tools["simulate_scenario"].invoke({"surge_patients": 10})
    assert sim["source"] == "simulated" and sim["arrivals_mean"]["surge"] == 10
    assert "error" in tools["predict_admission"].invoke({"age": 60, "sex": "F"})  # no model trained: error, not crash


def test_agent_answers_through_tools(prepared_twin):
    model = scripted(
        AIMessage(content="", tool_calls=[{"name": "simulate_scenario", "args": {"surge_patients": 12}, "id": "t1", "type": "tool_call"}]),
        AIMessage(content="Listo."),
    )
    answer, messages = ask_agent(build_twin_agent(model, prepared_twin), "Simula 12 víctimas")
    assert answer == "Listo."
    tool_output = model.seen[1][-1].content
    assert '"surge": 12.0' in tool_output and '"source": "simulated"' in tool_output
