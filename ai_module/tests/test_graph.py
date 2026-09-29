"""LangGraph: individual nodes and end-to-end runs (input -> graph -> twin -> chain -> response)."""
from __future__ import annotations

import logging

import pytest

from ai_module.chains import LLMAnalyst, LLMPlanner, RulePlanner, TemplateAnalyst
from ai_module.graphs import NODE_ORDER, GraphDependencies, initial_state
from ai_module.graphs import nodes
from ai_module.services.ai_service import AIModule

from .conftest import FailingChatModel, scripted, structured


def _deps(twin, settings, analyst=None, planner=None) -> GraphDependencies:
    return GraphDependencies(settings=settings, twin=twin, analyst=analyst or TemplateAnalyst(),
                             planner=planner or RulePlanner(), fallback_analyst=TemplateAnalyst())


def _state(message="¿Estado de la UCI?", parameters=None):
    return initial_state("req-1", message, parameters or {})


# ---------------------------------------------------------------- nodes in isolation


def test_validate_input_accepts_and_normalizes(prepared_twin, settings):
    update = nodes.validate_input(_state("  ¿Estado de la UCI?  "), _deps(prepared_twin, settings))
    assert update["validated_input"]["message"] == "¿Estado de la UCI?" and update["language"] == "es"


@pytest.mark.parametrize("message, parameters", [("", {}), ("   ", {}), ("x" * 5000, {}), ("ok", {"scenario": {"surge_patients": -3}}),
                                                  (None, {}), ("ok", {"bogus": 1})])
def test_validate_input_rejects(prepared_twin, settings, message, parameters):
    update = nodes.validate_input(_state(message, parameters), _deps(prepared_twin, settings))
    assert update["status"] == "rejected" and update["errors"][0]["fatal"]


def test_plan_prefers_explicit_parameters_over_the_planner(prepared_twin, settings):
    state = {**_state(), "validated_input": {"message": "hola", "parameters": {"scenario": {"surge_patients": 9}}}}
    planner = LLMPlanner(FailingChatModel(messages=iter([])), "fake")  # would fail if called
    update = nodes.plan_twin_queries(state, _deps(prepared_twin, settings, planner=planner))
    assert update["plan"]["intent"] == "simulate" and update["plan"]["source"] == "parameters"


def test_plan_falls_back_to_state_when_the_planner_fails(prepared_twin, settings):
    state = {**_state(), "validated_input": {"message": "hola", "parameters": {}}}
    update = nodes.plan_twin_queries(state, _deps(prepared_twin, settings, planner=LLMPlanner(FailingChatModel(messages=iter([])), "f")))
    assert update["plan"]["intent"] == "state" and not update["errors"][0]["fatal"]


def test_load_context_reports_unprepared_twin_as_failure(settings):
    from ai_module.digital_twin.service import DigitalTwinService

    state = {**_state(), "plan": {"intent": "state"}, "validated_input": {"message": "x", "parameters": {}}}
    update = nodes.load_digital_twin_context(state, _deps(DigitalTwinService(settings), settings))
    assert update["status"] == "failed" and "prepare" in update["errors"][0]["message"]


def test_reason_falls_back_when_the_llm_fails(prepared_twin, settings):
    state = {**_state(), "validated_input": {"message": "x", "parameters": {}},
             "digital_twin_context": prepared_twin.get_context().model_dump()}
    analyst = LLMAnalyst(FailingChatModel(messages=iter([])), "fake")
    update = nodes.reason_or_analyze(state, _deps(prepared_twin, settings, analyst=analyst))
    assert update["metadata"]["llm_fallback"] and update["model_output"]["answer"]
    assert "SECRET" not in str(update["errors"])


def test_instrument_turns_unexpected_exceptions_into_controlled_failures(prepared_twin, settings):
    def broken(state, deps):
        raise ZeroDivisionError("boom with sk-ant-secret-value")

    update = nodes.instrument("broken", broken, _deps(prepared_twin, settings))(_state())
    assert update["status"] == "failed" and "boom" not in str(update["errors"])
    assert update["execution_history"][0]["node"] == "broken"


# ---------------------------------------------------------------- end to end


def _module(twin, settings, *llm_messages, planner_messages=()):
    analyst = LLMAnalyst(scripted(*llm_messages), "fake:model") if llm_messages else TemplateAnalyst()
    planner = LLMPlanner(scripted(*planner_messages), "fake:model") if planner_messages else RulePlanner()
    return AIModule(settings, twin=twin, analyst=analyst, planner=planner)


def _grounded_analysis(twin, scenario) -> dict:
    run = twin.simulate(scenario)
    return {"answer": f"La UCI llegaría a {run.icu.mean_utilization_pct}% de utilización media.",
            "findings": [{"statement": f"Pico medio de UCI {run.icu.peak_occupancy_mean} de {run.icu.capacity} camas.", "source": "simulated"}],
            "recommendations": ["Preparar expansión de UCI."], "caveats": ["Muestra pequeña."]}


def test_end_to_end_with_llm_planner_and_grounded_analysis(prepared_twin, settings):
    from ai_module.schemas.twin import ScenarioInput

    scenario = {"surge_patients": 40, "surge_duration_hours": 4}
    module = _module(prepared_twin, settings,
                     structured("AnalysisOutput", _grounded_analysis(prepared_twin, ScenarioInput(**scenario))),
                     planner_messages=(structured("TwinQueryPlan", {"intent": "simulate", "scenario": scenario}),))
    result = module.invoke("¿Qué pasa si llegan 40 víctimas en 4 horas?")

    assert result.metadata["status"] == "completed" and result.metadata["grounded"] is True
    assert result.metadata["intent"] == "simulate" and result.metadata["plan_source"] == "fake:model"
    assert result.metadata["llm"]["used"] and not result.metadata["llm"]["fallback"]
    assert [h["node"] for h in result.metadata["execution_history"]] == list(NODE_ORDER)
    assert result.digital_twin["simulated"]["scenario"]["arrivals_mean"]["surge"] == 40
    assert result.digital_twin["observed"]["source"] == "observed"
    # Facts rendered by code are labeled apart from the LLM's interpretation.
    assert "## Datos observados" in result.response and "Interpretación (generada por LLM fake:model; no son datos)" in result.response


def test_ungrounded_answer_is_regenerated_with_feedback(prepared_twin, settings):
    from ai_module.schemas.twin import ScenarioInput

    invented = {"answer": "La UCI necesitará 999 camas.", "findings": [], "recommendations": [], "caveats": []}
    analyst_model = scripted(structured("AnalysisOutput", invented),
                             structured("AnalysisOutput", _grounded_analysis(prepared_twin, ScenarioInput(surge_patients=10))))
    module = AIModule(settings, twin=prepared_twin, analyst=LLMAnalyst(analyst_model, "fake"), planner=RulePlanner())
    result = module.invoke("Simula", {"scenario": {"surge_patients": 10}})
    assert result.metadata["status"] == "completed" and result.metadata["attempts"] == 2 and result.metadata["grounded"]
    correction_prompt = analyst_model.seen[1][-1].content
    assert "999" in correction_prompt and "only numbers present in the context" in correction_prompt
    assert [h["node"] for h in result.metadata["execution_history"]].count("reason_or_analyze") == 2


def test_persistently_ungrounded_answer_falls_back_to_deterministic_response(prepared_twin, settings):
    invented = {"answer": "Habrá 999 muertes.", "findings": [], "recommendations": [], "caveats": []}
    module = _module(prepared_twin, settings, structured("AnalysisOutput", invented), structured("AnalysisOutput", invented))
    result = module.invoke("Estado")
    assert result.metadata["status"] == "completed" and result.metadata["grounded"] is False
    assert result.metadata["llm"]["fallback"] and "999" not in result.response
    assert any("ungrounded: 999" in e["message"] for e in result.metadata["errors"])


def test_llm_failure_still_answers_deterministically(prepared_twin, settings):
    module = AIModule(settings, twin=prepared_twin, analyst=LLMAnalyst(FailingChatModel(messages=iter([])), "f"), planner=RulePlanner())
    result = module.invoke("Estado de la UCI")
    assert result.metadata["status"] == "completed" and result.metadata["llm"]["fallback"]
    assert "SECRET" not in result.model_dump_json()


def test_without_llm_the_full_flow_works(prepared_twin, settings):
    result = AIModule(settings, twin=prepared_twin).invoke("What is the ICU status of the hospital?")
    assert result.metadata["status"] == "completed" and not result.metadata["llm"]["configured"]
    assert "Observed data" in result.response and "deterministic rules" in result.response


def test_prediction_request_end_to_end(trained_twin, settings):
    result = AIModule(settings, twin=trained_twin).invoke("Paciente", {"patient": {"age": 80, "sex": "F", "ed_acuity_esi": 1}})
    assert result.metadata["intent"] == "predict" and result.digital_twin["predicted"]["source"] == "predicted"
    assert result.metadata["model_version"] and "## Predicción" in result.response


def test_invalid_input_is_rejected_without_running_the_twin(prepared_twin, settings):
    result = AIModule(settings, twin=prepared_twin).invoke("", {})
    assert result.metadata["status"] == "rejected" and result.digital_twin == {}
    assert [h["node"] for h in result.metadata["execution_history"]] == ["receive_input", "validate_input", "handle_error"]


def test_unprepared_twin_fails_cleanly(settings):
    result = AIModule(settings).invoke("Estado")
    assert result.metadata["status"] == "failed" and "prepare" in result.response


def test_logs_have_request_id_and_no_user_text(prepared_twin, settings, caplog):
    with caplog.at_level(logging.INFO, logger="ai_module"):
        result = AIModule(settings, twin=prepared_twin).invoke("TEXTO-PRIVADO-DEL-USUARIO")
    events = [r.structured for r in caplog.records if hasattr(r, "structured")]
    assert {e["event"] for e in events} >= {"graph_node", "twin_query", "ai_invoke"}
    assert all(e["request_id"] == result.request_id for e in events if e["event"] in ("graph_node", "ai_invoke"))
    assert "TEXTO-PRIVADO-DEL-USUARIO" not in caplog.text
