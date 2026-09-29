"""Agent wiring, exercised with a scripted fake chat model (no API key, no network)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from hospital_assistant import ask, build_agent, load_dataset, tool_calls_made
from hospital_assistant.data import DatasetError
from hospital_assistant.tools import build_tools


class ScriptedModel(GenericFakeChatModel):
    """Plays back a fixed list of AIMessages; tool binding is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


def _script(*messages: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(messages))


def _call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_agent_runs_tool_and_returns_final_answer(admissions):
    model = _script(
        _call("aggregate_admissions", {"metric": "mortality_rate_pct", "group_by": "sex"}),
        AIMessage(content="Mortality is 50% in men and 25% in women (4 admissions each)."),
    )
    answer, messages = ask(build_agent(admissions, model=model), "Mortalidad por sexo?")

    assert answer.startswith("Mortality is 50%")
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert json.loads(tool_msg.content)["groups"]["M"]["value"] == 50.0  # produced by the real tool, not the model
    assert [c["name"] for c in tool_calls_made(messages)] == ["aggregate_admissions"]


def test_invalid_tool_arguments_are_reported_back_and_agent_recovers(admissions):
    model = _script(
        _call("aggregate_admissions", {"metric": "count", "group_by": "patient_id"}, "bad"),
        _call("aggregate_admissions", {"metric": "count", "group_by": "sex"}, "good"),
        AIMessage(content="4 women and 4 men."),
    )
    answer, messages = ask(build_agent(admissions, model=model), "How many by sex?")

    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error"  # schema rejected group_by='patient_id'; nothing executed
    assert "groups" in json.loads(tool_msgs[1].content)
    assert answer == "4 women and 4 men."


def test_tool_level_query_errors_are_returned_as_data(admissions):
    tools = {t.name: t for t in build_tools(admissions)}
    result = tools["aggregate_admissions"].invoke(
        {"metric": "count", "filters": [{"column": "age", "op": "in", "value": "not-a-list"}]}
    )
    assert "error" in result


def test_follow_up_questions_reuse_history(admissions):
    model = _script(AIMessage(content="first"), AIMessage(content="second"))
    agent = build_agent(admissions, model=model)
    _, history = ask(agent, "q1")
    answer, history = ask(agent, "q2", history)
    assert answer == "second" and sum(1 for m in history if m.type == "human") == 2


def test_agent_has_exactly_the_three_read_only_tools(admissions):
    assert sorted(t.name for t in build_tools(admissions)) == [
        "aggregate_admissions", "get_dataset_overview", "list_top_diagnoses",
    ]


def test_load_dataset_errors_are_actionable(tmp_path: Path, admissions):
    with pytest.raises(DatasetError, match="Import it first"):
        load_dataset(tmp_path / "missing.parquet")
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(bad)
    with pytest.raises(DatasetError, match="missing columns"):
        load_dataset(bad)
    good = tmp_path / "good.parquet"
    admissions.to_parquet(good)
    assert len(load_dataset(good)) == 8
