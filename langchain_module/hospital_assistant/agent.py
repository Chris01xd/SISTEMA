"""The LangChain agent: an LLM that can only reach the data through `tools`."""
from __future__ import annotations

from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from hospital_assistant.config import Settings
from hospital_assistant.tools import build_tools

SYSTEM_PROMPT = """You are a data analyst assistant for a hospital emergency-management team. You answer \
questions about a table of hospital admissions (MIMIC-IV demo: real, de-identified, 100 patients).

Rules:
- Every number you state must come from a tool result in this conversation. Never estimate, recall or \
invent figures. If the tools cannot answer the question, say so plainly and explain what they can do.
- If a tool returns an error, fix the request and retry; do not apologise for it to the user.
- Always report how many admissions a figure is based on (`admissions_matched` / `admissions`) and warn \
when it is small: this is a 100-patient demo, so results are illustrative, not statistically representative.
- Dates are shifted by the data provider; only durations and hour of day are meaningful.
- You provide descriptive analytics for decision support. Do not give clinical advice or individual \
patient recommendations, and do not present these figures as real hospital performance.
- Answer in the language the user wrote in, concisely."""

_RECURSION_LIMIT = 12  # bounds the tool-calling loop


def build_model(settings: Settings) -> BaseChatModel:
    kwargs: dict[str, Any] = {"temperature": settings.temperature, "timeout": settings.timeout}
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    if ":" in settings.model:
        return init_chat_model(settings.model, **kwargs)
    return init_chat_model(settings.model, model_provider="anthropic", **kwargs)


def build_agent(df: pd.DataFrame, model: BaseChatModel | str | None = None, settings: Settings | None = None):
    if model is None:
        model = build_model(settings or Settings.from_env())
    return create_agent(model, tools=build_tools(df), system_prompt=SYSTEM_PROMPT)


def ask(agent, question: str, history: list[BaseMessage] | None = None) -> tuple[str, list[BaseMessage]]:
    """Runs one turn. Returns the answer and the full message list (pass it back as
    `history` for a follow-up question)."""
    messages = [*(history or []), ("user", question)]
    result = agent.invoke({"messages": messages}, config={"recursion_limit": _RECURSION_LIMIT})
    all_messages: list[BaseMessage] = result["messages"]
    final = next((m for m in reversed(all_messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    return (final.text if final else ""), all_messages


def tool_calls_made(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """The tool invocations in a conversation, for showing where an answer came from."""
    return [call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]
