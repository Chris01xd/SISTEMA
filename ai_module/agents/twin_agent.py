"""Optional tool-calling agent: the LLM decides which twin tools to call.

The main flow (``graphs/``) is deterministic and does not need this agent; it exists for
conversational use and as a ready-made Langflow "Agent" building block."""
from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from ai_module.agents.tools import build_twin_tools
from ai_module.digital_twin.service import DigitalTwinService
from ai_module.prompts import load_prompt_text

RECURSION_LIMIT = 12  # bounds the tool-calling loop


def build_twin_agent(model: BaseChatModel, service: DigitalTwinService):
    return create_agent(model, tools=build_twin_tools(service), system_prompt=load_prompt_text("agent_system"))


def ask_agent(agent: Any, question: str, history: list[BaseMessage] | None = None) -> tuple[str, list[BaseMessage]]:
    """One conversational turn; returns the answer and the full message list for follow-ups."""
    result = agent.invoke({"messages": [*(history or []), ("user", question)]}, config={"recursion_limit": RECURSION_LIMIT})
    messages: list[BaseMessage] = result["messages"]
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    return (final.text if final else ""), messages
