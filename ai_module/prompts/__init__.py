"""Prompts live as text files next to this module so they can be reviewed, versioned and
reused (e.g. by Langflow prompt components) without touching business logic."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).resolve().parent
PROMPT_NAMES = ("analyst_system", "analyst_user", "planner_system", "planner_user", "agent_system", "ml_explainer_system")


@lru_cache
def load_prompt_text(name: str) -> str:
    """Raw text of ``<name>.md``. Raises ``KeyError`` for unknown names."""
    if name not in PROMPT_NAMES:
        raise KeyError(f"Unknown prompt '{name}'. Available: {PROMPT_NAMES}")
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def prompt_version(name: str) -> str:
    """Short content hash, recorded in response metadata for traceability."""
    return hashlib.sha256(load_prompt_text(name).encode()).hexdigest()[:10]


def _escape(text: str) -> str:
    # System prompts are static: escape braces so they are never read as template variables.
    return text.replace("{", "{{").replace("}", "}}")


def analyst_prompt() -> ChatPromptTemplate:
    """Variables: question, context_json, correction_notes."""
    return ChatPromptTemplate.from_messages(
        [("system", _escape(load_prompt_text("analyst_system"))), ("human", load_prompt_text("analyst_user"))]
    )


def planner_prompt() -> ChatPromptTemplate:
    """Variables: question."""
    return ChatPromptTemplate.from_messages(
        [("system", _escape(load_prompt_text("planner_system"))), ("human", load_prompt_text("planner_user"))]
    )
