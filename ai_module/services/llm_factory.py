"""Provider abstraction: builds a LangChain chat model from settings only.

Switching provider (Anthropic, OpenAI, Ollama, ...) is a configuration change
(``AI_LLM_PROVIDER`` / ``AI_LLM_MODEL`` + that provider's ``langchain-<provider>`` package);
chains, graph and service code never reference a concrete provider class."""
from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ai_module.config.settings import AISettings

# Environment variable each provider SDK reads when no explicit key is configured.
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
}
# Providers whose client does not take the generic kwargs below (local servers, mostly).
_LOCAL_PROVIDERS = {"ollama"}


class LLMConfigurationError(RuntimeError):
    """The configured provider cannot be used (package or credentials missing). Safe to show."""


def configuration_problem(settings: AISettings) -> str | None:
    """Why the configured LLM cannot be built, or None. Never includes secret values."""
    if not settings.llm_enabled:
        return "LLM disabled (AI_LLM_PROVIDER=none)"
    env_var = PROVIDER_KEY_ENV.get(settings.llm_provider)
    if env_var and not settings.llm_api_key and not os.environ.get(env_var):
        return f"{env_var} (or AI_LLM_API_KEY) is not set for provider '{settings.llm_provider}'"
    return None


def build_chat_model(settings: AISettings) -> BaseChatModel:
    """Returns the configured chat model. Raises ``LLMConfigurationError`` when it cannot be built."""
    problem = configuration_problem(settings)
    if problem:
        raise LLMConfigurationError(problem)
    kwargs: dict[str, Any] = {"temperature": settings.llm_temperature}
    if settings.llm_provider in _LOCAL_PROVIDERS:
        kwargs["num_predict"] = settings.llm_max_tokens
    else:
        kwargs.update(timeout=settings.llm_timeout_seconds, max_tokens=settings.llm_max_tokens, max_retries=settings.llm_max_retries)
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key.get_secret_value()
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    try:
        return init_chat_model(settings.llm_model, model_provider=settings.llm_provider, **kwargs)
    except ImportError as exc:
        raise LLMConfigurationError(
            f"Provider '{settings.llm_provider}' needs its LangChain integration package "
            f"(pip install langchain-{settings.llm_provider.replace('_', '-')})"
        ) from exc
    except ValueError as exc:
        raise LLMConfigurationError(f"Unsupported LLM configuration for provider '{settings.llm_provider}'") from exc
