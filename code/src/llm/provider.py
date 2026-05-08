"""
Unified LLM Provider — Abstract factory for multiple LLM backends.

Supported providers:
  - openai    : GPT-4o, GPT-4o-mini, etc. (via langchain-openai)
  - deepseek  : DeepSeek-V3, DeepSeek-R1, etc. (OpenAI-compatible API)
  - qwen      : Qwen-Max, Qwen-Turbo, etc. (OpenAI-compatible API via DashScope)
  - ollama    : Local models like Llama 3, Mistral, etc. (via langchain-ollama)

Design rationale:
  1. All agents call get_llm_provider() and get an object with an `invoke(messages)`
     method. They don't care whether it's OpenAI, DeepSeek, or Ollama underneath.
  2. Each concrete provider wraps the LangChain chat model for that vendor,
     normalizing initialization differences (api_key vs base_url vs local_url).
  3. The factory reads a single env var LLM_PROVIDER to decide which backend
     to instantiate, making switching as easy as changing one line in .env.
"""

from __future__ import annotations

import structlog
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage

from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Guarantees a uniform interface so agents remain provider-agnostic.
    """

    @abstractmethod
    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """
        Send a list of LangChain messages to the LLM and return the response.

        Args:
            messages: A list of SystemMessage / HumanMessage / AIMessage.

        Returns:
            An AIMessage-like object with a `.content` attribute.
        """
        ...


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT series (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.1, **kwargs: Any) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )
        logger.debug("openai_provider.initialized", model=model)

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        return self._llm.invoke(messages)


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek models (DeepSeek-V3, DeepSeek-R1, etc.).

    DeepSeek's API is fully OpenAI-compatible, so we reuse ChatOpenAI
    with a custom base_url.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.1,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            temperature=temperature,
            **kwargs,
        )
        logger.debug("deepseek_provider.initialized", model=model, base_url=base_url)

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        return self._llm.invoke(messages)


class QwenProvider(BaseLLMProvider):
    """
    Alibaba Qwen series (Qwen-Max, Qwen-Turbo, etc.).

    DashScope provides an OpenAI-compatible endpoint, so we reuse ChatOpenAI.
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.1,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            temperature=temperature,
            **kwargs,
        )
        logger.debug("qwen_provider.initialized", model=model, base_url=base_url)

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        return self._llm.invoke(messages)


class OllamaProvider(BaseLLMProvider):
    """
    Local models via Ollama (Llama 3, Mistral, Qwen-local, etc.).

    Uses langchain-ollama's ChatOllama. No API key required — the model
    runs on your own machine or a private server.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "langchain-ollama is required for Ollama provider. "
                "Install it with: pip install langchain-ollama"
            ) from exc

        self._llm = ChatOllama(
            model=model,
            base_url=base_url or self.DEFAULT_BASE_URL,
            temperature=temperature,
            **kwargs,
        )
        logger.debug("ollama_provider.initialized", model=model, base_url=base_url)

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        return self._llm.invoke(messages)


# ---------------------------------------------------------------------------
# Provider registry — maps provider name -> constructor
# ---------------------------------------------------------------------------
_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
    "ollama": OllamaProvider,
}


def get_llm_provider(temperature: float = 0.1, **kwargs: Any) -> BaseLLMProvider:
    """
    Factory function: create the appropriate LLM provider based on settings.

    Reads from environment / .env:
      - LLM_PROVIDER   : "openai" | "deepseek" | "qwen" | "ollama"
      - LLM_MODEL      : model name, e.g. "gpt-4o-mini", "deepseek-chat", "llama3"
      - LLM_API_KEY    : API key (not needed for ollama)
      - LLM_BASE_URL   : optional custom base URL

    Args:
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
                     Each agent can pass its own value.
        **kwargs:    Extra arguments passed straight to the underlying LangChain model.

    Returns:
        An instance of BaseLLMProvider ready for invoke().

    Raises:
        ValueError: If LLM_PROVIDER is unknown.
    """
    settings = get_settings()

    provider_name = settings.llm_provider.lower()
    model = settings.resolved_llm_model
    api_key = settings.resolved_llm_api_key
    base_url = settings.llm_base_url or None

    provider_cls = _PROVIDER_REGISTRY.get(provider_name)
    if provider_cls is None:
        supported = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Supported providers: {supported}"
        )

    logger.info(
        "llm_provider.selected",
        provider=provider_name,
        model=model,
        temperature=temperature,
    )

    init_kwargs: dict[str, Any] = {"temperature": temperature, **kwargs}

    if provider_name == "ollama":
        # Ollama doesn't need an API key
        init_kwargs["model"] = model
        if base_url:
            init_kwargs["base_url"] = base_url
        return OllamaProvider(**init_kwargs)

    # OpenAI-compatible providers need api_key + model
    init_kwargs["model"] = model
    init_kwargs["api_key"] = api_key
    if base_url:
        init_kwargs["base_url"] = base_url

    return provider_cls(**init_kwargs)
