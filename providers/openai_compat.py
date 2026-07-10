"""
OpenAI-compatible provider base.

DeepSeek, GLM, Qwen, Moonshot, and many other Chinese LLM providers
use the OpenAI chat-completions API format.  This module provides a
shared implementation — individual providers only need to supply
their endpoint, model name, and tokenizer configuration.
"""

import tiktoken
from openai import OpenAI

from providers.base import BaseProvider, ChatResponse


class OpenAICompatProvider(BaseProvider):
    """
    Provider for any LLM that exposes an OpenAI-compatible
    ``/v1/chat/completions`` endpoint.

    Subclass and override the class-level constants to add a new provider.
    """

    # ------------------------------------------------------------------
    # Subclass overrides
    # ------------------------------------------------------------------

    # These MUST be set by subclasses (or via __init__ kwargs).
    _api_key: str
    _base_url: str
    _model: str
    _temperature: float
    _max_tokens: int
    _context_window: int
    _tokenizer_encoding: str = "cl100k_base"
    """tiktoken encoding name.  ``cl100k_base`` is a good approximation
       for most Chinese LLMs (DeepSeek, GLM, Qwen).  Override if the
       provider publishes its own encoding."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        context_window: int = 65536,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_window = context_window

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._tokenizer = tiktoken.get_encoding(self._tokenizer_encoding)

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    @property
    def context_window(self) -> int:
        return self._context_window

    def count_tokens(self, text: str) -> int:
        """Token count via tiktoken (cl100k_base by default)."""
        return len(self._tokenizer.encode(text))

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        """Send a chat-completion request and return a ChatResponse."""

        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # Extract token usage from the API response (REAL counts, not estimates)
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        result = ChatResponse(
            content=msg.content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        if msg.tool_calls:
            result.tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return result
