"""
Provider abstract base class and ChatResponse.

All LLM providers MUST implement BaseProvider.  The framework never
imports a concrete provider directly — it only talks to BaseProvider.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChatResponse:
    """Structured return from a provider's chat() call."""

    content: str | None = None
    """The assistant's text response (None when tool_calls are present)."""

    tool_calls: list[dict] | None = None
    """Function-calling tool calls, if any."""

    prompt_tokens: int = 0
    """Tokens consumed by the input (messages + tools).  From API ``usage``."""

    completion_tokens: int = 0
    """Tokens consumed by the output.  From API ``usage``."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseProvider(ABC):
    """
    Abstract provider interface.

    Every LLM backend (DeepSeek, GLM, OpenAI, Ollama, ...) must
    subclass this and implement the three abstract members.
    """

    # ------------------------------------------------------------------
    # Abstract — must be implemented by every provider
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        """
        Send messages to the LLM and return a structured response.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            tools: Optional OpenAI-format tool definitions.

        Returns:
            ChatResponse with content, tool_calls, and token counts.
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in *text* using this provider's
        tokenizer (or a close approximation).

        Used for pre-flight checks (compression threshold, budget).
        """
        ...

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window size in tokens (e.g. 65536 for DeepSeek)."""
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def estimate_message_tokens(self, messages: list[dict]) -> int:
        """Quick estimate of total tokens across a message list."""
        total = 0
        for m in messages:
            content = m.get("content") or ""
            total += self.count_tokens(content)
            # Rough overhead per message (role marker, formatting)
            total += 4
        return total
