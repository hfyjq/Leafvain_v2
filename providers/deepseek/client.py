"""
DeepSeek LLM client — wraps the OpenAI SDK.

This is the ONLY module in the entire codebase that imports `openai`.
All other modules interact with the LLM through the Provider interface.
"""

from openai import OpenAI


class DeepSeekClient:
    """Unified LLM client for DeepSeek API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "<MODEL_NAME>",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """
        Send messages to the LLM and return the assistant response message.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            tools: Optional list of OpenAI function-calling tool definitions.

        Returns:
            dict with keys:
              - "role": "assistant"
              - "content": str | None  (None when tool_calls are present)
              - "tool_calls": list[dict] | None
        """
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

        result: dict = {"role": "assistant", "content": msg.content}

        if msg.tool_calls:
            result["tool_calls"] = [
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
