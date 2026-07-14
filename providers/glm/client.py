"""
智谱 GLM provider — OpenAI-compatible API.

Thin subclass of OpenAICompatProvider.  Change ``config.yaml`` to
``provider: glm`` and set ``${GLM_API_KEY}`` to switch.
"""

from providers.openai_compat import OpenAICompatProvider


class GLMProvider(OpenAICompatProvider):
    """
    智谱 GLM LLM provider.

    Uses the OpenAI-compatible chat-completions endpoint at
    ``https://open.bigmodel.cn/api/paas/v4``.

    Token counting via tiktoken ``cl100k_base`` (GLM's actual tokenizer
    is similar; API response ``usage`` provides exact counts).
    """
    pass
