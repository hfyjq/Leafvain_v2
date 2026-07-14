"""
AgentLoopStage — the core LLM interaction.

Calls the existing ``core.agent_loop.agent_loop()`` function with
context prepared by PreProcessStage.
"""

from __future__ import annotations

from core.agent_loop import agent_loop
from core.pipeline.stage import Stage, StageContext, register_stage


@register_stage
class AgentLoopStage(Stage):
    """Run the main agent loop (LLM + tool calls)."""

    async def process(self, ctx: StageContext) -> StageContext:
        tool_result_max_chars = ctx.config.get("tool_result_max_chars", 2000)

        response, updated_messages, session_chunks, tokens_used = agent_loop(
            user_message=ctx.user_message,
            provider=ctx.provider,
            messages=ctx.messages,
            session_chunks=[],  # TODO: per-session chunk storage
            memory_context=ctx.memory_context,
            tool_result_max_chars=tool_result_max_chars,
        )

        ctx.response = response
        ctx.messages = updated_messages
        ctx.tokens_used = tokens_used

        return ctx
