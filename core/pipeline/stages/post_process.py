"""
PostProcessStage — runs after the LLM response.

Responsibilities:
  - Persist the turn to memory (record_turn)
  - Collect context-usage stats for display
  - Future: content safety check, metrics, logging
"""

from __future__ import annotations

from core.pipeline.stage import Stage, StageContext, register_stage


@register_stage
class PostProcessStage(Stage):
    """Persist conversation turn and collect stats."""

    async def process(self, ctx: StageContext) -> StageContext:
        pool = ctx.session_pool
        session_key = str(ctx.session)

        user_msg = {"role": "user", "content": ctx.user_message}
        assistant_msg = {"role": "assistant", "content": ctx.response}

        compressed = pool.record_turn(
            session_key=session_key,
            user_message=user_msg,
            assistant_message=assistant_msg,
            token_count=ctx.tokens_used,
            messages=ctx.messages,
        )

        if compressed is not None:
            ctx.messages = compressed
            ctx.metadata["compressed"] = True

        ctx.metadata["context_usage"] = pool.get_context_usage(
            session_key, ctx.messages,
        )
        ctx.metadata["last_compression"] = pool.last_compression

        return ctx
