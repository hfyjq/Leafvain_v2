"""
PreProcessStage — runs before the LLM call.

Responsibilities:
  - Skip empty messages
  - Load / restore session from SessionPool
  - Inject memory context
  - Handle CLI commands (/clear, /compact)
"""

from __future__ import annotations

from core.pipeline.stage import Stage, StageContext, register_stage


@register_stage
class PreProcessStage(Stage):
    """Prepare context for the agent loop."""

    async def process(self, ctx: StageContext) -> StageContext:
        user_input = ctx.user_message.strip()

        # ---- empty message → stop pipeline ----
        if not user_input:
            ctx.metadata["stop"] = True
            return ctx

        pool = ctx.session_pool
        session_key = str(ctx.session)

        # ---- handle /clear ----
        if user_input.lower() == "/clear":
            new_sid, msgs = pool.new_session(session_key)
            ctx.session_id = new_sid
            ctx.messages = msgs
            ctx.metadata["stop"] = True
            ctx.response = (
                f"Session reset.\n"
                f"New session: {new_sid}\n"
                f"Profile and long-term facts preserved."
            )
            return ctx

        # ---- handle /compact ----
        if user_input.lower() == "/compact":
            sid, msgs = pool.get_or_create(session_key, ctx.provider)
            compacted, was_done = pool.compact(session_key, msgs)
            if was_done:
                stats = pool.last_compression or {}
                ctx.messages = compacted
                ctx.session_id = sid
                ctx.metadata["stop"] = True
                ctx.response = (
                    f"Context compacted: {len(msgs)} → {len(compacted)} messages. "
                    f"Summary: {stats.get('summary_chars', 0)} chars "
                    f"[{stats.get('source', 'llm')}]."
                )
            else:
                ctx.messages = msgs
                ctx.session_id = sid
                ctx.metadata["stop"] = True
                ctx.response = "Not enough turns to compact."
            return ctx

        # ---- normal message flow ----
        sid, msgs = pool.get_or_create(session_key, ctx.provider)
        ctx.session_id = sid
        ctx.messages = msgs
        ctx.memory_context = pool.get_context(
            session_key, user_input, messages=msgs,
        )
        ctx.memory_manager = pool._mgr

        return ctx
