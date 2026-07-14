"""
PipelineScheduler — sequential stage executor.

On startup, discovers all registered stages, sorts them by
``STAGES_ORDER``, and executes them in order for each incoming
:class:`~channels.base.MessageEvent`.
"""

from __future__ import annotations

from channels.base import MessageEvent
from core.pipeline.stage import Stage, StageContext, registered_stages

# Built-in stages in execution order.  Any registered stage not listed
# here runs after the built-in ones.
STAGES_ORDER = [
    "PreProcessStage",
    "AgentLoopStage",
    "PostProcessStage",
]


def _ensure_builtin_stages_imported() -> None:
    """Lazily import built-in stages so they self-register."""
    try:
        import core.pipeline.stages.pre_process  # noqa: F401
        import core.pipeline.stages.agent_loop  # noqa: F401
        import core.pipeline.stages.post_process  # noqa: F401
    except ImportError:
        pass  # Stages will be registered when their modules are imported


class PipelineScheduler:
    """Orchestrates the ordered execution of pipeline stages.

    Usage::

        scheduler = PipelineScheduler(session_pool, provider, config)
        await scheduler.initialize()

        async def on_event(event: MessageEvent) -> None:
            response = await scheduler.execute(event)
            # deliver response to user ...
    """

    def __init__(
        self,
        session_pool,
        provider,
        config: dict | None = None,
    ) -> None:
        _ensure_builtin_stages_imported()

        # Sort stages by STAGES_ORDER; unlisted stages go last
        def _sort_key(cls: type[Stage]) -> int:
            try:
                return STAGES_ORDER.index(cls.__name__)
            except ValueError:
                return len(STAGES_ORDER)

        self._stage_classes = sorted(registered_stages, key=_sort_key)
        self._stages: list[Stage] = []

        self._session_pool = session_pool
        self._provider = provider
        self._config = config or {}

    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        """Instantiate all registered stages."""
        self._stages = [cls() for cls in self._stage_classes]

    # ------------------------------------------------------------------
    async def execute(self, event: MessageEvent) -> str:
        """Run all stages against *event*.

        Returns the final response string (empty if none produced).
        """
        ctx = StageContext(
            session=event.session,
            user_message=event.message.content,
            message_chain=event.message.message_chain,
            # Shared dependencies available to all stages
            session_pool=self._session_pool,
            provider=self._provider,
            config=self._config,
        )

        for stage in self._stages:
            ctx = await stage.process(ctx)

            if ctx.metadata.get("stop"):
                break

        # Invoke the channel's reply callback
        if event.reply_callback:
            event.extras["response"] = ctx.response
            try:
                event.reply_callback(event)
            except Exception:
                pass

        return ctx.response
