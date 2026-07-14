"""
Pipeline Stage ABC + StageContext.

A Stage processes a :class:`~channels.base.MessageEvent`.  Stages are
executed in order by :class:`PipelineScheduler`.  Three built-in stages
are provided out of the box.

To add a custom stage, subclass ``Stage`` and call ``register_stage``:

    @register_stage
    class MyStage(Stage):
        async def process(self, ctx: StageContext) -> StageContext:
            ...
            return ctx
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from channels.base import MessageSession
    from channels.message import MessageChain


# ------------------------------------------------------------------
# Stage registry
# ------------------------------------------------------------------

registered_stages: list[type["Stage"]] = []


def register_stage(cls: type["Stage"]) -> type["Stage"]:
    """Decorator: register a Stage subclass for pipeline execution."""
    registered_stages.append(cls)
    return cls


# ------------------------------------------------------------------
# StageContext — data flowing through pipeline stages
# ------------------------------------------------------------------

@dataclass
class StageContext:
    """Mutable context passed between pipeline stages.

    Shared dependencies (session_pool, provider, config) are set by
    the PipelineScheduler and available to all stages.

    Per-turn data (session, messages, response, etc.) is populated
    by earlier stages and consumed by later ones.
    """

    # ---- set by scheduler before each pipeline run ----
    session: "MessageSession"
    user_message: str              # plain-text fallback (backward compat)
    session_pool: Any = None       # SessionPool
    provider: Any = None           # LLM provider client
    config: dict = field(default_factory=dict)

    # ---- rich content (set by scheduler from event.message.message_chain) ----
    message_chain: "MessageChain | None" = None

    # ---- populated by PreProcessStage ----
    messages: list[dict] = field(default_factory=list)
    memory_context: str = ""
    memory_manager: Any = None     # MemoryManager instance
    session_id: str = ""

    # ---- populated by AgentLoopStage ----
    response: str = ""
    tokens_used: int = 0

    # ---- populated by PostProcessStage ----
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Stage ABC
# ------------------------------------------------------------------

class Stage(ABC):
    """One step in the message-processing pipeline.

    Subclasses override ``process(ctx)``, optionally modifying *ctx*
    and returning it.
    """

    @abstractmethod
    async def process(self, ctx: StageContext) -> StageContext:
        """Process *ctx* and return it (possibly modified).

        If the stage determines that no further processing is needed
        it may set ``ctx.metadata["stop"] = True`` — the scheduler
        will skip remaining stages.
        """
        ...
