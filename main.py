"""
Leafvain v2 — Entry point.

Responsibilities (and ONLY these):
  1. Load configuration.
  2. Initialize the module bus (providers, skills, channels).
  3. Initialize MemoryManager → SessionPool.
  4. Build PipelineScheduler (PreProcess → AgentLoop → PostProcess).
  5. Get the active channel and run it.
"""

import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path so absolute imports work
# regardless of where the process is launched from.
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)

from bus.core import init_bus, load_config
from bus.registry import get_provider, get_channel
from core.memory.manager import MemoryManager
from core.pipeline.scheduler import PipelineScheduler
from core.prompt_assembler import set_project_root
from core.session_pool import SessionPool


async def main() -> None:
    # 1. Load configuration
    config = load_config(_PROJECT_ROOT / "config.yaml")

    # 1.5 Initialize prompt assembler with project root
    set_project_root(_PROJECT_ROOT)

    # 2. Get the active provider
    provider_name = config.get("model", {}).get("provider", "deepseek")

    # 3. Initialize memory manager (shared across all sessions)
    memory_manager = None
    if config.get("memory", {}).get("enabled", True):
        memory_manager = MemoryManager(config, provider=None)
        print(
            f"[memory] MemoryManager initialized "
            f"(path={config.get('memory', {}).get('storage_path', 'data/memory')})"
        )

    session_pool = SessionPool(memory_manager) if memory_manager else None

    # 4. Initialize bus — loads providers, skills, channels
    #    Channels get session_pool and scheduler injected later
    init_bus(config, session_pool=session_pool)

    # Now we can get the provider (registered by init_bus)
    provider = get_provider(provider_name)

    # Patch the provider into the memory manager (needed for compression)
    if memory_manager is not None:
        memory_manager._provider = provider

    # 5. Build PipelineScheduler
    scheduler = PipelineScheduler(
        session_pool=session_pool,
        provider=provider,
        config=config,
    )
    await scheduler.initialize()
    print("[pipeline] Scheduler initialized "
          f"({len(scheduler._stages)} stages)")

    # 6. Get the active channel, inject scheduler, and run
    channel_name = config.get("channel", {}).get("default", "cli_channel")
    channel = get_channel(channel_name)
    channel._scheduler = scheduler
    channel._pool = session_pool

    print(f"[main] Starting channel: {channel_name}")
    await channel.run()


if __name__ == "__main__":
    asyncio.run(main())
