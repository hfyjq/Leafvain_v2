"""
Leafvain v2 — Entry point.

Responsibilities (and ONLY these):
  1. Load configuration.
  2. Initialize the module bus (providers, skills, channels).
  3. Initialize the memory manager.
  4. Get the active provider and channel.
  5. Start the interactive loop.
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

load_dotenv()

from bus.core import init_bus, load_config
from bus.registry import get_provider, get_channel
from core.memory.manager import MemoryManager


async def main() -> None:
    # 1. Load configuration
    config = load_config(_PROJECT_ROOT / "config.yaml")

    # 2. Initialize bus — loads all providers, skills, channels
    init_bus(config)

    # 3. Get the active provider
    provider_name = config.get("model", {}).get("provider", "deepseek")
    provider = get_provider(provider_name)

    # 4. Initialize memory manager
    memory_manager = None
    if config.get("memory", {}).get("enabled", True):
        memory_manager = MemoryManager(config, provider)
        print(f"[memory] MemoryManager initialized "
              f"(path={config.get('memory', {}).get('storage_path', 'data/memory')})")

    # 5. Get the CLI channel and start
    channel_name = config.get("channel", {}).get("default", "cli_channel")
    channel_run = get_channel(channel_name)
    await channel_run(provider, memory_manager=memory_manager)


if __name__ == "__main__":
    asyncio.run(main())
