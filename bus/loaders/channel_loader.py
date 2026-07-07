"""
Load a channel from its manifest.yaml and register its run function.
"""

import importlib
from pathlib import Path

import yaml

from bus.registry import register_channel


def load_channel(channel_dir: Path) -> None:
    """
    1. Read <channel_dir>/manifest.yaml.
    2. Validate kind == "channel".
    3. Import the handler module, find run().
    4. Call register_channel(name, run_function).
    """
    manifest_path = channel_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Channel manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    if manifest.get("kind") != "channel":
        raise ValueError(
            f"Expected kind='channel' in {manifest_path}, got '{manifest.get('kind')}'"
        )

    name = manifest["name"]
    module_name = f"channels.{channel_dir.name}.handler"
    module = importlib.import_module(module_name)

    run_func = getattr(module, "run", None)
    if run_func is None:
        raise AttributeError(
            f"Channel '{name}': no 'run' function found in {module_name}.py"
        )

    register_channel(name, run_func)
    print(f"[bus] Channel loaded: {name}")
