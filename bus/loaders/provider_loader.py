"""
Load a provider from its manifest.yaml, instantiate its client,
and register it in the global registry.
"""

import importlib
import os
from pathlib import Path

import yaml

from bus.registry import register_provider


def load_provider(provider_dir: Path, config: dict) -> None:
    """
    1. Read <provider_dir>/manifest.yaml.
    2. Validate kind == "provider".
    3. Dynamically import the client module.
    4. Instantiate the client with config values.
    5. Register via register_provider(name, instance).
    """
    manifest_path = provider_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Provider manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    if manifest.get("kind") != "provider":
        raise ValueError(
            f"Expected kind='provider' in {manifest_path}, got '{manifest.get('kind')}'"
        )

    name = manifest["name"]
    module_name = f"providers.{provider_dir.name}.client"
    module = importlib.import_module(module_name)

    # Find the client class by convention: <Name>Client or DeepSeekClient
    client_class = None
    for attr_name in dir(module):
        if attr_name.endswith("Client") and not attr_name.startswith("_"):
            client_class = getattr(module, attr_name)
            break

    if client_class is None:
        raise AttributeError(
            f"No *Client class found in {module_name}"
        )

    # Extract model config from the global config
    model_cfg = config.get("model", {})

    client_instance = client_class(
        api_key=model_cfg.get("api_key", ""),
        base_url=model_cfg.get("base_url", "<BASE_URL>"),
        model=model_cfg.get("model_name", "<MODEL_NAME>"),
        temperature=model_cfg.get("temperature", 0.3),
        max_tokens=model_cfg.get("max_tokens", 4096),
    )

    register_provider(name, client_instance)
    print(f"[bus] Provider loaded: {name}")
