"""
Load a provider from its manifest.yaml, instantiate its client,
validate it implements BaseProvider, and register it.
"""

import importlib
from pathlib import Path

import yaml

from bus.registry import register_provider
from providers.base import BaseProvider


def load_provider(provider_dir: Path, config: dict) -> None:
    """
    1. Read <provider_dir>/manifest.yaml.
    2. Validate kind == "provider".
    3. Dynamically import the client module, find the *Provider class.
    4. Instantiate with config values (falling back to manifest defaults).
    5. Validate it's a BaseProvider instance.
    6. Register via register_provider(name, instance).
    """
    manifest_path = provider_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Provider manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    if manifest.get("kind") != "provider":
        raise ValueError(
            f"Expected kind='provider' in {manifest_path}, "
            f"got '{manifest.get('kind')}'"
        )

    name = manifest["name"]
    manifest_cfg = manifest.get("config", {})

    # --- Import the provider class ---
    module_name = f"providers.{provider_dir.name}.client"
    module = importlib.import_module(module_name)

    # Find the *Provider class (by convention)
    provider_class = None
    for attr_name in dir(module):
        if attr_name.endswith("Provider") and not attr_name.startswith("_"):
            candidate = getattr(module, attr_name)
            if isinstance(candidate, type) and issubclass(candidate, BaseProvider):
                provider_class = candidate
                break

    if provider_class is None:
        raise AttributeError(
            f"No BaseProvider subclass found in {module_name}. "
            f"Expected a class named *Provider that inherits from BaseProvider."
        )

    # --- Merge config: manifest defaults < config.yaml overrides ---
    model_cfg = config.get("model", {})

    provider_instance = provider_class(
        api_key=model_cfg.get("api_key", ""),
        base_url=model_cfg.get("base_url", manifest_cfg.get("base_url", "")),
        model=model_cfg.get("model_name", manifest_cfg.get("default_model", "")),
        temperature=model_cfg.get("temperature", 0.3),
        max_tokens=model_cfg.get("max_tokens", 4096),
        context_window=model_cfg.get(
            "context_window",
            manifest_cfg.get("context_window", 65536),
        ),
    )

    # Safety check
    if not isinstance(provider_instance, BaseProvider):
        raise TypeError(
            f"Provider '{name}' ({provider_class.__name__}) "
            f"does not implement BaseProvider."
        )

    register_provider(name, provider_instance)
    print(
        f"[bus] Provider loaded: {name} "
        f"(model={provider_instance._model}, "
        f"window={provider_instance.context_window})"
    )
