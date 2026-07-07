"""
Startup scanner: discovers and loads all modules (skills, providers, channels).

No hot-reload — this runs once at startup.
"""

import os
import re
from pathlib import Path

import yaml

from bus.loaders.skill_loader import load_skill
from bus.loaders.provider_loader import load_provider
from bus.loaders.channel_loader import load_channel
from core.security import SecurityGuard

# Pattern for ${VAR_NAME} in config string values
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value):
    """Recursively resolve ${VAR_NAME} placeholders in a config value."""
    if isinstance(value, str):
        def _replace(match):
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                raise ValueError(
                    f"Environment variable '{var_name}' is not set, "
                    f"but is required by config value: {value}"
                )
            return env_val
        return _ENV_VAR_PATTERN.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(config_path: str | Path) -> dict:
    """
    Read config.yaml, resolve ${VAR_NAME} environment variable references.

    Returns the fully-resolved config dict.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _resolve_env_vars(raw)


def init_bus(config: dict) -> SecurityGuard:
    """
    Initialize the module bus:
    1. Create SecurityGuard from config.
    2. Load all providers from providers/.
    3. Load all skills from skills/.
    4. Load all channels from channels/.

    Returns the SecurityGuard instance (needed by skills).

    Logs each successful load; skips and warns on failures so a
    single broken skill does not crash the framework.
    """
    # -- Security guard --
    sec_cfg = config.get("security", {})
    security = SecurityGuard(
        allowed_dirs=sec_cfg.get("allowed_directories", ["./data/workspaces"]),
        command_blacklist=sec_cfg.get("command_blacklist", []),
        audit_log_path=sec_cfg.get("audit_log_path", "./logs/audit.log"),
    )
    print("[bus] SecurityGuard initialized")

    # -- Providers --
    providers_dir = Path("providers")
    if providers_dir.is_dir():
        for entry in sorted(providers_dir.iterdir()):
            if entry.is_dir() and (entry / "manifest.yaml").exists():
                try:
                    load_provider(entry, config)
                except Exception as e:
                    print(f"[bus] WARNING: Failed to load provider '{entry.name}': {e}")

    # -- Skills --
    skills_dir = Path("skills")
    if skills_dir.is_dir():
        for entry in sorted(skills_dir.iterdir()):
            if entry.is_dir() and (entry / "manifest.yaml").exists():
                try:
                    load_skill(entry, security_guard=security)
                except Exception as e:
                    print(f"[bus] WARNING: Failed to load skill '{entry.name}': {e}")

    # -- Channels --
    channels_dir = Path("channels")
    if channels_dir.is_dir():
        for entry in sorted(channels_dir.iterdir()):
            if entry.is_dir() and (entry / "manifest.yaml").exists():
                try:
                    load_channel(entry)
                except Exception as e:
                    print(f"[bus] WARNING: Failed to load channel '{entry.name}': {e}")

    print("[bus] Initialization complete")
    return security
