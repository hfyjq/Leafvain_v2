"""
In-memory singleton registry for tools, providers, and channels.

Three namespaces:
  - tools:       tool_name -> callable handler
  - tool_schemas: list of JSON Schema dicts (for LLM function calling)
  - providers:   provider_name -> client instance
  - channels:    channel_name -> handler callable (async run function)
"""

from typing import Any, Callable

_registry: dict[str, dict] = {
    "tools": {},
    "tool_schemas": [],
    "providers": {},
    "channels": {},
}


def register_tool(name: str, handler: Callable, schema: dict) -> None:
    """Register a tool with its handler function and JSON Schema definition."""
    _registry["tools"][name] = handler
    # Build the OpenAI function-calling format
    tool_def = {
        "type": "function",
        "function": {
            "name": name,
            "description": schema.get("description", ""),
            "parameters": schema.get("inputSchema", {}),
        },
    }
    _registry["tool_schemas"].append(tool_def)


def get_tool_schemas() -> list[dict]:
    """Return all registered tool schemas for LLM function calling."""
    return _registry["tool_schemas"]


def get_tool_summaries() -> list[str]:
    """Return name: one-line-description strings for each tool.

    Used by the prompt assembler to generate the <available_tools>
    section — the model sees tool names and short descriptions in
    the system prompt, while full JSON Schemas are sent via the
    ``tools`` parameter of the chat API.
    """
    summaries: list[str] = []
    for tool_def in _registry["tool_schemas"]:
        name = tool_def["function"]["name"]
        desc = tool_def["function"].get("description", "")
        # Collapse multi-line descriptions into a single line
        one_liner = " ".join(desc.split())
        summaries.append(f"{name}: {one_liner}")
    return summaries


def get_tool_schemas_filtered(names: list[str] | None = None) -> list[dict]:
    """Return tool schemas, optionally filtered to the given names.

    When *names* is None, behaves identically to :func:`get_tool_schemas`.
    When a list of names is provided, only schemas matching those names
    are returned (future lazy-load path).
    """
    if names is None:
        return _registry["tool_schemas"]
    return [
        td for td in _registry["tool_schemas"]
        if td["function"]["name"] in names
    ]


def get_tool_handler(name: str) -> Callable | None:
    """Return the handler callable for a given tool name, or None."""
    return _registry["tools"].get(name)


def list_tools() -> list[str]:
    """Return list of registered tool names."""
    return list(_registry["tools"].keys())


def register_provider(name: str, client: Any) -> None:
    """Register a provider client instance."""
    _registry["providers"][name] = client


def get_provider(name: str) -> Any:
    """Return a provider client instance by name. Raises KeyError if not found."""
    if name not in _registry["providers"]:
        raise KeyError(f"Provider '{name}' not registered. Available: {list(_registry['providers'].keys())}")
    return _registry["providers"][name]


def register_channel(name: str, handler) -> None:
    """Register a Channel instance."""
    _registry["channels"][name] = handler


def get_channel(name: str):
    """Return a Channel instance by name. Raises KeyError if not found."""
    if name not in _registry["channels"]:
        raise KeyError(f"Channel '{name}' not registered. Available: {list(_registry['channels'].keys())}")
    return _registry["channels"][name]


def reset() -> None:
    """Clear all registries. Useful for testing."""
    _registry["tools"].clear()
    _registry["tool_schemas"].clear()
    _registry["providers"].clear()
    _registry["channels"].clear()
