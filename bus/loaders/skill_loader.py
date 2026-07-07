"""
Load a skill from its manifest.yaml, validate JSON Schema tool definitions,
look up handler functions, and register tools in the global registry.

Also supports SKILL.md (agentskills.io standard) for metadata and
ecosystem compatibility.  The manifest.yaml remains the authoritative
source for tool definitions; SKILL.md provides name/description/instructions.
"""

import importlib
import re
from pathlib import Path

import yaml

from bus.registry import register_tool

# Allowed JSON Schema types (subset for Phase 2)
_VALID_JSON_TYPES = {"string", "number", "integer", "boolean", "array", "object"}

# Known JSON Schema $schema URIs we accept
_VALID_SCHEMA_URIS = {
    "https://json-schema.org/draft/2020-12/schema",
    "https://json-schema.org/draft/2019-09/schema",
    "http://json-schema.org/draft-07/schema#",
}


def validate_tool_schema(schema: dict) -> None:
    """
    Minimal JSON Schema 2020-12 validation for tool inputSchema.

    Checks:
      - Root type is 'object'
      - Properties are valid dicts with recognized types
      - Required fields exist in properties
      - $schema (if present) is a recognized URI

    Raises ValueError with details if the schema is invalid.
    """
    if not isinstance(schema, dict):
        raise ValueError("inputSchema must be a dict")

    # $schema is optional but recommended — validate if present
    schema_uri = schema.get("$schema")
    if schema_uri is not None:
        if schema_uri not in _VALID_SCHEMA_URIS:
            raise ValueError(
                f"Unrecognized $schema URI: '{schema_uri}'. "
                f"Expected one of: {_VALID_SCHEMA_URIS}"
            )

    if schema.get("type") != "object":
        raise ValueError(
            f"inputSchema root type must be 'object', got '{schema.get('type')}'"
        )

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("inputSchema 'properties' must be a dict")

    for prop_name, prop_def in properties.items():
        if not isinstance(prop_def, dict):
            raise ValueError(
                f"Property '{prop_name}' definition must be a dict, "
                f"got {type(prop_def).__name__}"
            )
        prop_type = prop_def.get("type")
        if prop_type not in _VALID_JSON_TYPES:
            raise ValueError(
                f"Property '{prop_name}' has invalid type '{prop_type}'. "
                f"Must be one of: {_VALID_JSON_TYPES}"
            )

    required = schema.get("required", [])
    if not isinstance(required, list):
        raise ValueError("'required' must be a list")

    for req_prop in required:
        if req_prop not in properties:
            raise ValueError(
                f"Required property '{req_prop}' is not defined in 'properties'"
            )


# ------------------------------------------------------------------
# SKILL.md helpers (agentskills.io standard)
# ------------------------------------------------------------------

def _parse_skill_md(skill_dir: Path) -> dict | None:
    """
    Read and parse a SKILL.md file if it exists.

    Returns a dict with keys:
      - name: str (from YAML frontmatter)
      - description: str
      - body: str (Markdown body after frontmatter)
      - raw_metadata: dict (any extra frontmatter fields)

    Returns None if SKILL.md does not exist or cannot be parsed.
    """
    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.exists():
        return None

    text = skill_md_path.read_text(encoding="utf-8")

    # Parse YAML frontmatter (delimited by ---)
    # SKILL.md format:
    #   ---
    #   name: skill-name
    #   description: ...
    #   ---
    #   Markdown body...
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if not frontmatter_match:
        print(f"[bus] WARNING: {skill_md_path} has no valid YAML frontmatter")
        return None

    try:
        frontmatter = yaml.safe_load(frontmatter_match.group(1))
    except yaml.YAMLError as e:
        print(f"[bus] WARNING: {skill_md_path} YAML parse error: {e}")
        return None

    if not isinstance(frontmatter, dict):
        print(f"[bus] WARNING: {skill_md_path} frontmatter is not a dict")
        return None

    name = frontmatter.get("name", "").strip()
    description = frontmatter.get("description", "").strip()
    body = frontmatter_match.group(2).strip()

    # Validate required fields (per agentskills.io spec)
    if not name:
        print(f"[bus] WARNING: {skill_md_path} missing required 'name' field")
    if not description:
        print(f"[bus] WARNING: {skill_md_path} missing required 'description' field")

    # Validate name format: lowercase, digits, hyphens only; 1-64 chars
    if name and not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", name):
        print(
            f"[bus] WARNING: SKILL.md name '{name}' does not match "
            f"agentskills.io spec (lowercase letters, digits, hyphens only, "
            f"no leading/trailing hyphens, no consecutive hyphens)"
        )

    return {
        "name": name,
        "description": description,
        "body": body,
        "raw_metadata": {k: v for k, v in frontmatter.items()
                         if k not in ("name", "description")},
    }


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_skill(skill_dir: Path, security_guard=None) -> None:
    """
    1. Read <skill_dir>/manifest.yaml.
    2. Validate kind == "skill".
    3. For each tool: validate inputSchema + register.
    4. Resolve handler functions from handler.py.
    5. Optionally parse SKILL.md for metadata/ecosystem compatibility.
    """
    manifest_path = skill_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Skill manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    if manifest.get("kind") != "skill":
        raise ValueError(
            f"Expected kind='skill' in {manifest_path}, got '{manifest.get('kind')}'"
        )

    manifest_name = manifest["name"]

    # --- Import the handler module ---
    module_name = f"skills.{skill_dir.name}.handler"
    module = importlib.import_module(module_name)

    # If the handler module has an init() function, call it with security guard
    if hasattr(module, "init") and security_guard is not None:
        module.init(security_guard)

    # --- Register tools from manifest ---
    tools_registered = 0
    for tool_def in manifest.get("tools", []):
        tool_name = tool_def["name"]
        tool_description = tool_def.get("description", "")
        input_schema = tool_def.get("inputSchema", {})

        # Validate schema before registering
        validate_tool_schema(input_schema)

        # Look up handler function (must match tool name exactly)
        handler = getattr(module, tool_name, None)
        if handler is None:
            raise AttributeError(
                f"Tool '{tool_name}' defined in {manifest_path} "
                f"but no function '{tool_name}' found in {module_name}.py"
            )

        combined_schema = {
            "description": tool_description,
            "inputSchema": input_schema,
        }

        register_tool(tool_name, handler, combined_schema)
        tools_registered += 1

    # --- Parse SKILL.md for ecosystem compatibility ---
    skill_md = _parse_skill_md(skill_dir)
    skill_md_name = None
    if skill_md and skill_md["name"]:
        skill_md_name = skill_md["name"]
        print(
            f"[bus] Skill loaded: {manifest_name} "
            f"(SKILL.md: {skill_md_name}, {tools_registered} tools)"
        )
    else:
        print(f"[bus] Skill loaded: {manifest_name} ({tools_registered} tools)")

    if skill_md and skill_md.get("raw_metadata", {}).get("mcp_note"):
        print(f"[bus]   MCP note: {skill_md['raw_metadata']['mcp_note'][:120]}...")
