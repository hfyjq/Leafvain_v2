# MCP Ecosystem Adaptation Guide

## Current State (v0.2.0)

Leafvain skills use a **dual-format** approach:
- `manifest.yaml` — authoritative tool definitions (JSON Schema 2020-12 + handler mapping)
- `SKILL.md` — agentskills.io standard metadata (YAML frontmatter + Markdown body)

### Naming Convention
| Context | Convention | Example |
|---------|-----------|---------|
| Python module path | underscores | `skills.file_parser.handler` |
| SKILL.md `name` field | hyphens | `file-parser` |
| Directory name | underscores | `skills/file_parser/` |

The directory name uses underscores for Python `importlib` compatibility.
The SKILL.md `name` uses hyphens per agentskills.io spec.
This is a known deviation — see "Future Adaptation" below.

## How to Publish to the MCP Ecosystem

### Option 1: Thin Adapter Script (Recommended)

Create a script that generates MCP-compatible skill directories:

```python
# scripts/export_for_mcp.py
# Reads all skills/ subdirectories, generates MCP-compatible copies
# with:
#   - Directory renamed from file_parser → file-parser
#   - SKILL.md preserved as-is
#   - scripts/ subdirectory with handler logic
#   - manifest.yaml converted to MCP tool definitions
```

### Option 2: Symlink Adapter

```bash
# For each skill, create a symlink with hyphenated name
ln -s skills/file_parser skills/file-parser
ln -s skills/office_parser skills/office-parser
```

### Option 3: MCP Server Wrapper

Wrap the entire Leafvain bus as an MCP server that exposes all
registered tools via the MCP `tools/list` and `tools/call` protocol.

```python
# A minimal MCP server would:
# 1. Call init_bus() to load all skills
# 2. Expose get_tool_schemas() as tools/list response
# 3. Expose get_tool_handler() as tools/call dispatch
```

## What's Already Compatible

- ✅ Tool definitions use JSON Schema 2020-12 (`$schema` declared)
- ✅ SKILL.md follows agentskills.io frontmatter format
- ✅ name/description fields meet the spec requirements
- ✅ Tools are registered in a format compatible with MCP `tools/list`

## What Needs Adaptation

- ❌ Directory names use underscores (Python constraint) vs hyphens (MCP standard)
- ❌ Handler functions are Python callables, not MCP server endpoints
- ❌ No `scripts/` subdirectory per SKILL.md standard

## Recommended Timeline

1. **v0.3.0**: Add `scripts/export_for_mcp.py` adapter
2. **v0.4.0**: Implement MCP server wrapper (expose bus as STDIO MCP server)
3. **v0.5.0**: Full agentskills.io directory structure compliance
