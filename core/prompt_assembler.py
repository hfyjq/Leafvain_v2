"""
Dynamic system-prompt assembly from multiple sources.

Replaces the hardcoded ``SYSTEM_PROMPT`` constant in agent_loop.py
with a multi-source pipeline:

  1. Core behavioural instructions (framework-agnostic rules).
  2. User project context  — LEAF.md (user's project description).
  3. User memory index     — LEAF_memory.md (auto-maintained by the agent).
  4. Git status            — current branch + recent commits (memoized).
  5. Available tools       — name + one-liner, auto-generated from registry.

**Important:** This module is for the **Leafvain agent's** context.
It does NOT read Claude Code's CLAUDE.md or MEMORY.md — those belong
to the framework developer, not the agent's end user.  The LEAF_*
namespace keeps the two worlds separate.

All sections use XML-style tags so the model can structurally distinguish
live conversation from reference material (consistent with the existing
``<background_context>`` convention).
"""

import subprocess
import time
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------
# Module-level state (set once by main.py before first use)
# ------------------------------------------------------------------

_project_root: Optional[Path] = None


def set_project_root(root: Path) -> None:
    """Store the project root for git-status resolution.

    Must be called once from ``main.py`` before the first prompt is
    assembled so git commands run in the correct directory.
    """
    global _project_root
    _project_root = root.resolve()


def _get_project_root() -> Path:
    """Return the stored project root, falling back to cwd."""
    if _project_root is not None:
        return _project_root
    return Path.cwd()


# ------------------------------------------------------------------
# Core behavioural instructions
# ------------------------------------------------------------------
# This is the "spine" of the system prompt.  It does NOT include a
# tool list any more — the <available_tools> section is generated
# dynamically from the registry so it stays in sync with what is
# actually registered.
# ------------------------------------------------------------------

_CORE_INSTRUCTIONS = r"""You are a document analysis assistant. You can parse documents, search their contents, and generate summaries.

## CONTEXT FORMAT
Content wrapped in `<background_context>…</background_context>` tags is **historical reference material** — facts, user profile, and session notes from past conversations.  It is NOT part of the live conversation.

Rules for `<background_context>`:
- Use it silently as background knowledge to give better answers.
- NEVER mention, reference, or acknowledge it in your replies.
- NEVER say "I remember", "based on the background", "according to your profile", "as we discussed before", etc.
- If the user asks "what were we talking about" or "where were we", you may briefly answer — but NEVER volunteer a recap or review of the conversation unprompted.

## CONVERSATION STYLE
- This is a continuous conversation.  Do NOT greet, reintroduce yourself, or imply a new session has started.
- Answer the user's latest message directly.  No preamble, no review, no recap.
- NEVER fabricate narratives about previous turns (e.g. "you interrupted my explanation", "we were just about to discuss X").

## CONTEXT TAGS
XML tags like `<system_instructions>`, `<leaf_context>`, `<leaf_memory>`, `<git_status>`, and `<available_tools>` are **reference for you** — they describe the environment you are running in.  Do NOT mention them in your replies unless the user explicitly asks about them.

## IMPORTANT RULES
1. When the user mentions a file, call the appropriate parse tool for its type.
   - If the user gives a bare filename, try both: the path as-is, and "data/workspaces/<filename>".
2. Use search_chunks or summarize_document to work with loaded document content.
3. NEVER ask to see the full document text — always use search_chunks.
4. Answer in the same language as the user's question.
5. When you're done, give the final answer directly — do not call more tools unless the user asks a follow-up."""


# ------------------------------------------------------------------
# Memoized git status
# ------------------------------------------------------------------

_git_status_cache: Optional[str] = None
_git_status_time: float = 0.0
_GIT_CACHE_TTL: float = 300.0  # seconds


def _get_git_status() -> str:
    """Return current branch + last 5 commits, memoized for the session."""
    global _git_status_cache, _git_status_time

    now = time.monotonic()
    if _git_status_cache is not None and (now - _git_status_time) < _GIT_CACHE_TTL:
        return _git_status_cache

    lines: list[str] = []

    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
        ).strip()
        if branch:
            lines.append(f"Branch: {branch}")
    except (subprocess.CalledProcessError, FileNotFoundError, UnicodeDecodeError):
        lines.append("Branch: (unknown)")

    try:
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5"],
            text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
        ).strip()
        if log:
            lines.append("Recent commits:")
            for commit_line in log.split("\n"):
                lines.append(f"  {commit_line}")
    except (subprocess.CalledProcessError, FileNotFoundError, UnicodeDecodeError):
        lines.append("Recent commits: (unavailable)")

    _git_status_cache = "\n".join(lines)
    _git_status_time = now
    return _git_status_cache


# ------------------------------------------------------------------
# File loaders  (LEAF namespace — user-facing agent context)
# ------------------------------------------------------------------

# Paths the agent looks for user project context.
#  1. <project_root>/LEAF.md          — checked first
#  2. data/workspaces/LEAF.md         — fallback (user workspace)
# If neither exists the section is omitted silently.
_LEAF_MD_PATHS = (
    Path("LEAF.md"),
    Path("data/workspaces/LEAF.md"),
)

# Agent-maintained cross-session memory index.
# The agent writes this file as it learns about the user.
_LEAF_MEMORY_PATH = Path("data/memory/LEAF_memory.md")


def _load_leaf_md() -> str:
    """Read LEAF.md — the user's project context for the agent.

    This is the user-facing equivalent of CLAUDE.md: it describes
    what project the USER is working on (not the Leafvain framework
    itself).  The agent uses it to understand the user's domain.
    """
    for path in _LEAF_MD_PATHS:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""  # silently omit — no user project context yet


def _load_leaf_memory() -> str:
    """Read LEAF_memory.md — the agent's cross-session memory index.

    Maintained automatically by the agent (via MemoryManager), this
    file contains an index of what the agent has learned about the
    user across all conversations.  It is the agent's equivalent of
    Claude Code's MEMORY.md.
    """
    if not _LEAF_MEMORY_PATH.exists():
        return ""  # no memory yet — fresh agent
    try:
        return _LEAF_MEMORY_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def assemble_system_prompt(memory_context: str = "") -> str:
    """Build the full system prompt from all configured sources.

    Args:
        memory_context: Optional memory context from MemoryManager
            (wrapped in ``<background_context>`` by the caller / manager).
            When empty, the section is omitted.

    Returns:
        The complete system prompt string ready to use as the first
        ``{"role": "system", "content": ...}`` message.
    """
    from bus.registry import get_tool_summaries

    sections: list[str] = []

    # 1. Core behavioural instructions (framework-agnostic)
    sections.append(
        "<system_instructions>\n"
        + _CORE_INSTRUCTIONS
        + "\n</system_instructions>"
    )

    # 2. User project context — LEAF.md (user's domain, NOT framework dev)
    leaf_md = _load_leaf_md()
    if leaf_md:
        sections.append(
            "<leaf_context>\n"
            + leaf_md
            + "\n</leaf_context>"
        )

    # 3. Agent cross-session memory — LEAF_memory.md (auto-maintained)
    leaf_memory = _load_leaf_memory()
    if leaf_memory:
        sections.append(
            "<leaf_memory>\n"
            + leaf_memory
            + "\n</leaf_memory>"
        )

    # 4. Git status (memoized — shows user's project branch/commits)
    git_status = _get_git_status()
    sections.append(
        "<git_status>\n"
        + git_status
        + "\n</git_status>"
    )

    # 5. Available tools (auto-generated from registry)
    tool_lines = get_tool_summaries()
    tools_text = "\n".join(f"- {tl}" for tl in tool_lines)
    sections.append(
        "<available_tools>\n"
        + tools_text
        + "\n</available_tools>"
    )

    # 6. Memory context (only when provided by MemoryManager)
    if memory_context:
        sections.append(memory_context)

    return "\n\n".join(sections)
