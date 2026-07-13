"""
Core agent loop: Function Calling dispatch with hard constraint enforcement.

Key invariants (the four hard constraints):
  1. Streaming  — enforced by skill handlers (fitz iterator / line iterator).
  2. Chunking   — enforced by core.chunker (≤1500 chars, metadata).
  3. Isolation  — parse results stored in session_chunks, NEVER in LLM messages.
  4. Summary    — iterative aggregation via core.summarizer.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from bus.registry import get_tool_handler, get_tool_schemas
from core.prompt_assembler import assemble_system_prompt
from core.retriever import BM25Retriever
from core.summarizer import iterative_summarize


# ------------------------------------------------------------------
# Return type for agent_loop — carries state across turns
# ------------------------------------------------------------------

SessionState = Tuple[str, List[Dict], List[Dict], int]
# (response_text, updated_messages, updated_session_chunks, tokens_used)


def agent_loop(
    user_message: str,
    provider,  # Provider client with .chat(messages, tools) method
    messages: List[Dict] | None = None,
    session_chunks: List[Dict] | None = None,
    memory_context: str = "",
    max_turns: int = 15,
    tool_result_max_chars: int = 2000,
) -> SessionState:
    """
    Process one user message with tool-calling support.

    Args:
        user_message: The user's natural-language input.
        provider: LLM provider client instance.
        messages: Existing conversation messages (persisted across turns).
        session_chunks: Previously parsed chunks (persisted across turns).
        max_turns: Safety limit to prevent infinite tool-calling loops.

    Returns:
        (response_text, updated_messages, updated_session_chunks, tokens_used)
    """
    # ------------------------------------------------------------------
    # Per-session state (IN-MEMORY, never in LLM context)
    # ------------------------------------------------------------------
    if session_chunks is None:
        session_chunks = []
    retriever = BM25Retriever()
    if session_chunks:
        retriever.index(session_chunks)

    tokens_used = 0

    # ------------------------------------------------------------------
    # Build / extend messages
    # ------------------------------------------------------------------
    if messages is None or len(messages) == 0:
        system_content = assemble_system_prompt(memory_context)
        messages = [
            {"role": "system", "content": system_content},
        ]
        print(f"[agent] Created system prompt"
              + (f" with memory context ({len(memory_context)} chars)"
                 if memory_context else " (no memory yet)"))

    # Inject / refresh memory context in the system message.
    # Previous memory sections are stripped before the new one is
    # appended — otherwise the system message balloons with
    # duplicated profile data across turns.
    if memory_context:
        injected = False
        for i, m in enumerate(messages):
            if m.get("role") == "system":
                existing = m.get("content", "")

                # Strip any previously-injected memory sections so
                # they don't accumulate across turns.  We identify
                # them by the section headers we use in manager.py.
                for header in (
                    "## USER PROFILE (internal",
                    "## RELEVANT CONTEXT (internal",
                    "## SESSION STATE (internal",
                    "## YOUR MEMORY — User Profile",
                    "## YOUR MEMORY — Relevant Past Conversations",
                    "## YOUR MEMORY — Current Session",
                ):
                    idx = existing.find(header)
                    if idx != -1:
                        existing = existing[:idx].rstrip()

                if memory_context not in existing:
                    messages[i] = {
                        "role": "system",
                        "content": existing + f"\n\n{memory_context}",
                    }
                    print(f"[agent] Injected memory context "
                          f"({len(memory_context)} chars)")
                injected = True
                break
        if not injected:
            messages.insert(0, {
                "role": "system",
                "content": assemble_system_prompt(memory_context),
            })
            print(f"[agent] Prepended system prompt with memory context "
                  f"({len(memory_context)} chars)")

    # On non-first turns, insert a brief system-level nudge so the
    # model treats this as a continuous conversation — not a new one.
    # Flash / small models in particular tend to ignore long system
    # prompts but respond well to short, positionally-primed nudges
    # right before the user message.
    has_prior_user = any(m.get("role") == "user" for m in messages)
    if has_prior_user:
        messages.append({
            "role": "system",
            "content": "[This is a continuous conversation. Reply directly — no greetings, no recaps.]",
        })

    messages.append({"role": "user", "content": user_message})

    tools = get_tool_schemas()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for turn in range(max_turns):
        response = provider.chat(messages, tools)
        tokens_used += response.total_tokens

        # Terminal: model returned a text response (no tool calls)
        content = response.content
        tool_calls = response.tool_calls

        if content and not tool_calls:
            return (content, messages, session_chunks, tokens_used)

        if not tool_calls:
            return ("(no response from model)", messages, session_chunks, tokens_used)

        # Append assistant message (with tool_calls) to history
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": tc["function"],
                }
                for tc in tool_calls
            ],
        })

        # --------------------------------------------------------------
        # Dispatch each tool call
        # --------------------------------------------------------------
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            tool_result_content: str

            try:
                handler = get_tool_handler(tool_name)
                if handler is None:
                    tool_result_content = f"Error: unknown tool '{tool_name}'"

                # ======================================================
                #  PARSE tools — chunk & index, never expose to LLM
                # ======================================================
                elif tool_name in (
                    "parse_pdf", "parse_txt",
                    "parse_docx", "parse_pptx", "parse_xlsx",
                ):
                    new_chunks = handler(**tool_args)
                    session_chunks.extend(new_chunks)
                    retriever.index(session_chunks)
                    tool_result_content = (
                        f"Parsed {len(new_chunks)} text chunks from the file. "
                        f"Total chunks in session: {len(session_chunks)}. "
                        f"The document is now loaded and ready. "
                        f"Use search_chunks to find specific information, "
                        f"or summarize_document for a full summary."
                    )

                # ======================================================
                #  SEARCH — auto-parse shortcut if file_path given
                # ======================================================
                elif tool_name == "search_chunks":
                    file_path = tool_args.get("file_path", "")
                    query = tool_args.get("query", "")

                    # Auto-parse shortcut: if a file_path is given and
                    # nothing is loaded yet, parse the file first.
                    if file_path and not session_chunks:
                        new_chunks = _auto_parse(file_path, provider)
                        if new_chunks:
                            session_chunks.extend(new_chunks)
                            retriever.index(session_chunks)

                    if not session_chunks:
                        tool_result_content = (
                            "No document loaded. Provide a file_path to auto-parse, "
                            "or call parse_pdf / parse_txt first."
                        )
                    elif not query:
                        tool_result_content = (
                            f"Document has {len(session_chunks)} chunks loaded. "
                            f"What would you like to search for? Provide a query."
                        )
                    else:
                        top_chunks = retriever.retrieve(query, top_k=3)
                        tool_result_content = _format_chunks(top_chunks)

                # ======================================================
                #  SUMMARIZE — auto-parse shortcut if file_path given
                # ======================================================
                elif tool_name == "summarize_document":
                    file_path = tool_args.get("file_path", "")

                    # Auto-parse shortcut
                    if file_path and not session_chunks:
                        new_chunks = _auto_parse(file_path, provider)
                        if new_chunks:
                            session_chunks.extend(new_chunks)
                            retriever.index(session_chunks)

                    if not session_chunks:
                        tool_result_content = (
                            "No document loaded. Provide a file_path to auto-parse, "
                            "or call parse_pdf / parse_txt first."
                        )
                    else:
                        summary = iterative_summarize(
                            session_chunks,
                            provider,
                        )
                        tool_result_content = summary

                # ======================================================
                #  Generic tool — pass result as JSON
                # ======================================================
                else:
                    tool_result_content = json.dumps(
                        handler(**tool_args), ensure_ascii=False
                    )

            except Exception as e:
                tool_result_content = f"Tool execution error: {type(e).__name__}: {e}"

            # Apply output budget: offload oversized results to disk
            tool_result_content = _format_tool_result(
                tool_result_content, tool_name,
                max_chars=tool_result_max_chars,
            )

            # Append tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result_content,
            })

    # Hit max_turns
    return (
        "(reached maximum tool-calling turns — please rephrase your request)",
        messages,
        session_chunks,
        tokens_used,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _auto_parse(file_path: str, provider) -> List[Dict]:
    """
    Auto-detect file type (by extension) and parse.

    Tries multiple path resolutions:
      1. file_path as-is
      2. data/workspaces/<file_path>
    """
    from skills.file_parser.handler import parse_pdf, parse_txt
    from skills.office_parser.handler import parse_docx, parse_pptx, parse_xlsx

    paths_to_try = [file_path, f"data/workspaces/{Path(file_path).name}"]
    resolved = None

    for p in paths_to_try:
        if Path(p).exists():
            resolved = p
            break

    if resolved is None:
        # Let the security guard reject it — if the user typed a full
        # path outside workspaces, we still try it so the error is clear.
        resolved = file_path

    ext = Path(resolved).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(resolved)
    elif ext == ".txt":
        return parse_txt(resolved)
    elif ext == ".docx":
        return parse_docx(resolved)
    elif ext in (".pptx", ".ppt"):
        return parse_pptx(resolved)
    elif ext in (".xlsx", ".xls"):
        return parse_xlsx(resolved)
    else:
        # Unknown extension — try txt
        return parse_txt(resolved)


# ------------------------------------------------------------------
# Tool result offloading (L1 — persistent output to disk)
# ------------------------------------------------------------------

_TOOL_RESULTS_DIR = Path("data/workspaces/tool_results")


def _resolve_tool_output_dir() -> Path:
    """Ensure the tool-results output directory exists and return its path."""
    _TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _TOOL_RESULTS_DIR


def _format_tool_result(
    content: str,
    tool_name: str,
    max_chars: int = 2000,
) -> str:
    """Apply output budget to a tool result.

    When *content* exceeds *max_chars* the full text is written to
    ``data/workspaces/tool_results/<ts>_<tool>.txt`` and a truncated
    preview + ``<persistent_output>`` tag is returned instead.

    Args:
        content: The raw tool result string.
        tool_name: Name of the tool that produced this result.
        max_chars: Threshold in characters.  0 disables offloading.

    Returns:
        Either the original *content* (if under threshold) or a
        preview string with a ``<persistent_output>`` pointer.
    """
    if max_chars <= 0 or len(content) <= max_chars:
        return content

    # Build a timestamped output file name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _resolve_tool_output_dir()
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in tool_name)
    out_path = out_dir / f"{ts}_{safe_name}.txt"

    out_path.write_text(content, encoding="utf-8")

    preview = content[:max_chars]
    truncated = len(content) - max_chars

    return (
        f"{preview}\n\n"
        f"... [truncated: {truncated:,} more chars]\n\n"
        f"<persistent_output file=\"{out_path}\">\n"
        f"Full output ({len(content):,} chars) written to disk.\n"
        f"</persistent_output>"
    )


def _format_chunks(chunks: List[Dict]) -> str:
    """Format top-K chunks for LLM consumption."""
    if not chunks:
        return "No relevant chunks found. Try a different query or rephrase."

    lines: List[str] = []
    for c in chunks:
        page = c.get("page", "?")
        idx = c.get("chunk_index", "?")
        text = c.get("text", "")
        lines.append(f"[Page {page}, Chunk {idx}]:\n{text}")

    return "\n\n---\n\n".join(lines)
