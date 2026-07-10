"""
Core agent loop: Function Calling dispatch with hard constraint enforcement.

Key invariants (the four hard constraints):
  1. Streaming  — enforced by skill handlers (fitz iterator / line iterator).
  2. Chunking   — enforced by core.chunker (≤1500 chars, metadata).
  3. Isolation  — parse results stored in session_chunks, NEVER in LLM messages.
  4. Summary    — iterative aggregation via core.summarizer.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from bus.registry import get_tool_handler, get_tool_schemas
from core.retriever import BM25Retriever
from core.summarizer import iterative_summarize


SYSTEM_PROMPT = """You are a document analysis assistant. You can parse documents, search their contents, and generate summaries.

## Available tools
- **parse_pdf / parse_txt**: Parse a file and index its contents.
- **search_chunks**: Search indexed content with keywords. You can pass a file_path to auto-parse first.
- **summarize_document**: Generate a full summary. You can pass a file_path to auto-parse first.

## IMPORTANT RULES
1. When the user mentions a file, call parse_pdf (for .pdf) or parse_txt (for .txt) to load it first.
   - If the user gives a bare filename like "te1.txt", try both: the path as-is, and "data/workspaces/te1.txt".
   - If the user says "总结这个文件" or "search for X in this file" and you already have a file loaded,
     use search_chunks or summarize_document WITHOUT file_path — it will use the already-parsed content.
2. For summaries, you can pass file_path directly to summarize_document to auto-parse + summarize in one step.
3. For searching, you can pass file_path + query to search_chunks to auto-parse + search in one step.
4. NEVER ask to see the full document text — always use search_chunks.
5. Answer in the same language as the user's question.
6. When you're done, give the final answer directly — do not call more tools unless the user asks a follow-up."""


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
    if messages is None:
        system_content = SYSTEM_PROMPT
        if memory_context:
            system_content += f"\n\n{memory_context}"
        messages = [
            {"role": "system", "content": system_content},
        ]
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
