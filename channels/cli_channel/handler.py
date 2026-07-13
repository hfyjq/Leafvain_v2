"""
CLI channel: interactive input loop with persistent memory.

Session state (messages, chunks, memory) persists across turns AND
across process restarts via the MemoryManager.
"""

import os
import sys
from typing import Dict, List

from core.agent_loop import agent_loop
from core.prompt_assembler import assemble_system_prompt


# ------------------------------------------------------------------
# Input — plain ``input()`` with post-turn console drain on Windows
# ------------------------------------------------------------------
#
# Multi-line paste (Ctrl+V) into the Windows console is fundamentally
# unreliable because ``input()`` uses C runtime buffered I/O while
# ``msvcrt`` reads from a separate console API layer.  Rather than
# fighting this, we accept single-line ``input()`` and drain any
# residual characters after each turn so they don't leak into the
# next one.  For code / multi-line content, save it as a file and use
# ``parse_txt`` / ``parse_pdf``.
#


def _drain_console() -> None:
    """Discard any lingering console-input characters between turns.

    Called after each agent response so that paste remnants from the
    *previous* ``input()`` call don't become the *next* turn's input.
    """
    if os.name != "nt":
        return
    import msvcrt

    count = 0
    while msvcrt.kbhit():
        msvcrt.getwch()
        count += 1
        if count > 50_000:          # safety valve
            break

    if count > 10:
        print(
            f"\n[提示] 检测到粘贴残留（{count} 字符已丢弃）。\n"
            f"[提示] 分析代码 / 长文本请先保存为文件，再用 "
            f"\"解析 文件名\" 加载。"
        )


WELCOME_BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║        Leafvain v2  —  Document Analysis Agent           ║
║                                                          ║
║  Examples:                                               ║
║    解析 te1.txt                                           ║
║    总结这个文档                                             ║
║    搜索 XXX 相关的内容                                      ║
║    总结 data/workspaces/report.pdf                        ║
║                                                          ║
║  Exit:  exit / quit / Ctrl+C                             ║
╚══════════════════════════════════════════════════════════╝
"""

# ------------------------------------------------------------------
# Context usage display helpers
# ------------------------------------------------------------------

_BAR_WIDTH = 20
_WARN_PCT = 70
_HIGH_PCT = 85


def _format_tokens(n: int) -> str:
    """Pretty-print token count (e.g. 15420 -> '15.4K')."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _render_usage_bar(pct: float, threshold_pct: float) -> str:
    """Draw a 20-char context-usage bar."""
    filled = int(pct / 100 * _BAR_WIDTH)
    filled = max(0, min(_BAR_WIDTH, filled))
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)

    if pct >= threshold_pct:
        return f"{bar} 🔴 COMPRESSING"
    elif pct >= _HIGH_PCT:
        return f"{bar} 🔴"
    elif pct >= _WARN_PCT:
        return f"{bar} ⚠"
    else:
        return bar


def _display_context_usage(memory_manager, messages) -> None:
    """Print a one-line context-usage summary after each turn."""
    if memory_manager is None:
        return

    usage = memory_manager.get_context_usage(messages)
    pct = usage["percentage"]
    est = usage["estimated_tokens"]
    win = usage["context_window"]
    threshold_pct = usage["threshold_pct"]

    bar = _render_usage_bar(pct, threshold_pct)
    print(
        f"\n  📊 context: {pct}% {bar}  "
        f"{_format_tokens(est)}/{_format_tokens(win)} tokens  "
        f"(threshold: {threshold_pct}%)"
    )


def _display_compression(memory_manager) -> None:
    """Print a one-line compression summary."""
    stats = memory_manager.last_compression
    if stats is None:
        return

    source_label = (
        "session note" if stats.get("source") == "session_note" else "LLM"
    )
    print(
        f"  ⚡ compressed: {stats['compressed_turns']} turns → summary "
        f"({stats['summary_chars']} chars) [{source_label}], "
        f"{stats['kept_turns']} turns kept  "
        f"({stats['old_message_count']} → {stats['new_message_count']} messages)"
    )


async def run(provider, memory_manager=None, config: dict | None = None) -> None:
    """
    Start the interactive CLI loop with persistent session state.

    Args:
        provider: The active LLM provider client instance.
        memory_manager: MemoryManager instance (optional).
        config: Full application config dict (optional).
    """
    print(WELCOME_BANNER)

    # Extract agent-level settings from config
    tool_result_max_chars = 2000
    if config is not None:
        tool_result_max_chars = config.get("tool_result_max_chars", 2000)

    # ------------------------------------------------------------------
    # Session state — persisted across turns and restarts
    # ------------------------------------------------------------------
    messages: List[Dict] | None = None
    session_chunks: List[Dict] = []
    session_id: str | None = None

    # If memory manager is available, restore previous session
    if memory_manager is not None:
        session_id, messages = memory_manager.load_session()
        if messages:
            # Ensure SYSTEM_PROMPT is present in restored messages.
            # Compressed sessions may only have <historical_context>.
            if messages[0].get("role") == "system":
                existing = messages[0].get("content", "")
                if "<system_instructions>" not in existing:
                    messages[0]["content"] = assemble_system_prompt() + "\n\n" + existing

            # Show a clear session-restore indicator with context stats
            usage = memory_manager.get_context_usage(messages)
            print(
                f"\n  📋 Restored session: {session_id}\n"
                f"     {len(messages)} messages  |  "
                f"{_format_tokens(usage['estimated_tokens'])}/"
                f"{_format_tokens(usage['context_window'])} tokens  "
                f"({usage['percentage']}%)"
            )
            if memory_manager.session_memory.exists(session_id):
                print(
                    f"  📝 session note loaded "
                    f"({memory_manager.session_memory.get_note_path(session_id)})"
                )
            print(f"  💡 Coming soon: /clear to start a fresh session")
        else:
            print(f"\n  🆕 New session: {session_id}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        # Flush any pending background-thread output BEFORE the prompt
        # so it never interleaves with the You> input line.
        if memory_manager is not None:
            memory_manager.flush_bg_status()

        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        # --- /clear — hard reset, discard all context ---
        if user_input.strip().lower() == "/clear":
            old_msg_count = len(messages) if messages else 0
            old_usage = ""
            if memory_manager is not None and messages:
                usage = memory_manager.get_context_usage(messages)
                old_usage = (
                    f"  {_format_tokens(usage['estimated_tokens'])}/"
                    f"{_format_tokens(usage['context_window'])} tokens "
                    f"({usage['percentage']}%)"
                )

            session_chunks = []
            if memory_manager is not None:
                session_id, messages = memory_manager.new_session()
                print(
                    f"\n  🆕 /clear — session reset\n"
                    f"     Cleared: {old_msg_count} messages"
                    + (f", {old_usage}" if old_usage else "") + "\n"
                    f"     New session: {session_id}\n"
                    f"     Memory (profile + long-term facts) preserved."
                )
            else:
                messages = None
                print(f"\n  🆕 /clear — session reset ({old_msg_count} messages cleared)")
            continue

        # --- /compact — soft compress, replace old turns with summary ---
        if user_input.strip().lower() == "/compact":
            if memory_manager is None or not session_id or not messages:
                print("\n  ⚠ /compact: no active session to compact")
                continue

            old_count = len(messages)
            old_usage = memory_manager.get_context_usage(messages)

            compressed, was_compressed = memory_manager.compact(
                session_id, messages,
            )

            if was_compressed:
                messages = compressed
                # Merge system prompt back in if compression replaced it
                if messages and messages[0].get("role") == "system":
                    existing = messages[0].get("content", "")
                    if "<system_instructions>" not in existing:
                        messages[0]["content"] = (
                            assemble_system_prompt() + "\n\n" + existing
                        )
                new_usage = memory_manager.get_context_usage(messages)
                print(
                    f"\n  📦 /compact — context compressed\n"
                    f"     Messages: {old_count} → {len(messages)} "
                    f"({old_count - len(messages)} removed)\n"
                    f"     Tokens: "
                    f"{_format_tokens(old_usage['estimated_tokens'])} → "
                    f"{_format_tokens(new_usage['estimated_tokens'])} "
                    f"({old_usage['percentage']}% → {new_usage['percentage']}%)"
                )
                _display_compression(memory_manager)
            else:
                usage = memory_manager.get_context_usage(messages)
                print(
                    f"\n  ℹ /compact: not enough turns to compress "
                    f"({len(messages)} messages, "
                    f"{usage['percentage']}% context used)"
                )
            continue

        if not user_input:
            continue

        print("\nAgent is thinking...")

        try:
            # Build memory context for this turn
            memory_ctx = ""
            if memory_manager is not None and session_id:
                memory_ctx = memory_manager.get_context(
                    session_id, user_input, messages=messages,
                )

            response, messages, session_chunks, tokens_used = agent_loop(
                user_input,
                provider,
                messages=messages,
                session_chunks=session_chunks,
                memory_context=memory_ctx,
                tool_result_max_chars=tool_result_max_chars,
            )

            print(f"\nAgent: {response}")

            # Record turn in memory (persist messages + check compression + extract facts)
            if memory_manager is not None and session_id:
                # Build message dicts for persistence
                user_msg = {"role": "user", "content": user_input}
                assistant_msg = {"role": "assistant", "content": response}

                compressed = memory_manager.record_turn(
                    session_id=session_id,
                    user_message=user_msg,
                    assistant_message=assistant_msg,
                    token_count=tokens_used,
                    messages=messages,
                )

                # If compression happened, use the compressed messages list
                if compressed is not None:
                    messages = compressed
                    # Merge SYSTEM_PROMPT into the compressed system message.
                    # After compression, messages[0] is a system message with
                    # <historical_context> — we must prepend the core
                    # behavioural instructions so the agent doesn't lose them.
                    if messages and messages[0].get("role") == "system":
                        existing = messages[0].get("content", "")
                        if "<system_instructions>" not in existing:
                            messages[0]["content"] = (
                                assemble_system_prompt() + "\n\n" + existing
                            )
                    elif messages:
                        messages.insert(0, {
                            "role": "system",
                            "content": assemble_system_prompt(),
                        })

                    _display_compression(memory_manager)

                # Show context usage after each turn
                if memory_manager is not None and messages:
                    _display_context_usage(memory_manager, messages)

                # Show session note status
                if (
                    memory_manager is not None
                    and session_id
                    and memory_manager.session_memory.exists(session_id)
                ):
                    note_path = memory_manager.session_memory.get_note_path(
                        session_id
                    )
                    try:
                        note_size = note_path.stat().st_size
                        size_str = (
                            f"{note_size / 1024:.1f}KB"
                            if note_size >= 1024
                            else f"{note_size}B"
                        )
                        print(f"  📝 session note: {size_str}  ({note_path})")
                    except OSError:
                        pass

                # Drain console residual after each turn
                _drain_console()

        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
