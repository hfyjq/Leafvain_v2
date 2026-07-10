"""
CLI channel: interactive input loop with persistent memory.

Session state (messages, chunks, memory) persists across turns AND
across process restarts via the MemoryManager.
"""

from typing import Dict, List

from core.agent_loop import agent_loop


WELCOME_BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║        Leafvain v2  —  Document Analysis Agent          ║
║                                                          ║
║  Examples:                                               ║
║    解析 te1.txt                                          ║
║    总结这个文档                                           ║
║    搜索 XXX 相关的内容                                    ║
║    总结 data/workspaces/report.pdf                        ║
║                                                          ║
║  Exit:  exit / quit / Ctrl+C                            ║
╚══════════════════════════════════════════════════════════╝
"""


async def run(provider, memory_manager=None) -> None:
    """
    Start the interactive CLI loop with persistent session state.

    Args:
        provider: The active LLM provider client instance.
        memory_manager: MemoryManager instance (optional).
    """
    print(WELCOME_BANNER)

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
            print(f"[memory] Restored session {session_id} "
                  f"({len(messages)} messages)")
        else:
            print(f"[memory] New session: {session_id}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        print("\nAgent is thinking...")

        try:
            # Build memory context for this turn
            memory_ctx = ""
            if memory_manager is not None and session_id:
                memory_ctx = memory_manager.get_context(
                    session_id, user_input,
                )

            response, messages, session_chunks, tokens_used = agent_loop(
                user_input,
                provider,
                messages=messages,
                session_chunks=session_chunks,
                memory_context=memory_ctx,
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
                    # Re-inject system prompt at the front if missing
                    if messages and messages[0].get("role") != "system":
                        from core.agent_loop import SYSTEM_PROMPT
                        messages.insert(0, {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        })

        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
