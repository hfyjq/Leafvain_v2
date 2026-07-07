"""
CLI channel: interactive input loop for the Leafvain agent.

Provides a simple stdin/stdout interface.  Each user input line is
dispatched to the agent loop and the response is printed.

Session state (messages, chunks) persists across turns so the agent
remembers previously parsed documents within a session.
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


async def run(provider) -> None:
    """
    Start the interactive CLI loop with persistent session state.

    Args:
        provider: The active LLM provider client instance.
    """
    print(WELCOME_BANNER)

    # Persistent across turns within a session
    messages: List[Dict] | None = None
    session_chunks: List[Dict] = []

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
            response, messages, session_chunks = agent_loop(
                user_input,
                provider,
                messages=messages,
                session_chunks=session_chunks,
            )
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
