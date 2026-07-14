"""
SessionPool — manages per-user session state on top of a shared MemoryManager.

MemoryManager already stores per-session state internally (keyed by
``session_id``), so we don't need multiple MemoryManager instances.
The pool simply:

1. Maps ``str(MessageSession)`` → internal ``session_id`` for
   load/restore.
2. Provides ``get_or_create()`` to retrieve (session_id, messages)
   for a given session.
3. Wraps ``MemoryManager`` methods so pipeline stages don't need to
   manage session IDs manually.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


class SessionPool:
    """Per-user session facade over a single MemoryManager.

    Each :class:`~channels.base.MessageSession` stringifies to a
    globally-unique key (e.g. ``"cli:private:local"``).  This key
    is used as the ``session_id`` for MemoryManager operations,
    so every user/group gets independent short-term history,
    compression state, and session notes.
    """

    def __init__(self, memory_manager) -> None:
        """
        Args:
            memory_manager: A fully-initialised
                :class:`~core.memory.manager.MemoryManager` instance.
        """
        self._mgr = memory_manager
        # Map external session_key → loaded state
        self._loaded: Dict[str, Tuple[str, List[dict]]] = {}

    # ------------------------------------------------------------------
    def get_or_create(
        self, session_key: str, provider=None
    ) -> Tuple[str, List[dict]]:
        """Return ``(session_id, messages)`` for *session_key*.

        On first access the session is loaded from persistent storage
        (or created fresh).  Subsequent calls return the cached state.
        """
        if session_key in self._loaded:
            return self._loaded[session_key]

        sid, messages = self._mgr.load_session(session_key)
        self._loaded[session_key] = (sid, messages)
        return sid, messages

    # ------------------------------------------------------------------
    # Convenience wrappers that delegate to MemoryManager
    # ------------------------------------------------------------------

    def get_context(
        self, session_key: str, user_query: str, messages=None
    ) -> str:
        """Thin wrapper around ``MemoryManager.get_context()``."""
        sid = self._resolve_sid(session_key)
        return self._mgr.get_context(sid, user_query, messages=messages)

    def record_turn(
        self,
        session_key: str,
        user_message: dict,
        assistant_message: dict,
        token_count: int,
        messages: List[dict],
    ) -> List[dict] | None:
        """Thin wrapper around ``MemoryManager.record_turn()``."""
        sid = self._resolve_sid(session_key)
        result = self._mgr.record_turn(
            session_id=sid,
            user_message=user_message,
            assistant_message=assistant_message,
            token_count=token_count,
            messages=messages,
        )
        return result

    def new_session(self, session_key: str) -> Tuple[str, List[dict]]:
        """Force a fresh session (``/clear``)."""
        sid, messages = self._mgr.new_session()
        self._loaded[session_key] = (sid, messages)
        return sid, messages

    def compact(
        self, session_key: str, messages: List[dict]
    ) -> Tuple[List[dict], bool]:
        """Manually compact the session (``/compact``)."""
        sid = self._resolve_sid(session_key)
        return self._mgr.compact(sid, messages)

    def get_context_usage(self, session_key: str, messages: List[dict]) -> dict:
        """Get context-window usage for the session."""
        return self._mgr.get_context_usage(messages)

    @property
    def last_compression(self) -> dict | None:
        """Stats from the most recent compression run."""
        return self._mgr.last_compression

    # ------------------------------------------------------------------
    def _resolve_sid(self, session_key: str) -> str:
        """Return the internal session_id for *session_key*."""
        if session_key in self._loaded:
            return self._loaded[session_key][0]
        # If not loaded yet, load it now
        sid, messages = self._mgr.load_session(session_key)
        self._loaded[session_key] = (sid, messages)
        return sid

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying MemoryManager."""
        self._mgr.close()
