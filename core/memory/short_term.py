"""
Short-term memory: SQLite-backed message store with session management.

Messages are saved after every turn and can be restored on startup
so conversations survive process restarts.

Tables:
  sessions  — one row per conversation session
  messages  — one row per message in a session
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ShortTermMemory:
    """SQLite-backed conversation message store."""

    def __init__(self, db_path: str = "data/memory/short_term.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def init_session(self, session_id: str | None = None) -> str:
        """
        Create a new session or verify an existing one.

        Returns the session_id.
        """
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:12]}"

        now = _now()
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) "
            "VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        self._conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        self._conn.commit()
        return session_id

    # ------------------------------------------------------------------
    # Message CRUD
    # ------------------------------------------------------------------

    def save_message(
        self,
        session_id: str,
        message: Dict,
        token_count: int = 0,
    ) -> None:
        """Persist a single message to the database."""
        role = message.get("role", "")
        content = message.get("content") or ""
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = json.dumps(message["tool_calls"], ensure_ascii=False)

        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, "
            "token_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, tool_calls, token_count, _now()),
        )
        self._conn.commit()

    def save_messages(
        self,
        session_id: str,
        messages: List[Dict],
        token_counts: List[int] | None = None,
    ) -> None:
        """Persist multiple messages in a single transaction."""
        if token_counts is None:
            token_counts = [0] * len(messages)

        with self._conn:
            for msg, tc in zip(messages, token_counts):
                self.save_message(session_id, msg, tc)

    def load_messages(self, session_id: str) -> List[Dict]:
        """
        Load ALL messages for a session, ordered by insertion time.

        Returns an empty list if the session does not exist.
        """
        rows = self._conn.execute(
            "SELECT role, content, tool_calls FROM messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()

        messages: List[Dict] = []
        for role, content, tool_calls_json in rows:
            msg: Dict = {"role": role, "content": content}
            if tool_calls_json:
                msg["tool_calls"] = json.loads(tool_calls_json)
            messages.append(msg)
        return messages

    def get_token_total(self, session_id: str) -> int:
        """Return the sum of token_count for a session."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM messages "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def session_exists(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                token_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session "
            "ON messages(session_id, id)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
