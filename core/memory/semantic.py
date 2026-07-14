"""
Semantic memory: SQLite key-value store for user profile and key facts.

Lightweight alternative to a knowledge graph.  Stores
category-tagged facts that survive process restarts.

v0.7.1: Added ``scope`` column (user | project) for two-level
memory separation.  User-scope facts (personal, preference) persist
across projects; project-scope facts (project, decision, knowledge,
etc.) are session/project-bound.
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SemanticMemory:
    """SQLite-backed user profile and key-fact store."""

    def __init__(self, db_path: str = "data/memory/profile.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(
        self,
        key: str,
        value: str,
        category: str = "general",
        confidence: float = 1.0,
        source_turn: int = 0,
        scope: str = "user",
    ) -> None:
        """Insert or update a profile key-value pair."""
        now = _now()
        self._conn.execute(
            "INSERT OR REPLACE INTO user_profile "
            "(key, value, category, confidence, source_turn, scope, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, value, category, confidence, source_turn, scope, now),
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        """Return the value for *key*, or None."""
        row = self._conn.execute(
            "SELECT value FROM user_profile WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def get_profile(self, scope: str | None = None) -> Dict[str, str]:
        """Return all profile entries as {key: value}.

        If *scope* is given, filter to that scope; otherwise return all.
        """
        if scope:
            rows = self._conn.execute(
                "SELECT key, value FROM user_profile "
                "WHERE scope = ? ORDER BY category, key",
                (scope,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, value FROM user_profile ORDER BY category, key"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_by_scope(self, scope: str) -> List[Tuple[str, str, str]]:
        """Return all (key, value, category) pairs for a given scope."""
        rows = self._conn.execute(
            "SELECT key, value, category FROM user_profile "
            "WHERE scope = ? ORDER BY category, key",
            (scope,),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def get_by_category(self, category: str) -> List[Tuple[str, str]]:
        """Return all (key, value) pairs for a category."""
        rows = self._conn.execute(
            "SELECT key, value FROM user_profile WHERE category = ? "
            "ORDER BY key",
            (category,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM user_profile WHERE key = ?", (key,))
        self._conn.commit()

    def all_categories(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT category FROM user_profile ORDER BY category"
        ).fetchall()
        return [r[0] for r in rows]

    def fact_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM user_profile"
        ).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                category    TEXT NOT NULL DEFAULT 'general',
                scope       TEXT NOT NULL DEFAULT 'user',
                confidence  REAL DEFAULT 1.0,
                source_turn INTEGER DEFAULT 0,
                updated_at  TEXT NOT NULL
            )
        """)
        # Migrate: add scope column to existing tables (safe to run on new tables too)
        try:
            self._conn.execute(
                "ALTER TABLE user_profile ADD COLUMN scope TEXT NOT NULL DEFAULT 'user'"
            )
        except sqlite3.OperationalError:
            pass  # column already exists
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_profile_category "
            "ON user_profile(category)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_profile_scope "
            "ON user_profile(scope)"
        )
        self._conn.commit()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
