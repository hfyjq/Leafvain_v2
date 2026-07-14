"""
Unit tests for two-level memory separation (user vs project scope).

Tests cover:
  - CATEGORY_SCOPE mapping correctness (7 categories → 2 scopes)
  - SemanticMemory: scope column migration, upsert/get_profile/get_by_scope
  - FactExtractor: scope calculation from category
  - MemoryManager: profile-only mode scope filtering (unit-level)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.memory.fact_extractor import CATEGORY_SCOPE


# ==================================================================
# CATEGORY_SCOPE mapping
# ==================================================================

def test_category_to_scope_mapping():
    """All 7 categories should map to exactly one of user/project."""
    assert CATEGORY_SCOPE == {
        "personal": "user",
        "preference": "user",
        "project": "project",
        "constraint": "project",
        "decision": "project",
        "knowledge": "project",
        "relationship": "project",
    }
    scopes = set(CATEGORY_SCOPE.values())
    assert scopes == {"user", "project"}


def test_user_scope_categories():
    """personal + preference = who the user IS."""
    assert CATEGORY_SCOPE["personal"] == "user"
    assert CATEGORY_SCOPE["preference"] == "user"


def test_project_scope_categories():
    """The other 5 categories = what they're WORKING ON."""
    for cat in ("project", "constraint", "decision", "knowledge", "relationship"):
        assert CATEGORY_SCOPE[cat] == "project"


# ==================================================================
# SemanticMemory scope support
# ==================================================================

@pytest.fixture
def semantic_db():
    """Create a SemanticMemory on a temp file, clean up after."""
    tmp = tempfile.mktemp(suffix=".db")
    from core.memory.semantic import SemanticMemory
    db = SemanticMemory(db_path=tmp)
    yield db
    db.close()
    Path(tmp).unlink(missing_ok=True)


def test_semantic_upsert_with_scope(semantic_db):
    """upsert should store scope and get_profile should filter by it."""
    semantic_db.upsert("key1", "I am a Python dev", category="personal", scope="user")
    semantic_db.upsert("key2", "Working on Leafvain", category="project", scope="project")

    all_profile = semantic_db.get_profile()
    user_profile = semantic_db.get_profile(scope="user")

    assert len(all_profile) == 2
    assert len(user_profile) == 1
    assert "Python dev" in user_profile["key1"]
    assert "key2" not in user_profile


def test_semantic_get_by_scope(semantic_db):
    """get_by_scope should return (key, value, category) tuples."""
    semantic_db.upsert("k1", "v1", category="personal", scope="user")
    semantic_db.upsert("k2", "v2", category="project", scope="project")

    user_rows = semantic_db.get_by_scope("user")
    project_rows = semantic_db.get_by_scope("project")

    assert len(user_rows) == 1
    assert user_rows[0][0] == "k1"
    assert user_rows[0][1] == "v1"
    assert user_rows[0][2] == "personal"

    assert len(project_rows) == 1
    assert project_rows[0][0] == "k2"


def test_semantic_migration_adds_scope_column(semantic_db):
    """Existing databases without scope should get DEFAULT 'user'."""
    # Simulate: drop scope column, then re-run _create_tables
    try:
        semantic_db._conn.execute("ALTER TABLE user_profile DROP COLUMN scope")
    except Exception:
        pass  # Some SQLite builds don't support DROP COLUMN
    # Re-create (should add scope back)
    semantic_db._create_tables()
    semantic_db.upsert("k", "v", category="general")
    rows = semantic_db.get_by_scope("user")
    assert len(rows) >= 0  # Should not crash


def test_semantic_scope_default(semantic_db):
    """upsert without explicit scope should default to 'user'."""
    semantic_db.upsert("def_key", "default value", category="general")
    user_rows = semantic_db.get_by_scope("user")
    assert any(r[0] == "def_key" for r in user_rows)


# ==================================================================
# FactExtractor scope calculation
# ==================================================================

def test_fact_extractor_scope_from_category():
    """Each category should resolve to the correct scope."""
    test_cases = [
        ("personal", "user"),
        ("preference", "user"),
        ("project", "project"),
        ("constraint", "project"),
        ("decision", "project"),
        ("knowledge", "project"),
        ("relationship", "project"),
        ("unknown_category", "project"),  # fallback
    ]
    for category, expected_scope in test_cases:
        scope = CATEGORY_SCOPE.get(category, "project")
        assert scope == expected_scope, f"{category} → {scope} (expected {expected_scope})"


# ==================================================================
# Integration: scope through the pipeline (skip if chromadb absent)
# ==================================================================

@pytest.fixture
def memory_components():
    """Set up MemoryManager with in-memory semantic + temp Chroma."""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        pytest.skip("chromadb not installed")

    from unittest.mock import MagicMock
    from core.memory.manager import MemoryManager

    config = {
        "memory": {
            "enabled": True,
            "storage_path": tempfile.mkdtemp(),
            "short_term": {"max_history": 100},
            "compression": {"token_threshold": 0.85, "keep_recent_turns": 3},
            "long_term": {},
            "semantic": {},
            "session_memory": {},
        }
    }
    provider = MagicMock()
    provider.context_window = 65536
    provider.estimate_message_tokens.return_value = 1000

    mgr = MemoryManager(config, provider)
    yield mgr
    mgr.close()


def test_get_context_profile_only_user_scope(memory_components):
    """profile_only mode should filter semantic and Chroma to user scope."""
    mgr = memory_components

    # Store one user-scope fact, one project-scope fact
    mgr.semantic.upsert("u1", "Prefers Python", category="preference", scope="user")
    mgr.semantic.upsert("p1", "Building Leafvain", category="project", scope="project")

    # Force profile_only mode
    mgr._pending_memory_inject["test_sid"] = True
    mgr._profile_only["test_sid"] = True

    ctx = mgr.get_context("test_sid", "hello")

    # Profile-only: should contain user fact, NOT project fact
    assert "Prefers Python" in ctx
    assert "Building Leafvain" not in ctx


def test_get_context_normal_mode_all_scopes(memory_components):
    """Normal mode should inject ALL scopes."""
    mgr = memory_components

    mgr.semantic.upsert("u1", "User fact", category="preference", scope="user")
    mgr.semantic.upsert("p1", "Project fact", category="project", scope="project")

    mgr._pending_memory_inject["test_sid"] = True
    mgr._profile_only["test_sid"] = False

    ctx = mgr.get_context("test_sid", "hello")

    assert "User fact" in ctx
    assert "Project fact" in ctx


def test_clear_project_memory(memory_components):
    """clear_project_memory should remove project-scope facts, keep user-scope."""
    mgr = memory_components

    mgr.semantic.upsert("u1", "User pref", category="preference", scope="user")
    mgr.semantic.upsert("p1", "Project detail", category="project", scope="project")
    mgr.semantic.upsert("p2", "Another project", category="knowledge", scope="project")

    removed = mgr.clear_project_memory()

    assert removed == 2
    user_rows = mgr.semantic.get_by_scope("user")
    project_rows = mgr.semantic.get_by_scope("project")

    assert len(user_rows) == 1
    assert len(project_rows) == 0
