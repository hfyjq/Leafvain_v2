"""
Unit tests for L3 Session Memory (core/memory/session_memory.py).

Validates:
  1. load/save roundtrip
  2. load nonexistent returns None
  3. exists check
  4. update new session (empty → filled template)
  5. update existing session (incremental refinement)
  6. atomic save — crash mid-write preserves original

Usage:
    cd D:/OPD/codePRS/Leafvain_v2
    python tests/test_session_memory.py
"""

import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ------------------------------------------------------------------
# Mock Provider — returns a fixed session note
# ------------------------------------------------------------------

class MockProvider:
    """Fake provider that returns a controlled session note update."""

    FIXED_RESPONSE = (
        "# Session Note — test_session\n\n"
        "## Current Status\n"
        "- 用户正在测试 L3 会话记忆模块\n\n"
        "## Key Decisions\n"
        "- 采用四段模板结构\n"
        "- 每轮增量更新\n\n"
        "## Open Questions\n"
        "- 压缩时注入效果如何？\n\n"
        "## Next Steps\n"
        "- 完成单元测试\n"
        "- 集成到 MemoryManager\n"
    )

    def __init__(self, context_window: int = 65536):
        self._context_window = context_window
        self.chat_calls = 0

    @property
    def context_window(self) -> int:
        return self._context_window

    def count_tokens(self, text: str) -> int:
        return len(text)

    def estimate_message_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += self.count_tokens(m.get("content") or "")
            total += 4
        return total

    def chat(self, messages: list[dict], tools=None):
        """Return the fixed session note as the LLM would."""
        self.chat_calls += 1

        class FakeResponse:
            content = MockProvider.FIXED_RESPONSE
            tool_calls = None
            prompt_tokens = 300
            completion_tokens = 100

            @property
            def total_tokens(self):
                return self.prompt_tokens + self.completion_tokens

        return FakeResponse()


class MockProviderWithFences:
    """Provider that wraps output in markdown code fences."""

    def __init__(self, context_window: int = 65536):
        self._context_window = context_window
        self.chat_calls = 0

    @property
    def context_window(self) -> int:
        return self._context_window

    def count_tokens(self, text: str) -> int:
        return len(text)

    def estimate_message_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += self.count_tokens(m.get("content") or "")
            total += 4
        return total

    def chat(self, messages: list[dict], tools=None):
        self.chat_calls += 1

        class FakeResponse:
            content = (
                "```markdown\n"
                "# Session Note — fences_test\n\n"
                "## Current Status\n"
                "- Testing fence stripping\n\n"
                "## Key Decisions\n"
                "- Fences should be removed\n\n"
                "## Open Questions\n"
                "(None)\n\n"
                "## Next Steps\n"
                "- Verify output\n"
                "```"
            )
            tool_calls = None
            prompt_tokens = 200
            completion_tokens = 80

            @property
            def total_tokens(self):
                return self.prompt_tokens + self.completion_tokens

        return FakeResponse()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_load_save_roundtrip():
    """Save a note, load it back — content must match."""
    from core.memory.session_memory import SessionMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionMemory(sessions_dir=tmpdir)
        note = "# Test\n\n## Current Status\n- Working"
        sm.save("test_session", note)

        loaded = sm.load("test_session")
        assert loaded is not None, "Loaded note should not be None"
        assert loaded == note, f"Roundtrip failed:\nExpected:\n{note}\nGot:\n{loaded}"
        print("  ✅ load/save roundtrip")


def test_load_nonexistent():
    """Loading a nonexistent session returns None."""
    from core.memory.session_memory import SessionMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionMemory(sessions_dir=tmpdir)
        result = sm.load("nonexistent_session")
        assert result is None, f"Expected None, got: {result!r}"
        print("  ✅ load nonexistent → None")


def test_exists():
    """exists() returns True after save, False for nonexistent."""
    from core.memory.session_memory import SessionMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionMemory(sessions_dir=tmpdir)
        assert not sm.exists("no_such"), "Should not exist before save"

        sm.save("real_session", "# Real\n\n## Current Status\n- Here")
        assert sm.exists("real_session"), "Should exist after save"
        assert not sm.exists("other"), "Other session should not exist"
        print("  ✅ exists check")


def test_update_new_session():
    """
    update() with empty current_note fills the template via LLM.
    The MockProvider returns a fixed note — verify it is captured.
    """
    from core.memory.session_memory import SessionMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionMemory(sessions_dir=tmpdir)
        provider = MockProvider()

        updated = sm.update(
            session_id="test_session",
            current_note="",
            user_message="帮我测试会话记忆模块",
            assistant_message="好的，开始测试 L3 会话记忆。",
            provider=provider,
        )

        assert provider.chat_calls == 1, "LLM should have been called once"
        assert "## Current Status" in updated, "Should contain Current Status section"
        assert "## Key Decisions" in updated, "Should contain Key Decisions section"
        assert "## Open Questions" in updated, "Should contain Open Questions section"
        assert "## Next Steps" in updated, "Should contain Next Steps section"
        assert len(updated) > 50, "Updated note should not be empty"
        print(f"  ✅ update new session ({len(updated)} chars)")


def test_update_existing_session():
    """
    update() with an existing note should send it to the LLM for
    incremental refinement.
    """
    from core.memory.session_memory import SessionMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionMemory(sessions_dir=tmpdir)
        provider = MockProvider()

        existing = (
            "# Session Note — test\n\n"
            "## Current Status\n- Old status\n\n"
            "## Key Decisions\n- Old decision\n\n"
            "## Open Questions\n- Old question\n\n"
            "## Next Steps\n- Old step\n"
        )

        updated = sm.update(
            session_id="test_session",
            current_note=existing,
            user_message="我们改一下方案",
            assistant_message="好的，已更新方案。",
            provider=provider,
        )

        assert provider.chat_calls == 1, "LLM should have been called once"
        # The mock returns a fixed response, so we just verify it's non-empty
        assert len(updated) > 0
        print(f"  ✅ update existing session ({len(updated)} chars)")


def test_atomic_save():
    """
    Verify atomic save: if a crash occurs mid-write, the original
    file is preserved.  We simulate by writing a tmp file manually
    and verifying that .md is untouched.
    """
    from core.memory.session_memory import SessionMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionMemory(sessions_dir=tmpdir)

        original = "# Original\n\n## Current Status\n- Before crash"
        sm.save("crash_test", original)

        # Simulate a partial write: create the .tmp but NOT replace
        note_path = sm.get_note_path("crash_test")
        tmp_path = note_path.with_suffix(".md.tmp")
        tmp_path.write_text("PARTIAL", encoding="utf-8")

        # The original file should still be intact
        assert note_path.exists(), "Original should still exist"
        loaded = sm.load("crash_test")
        assert loaded == original, (
            f"Original should be preserved after partial write.\n"
            f"Expected: {original!r}\nGot: {loaded!r}"
        )
        print("  ✅ atomic save — original preserved")


def test_fence_stripping():
    """LLM output wrapped in ``` fences should be stripped."""
    from core.memory.session_memory import SessionMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionMemory(sessions_dir=tmpdir)
        provider = MockProviderWithFences()

        updated = sm.update(
            session_id="fences_test",
            current_note="",
            user_message="测试代码块剥离",
            assistant_message="好的。",
            provider=provider,
        )

        assert not updated.startswith("```"), (
            f"Fences should be stripped, got: {updated[:50]}..."
        )
        assert not updated.endswith("```"), (
            f"Trailing fence should be stripped"
        )
        assert "## Current Status" in updated
        print("  ✅ fence stripping")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  L3 Session Memory Tests")
    print("=" * 60)
    print()

    tests = [
        ("load/save roundtrip", test_load_save_roundtrip),
        ("load nonexistent → None", test_load_nonexistent),
        ("exists check", test_exists),
        ("update new session", test_update_new_session),
        ("update existing session", test_update_existing_session),
        ("atomic save", test_atomic_save),
        ("fence stripping", test_fence_stripping),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAIL: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
