"""
Unit test for mid-term compression logic.

Validates:
  1. Token estimation triggers at the right threshold
  2. Turn boundary detection
  3. Compression produces shorter message list
  4. Backup is persisted to disk
  5. Context usage display helpers
  6. Post-compression _saved_count reset

This test uses a MockProvider and small context_window so
compression triggers after a few turns — no long conversation needed.

Usage:
    cd D:/OPD/codePRS/Leafvain_v2
    python tests/test_compressor.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ------------------------------------------------------------------
# Mock Provider — simulates LLM for compression summary
# ------------------------------------------------------------------

class MockProvider:
    """Fake provider that returns a fixed summary."""

    def __init__(self, context_window: int = 4000):
        self._context_window = context_window
        self.chat_calls = 0  # track how many times chat() was called

    @property
    def context_window(self) -> int:
        return self._context_window

    def count_tokens(self, text: str) -> int:
        """Naive token counter: ~1 token per char for Chinese, ~0.25 for English."""
        return len(text)

    def estimate_message_tokens(self, messages: list[dict]) -> int:
        """Same as BaseProvider.estimate_message_tokens."""
        total = 0
        for m in messages:
            total += self.count_tokens(m.get("content") or "")
            total += 4  # role overhead
        return total

    def chat(self, messages: list[dict], tools=None):
        """Return a mock summary."""
        self.chat_calls += 1
        # Return a ChatResponse-like object
        class FakeResponse:
            content = "用户讨论了文档分析系统的压缩功能测试，涉及多个技术话题。"
            tool_calls = None
            prompt_tokens = 200
            completion_tokens = 30

            @property
            def total_tokens(self):
                return self.prompt_tokens + self.completion_tokens

        return FakeResponse()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_long_conversation(turns: int) -> list[dict]:
    """
    Build a synthetic conversation with *turns* user↔assistant exchanges.

    Each message has ~200 chars so we can control when the threshold is hit.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a document analysis assistant with persistent memory. " * 3,
        }
    ]
    for i in range(turns):
        messages.append({
            "role": "user",
            "content": f"[Turn {i}] 请帮我分析这份文档的内容，提取关键信息。关于第{i}章的细节，我需要了解更多上下文。这些数据是否与之前的发现一致？" * 2,
        })
        messages.append({
            "role": "assistant",
            "content": f"[Turn {i}] 根据分析，第{i}章的主要内容涉及几个重要主题。首先是关于数据一致性的讨论，其次是方法论的验证。结果表明之前的假设基本成立。" * 2,
        })
    return messages


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_estimate_message_tokens():
    """Verify token estimation includes role overhead."""
    provider = MockProvider(context_window=4000)
    messages = make_long_conversation(1)
    total = provider.estimate_message_tokens(messages)
    assert total > 0, "Token estimate should be > 0"
    print(f"  ✅ estimate_message_tokens: {total} tokens for {len(messages)} messages")


def test_compression_triggers():
    """Compression should trigger when token estimate > 85% of context_window."""
    from core.memory.compressor import Compressor

    provider = MockProvider(context_window=2000)  # small window → easy trigger
    compressor = Compressor(threshold=0.85, keep_recent_turns=3)

    # Build conversation that exceeds threshold
    messages = make_long_conversation(8)  # ~8 * (200+200) + overhead ≈ 3200+ tokens > 1700
    total = provider.estimate_message_tokens(messages)
    threshold = int(2000 * 0.85)

    print(f"  total_tokens={total}, threshold={threshold}")
    assert compressor.should_compress(total, 2000), \
        f"should_compress should return True: {total} > {threshold}"

    # Run compression
    compressed, was_compressed = compressor.check_and_compress(
        session_id="test_session_1",
        messages=messages,
        provider=provider,
        context_window=2000,
    )

    assert was_compressed, "Compression should have been triggered"
    assert len(compressed) < len(messages), \
        f"Compressed list ({len(compressed)}) should be shorter than original ({len(messages)})"
    assert compressed[0]["role"] == "system", \
        "First message should be a system message with <historical_context>"
    assert "<historical_context>" in compressed[0]["content"], \
        "System message should contain <historical_context> tag"

    # Verify stats
    assert compressor.last_compression is not None
    stats = compressor.last_compression
    assert stats["compressed_turns"] > 0
    assert stats["old_message_count"] == len(messages)
    assert stats["new_message_count"] == len(compressed)

    print(f"  ✅ Compression triggered: {len(messages)} → {len(compressed)} messages")
    print(f"     Stats: {stats}")


def test_compression_not_triggered_when_below_threshold():
    """Compression should NOT trigger when token estimate < 85% of context_window."""
    from core.memory.compressor import Compressor

    provider = MockProvider(context_window=100000)  # huge window
    compressor = Compressor(threshold=0.85, keep_recent_turns=3)

    messages = make_long_conversation(3)
    compressed, was_compressed = compressor.check_and_compress(
        session_id="test_session_2",
        messages=messages,
        provider=provider,
        context_window=100000,
    )

    assert not was_compressed, "Compression should NOT trigger below threshold"
    assert compressed == messages, "Should return original messages unchanged"
    print(f"  ✅ Below threshold: compression correctly skipped")


def test_backup_persisted():
    """Compressed summary should be written to disk."""
    from core.memory.compressor import Compressor

    with tempfile.TemporaryDirectory() as tmpdir:
        provider = MockProvider(context_window=1500)
        compressor = Compressor(
            threshold=0.85,
            keep_recent_turns=2,
            backup_dir=tmpdir,
        )

        messages = make_long_conversation(6)
        compressed, was_compressed = compressor.check_and_compress(
            session_id="backup_test",
            messages=messages,
            provider=provider,
            context_window=1500,
        )

        assert was_compressed

        # Verify backup file exists
        backup_file = Path(tmpdir) / "backup_test.json"
        assert backup_file.exists(), f"Backup file should exist at {backup_file}"

        # Verify content
        import json
        data = json.loads(backup_file.read_text(encoding="utf-8"))
        assert data["session_id"] == "backup_test"
        assert len(data["summary"]) > 0
        print(f"  ✅ Backup persisted: {backup_file}")
        print(f"     Summary ({len(data['summary'])} chars): {data['summary'][:80]}...")


def test_load_compressed_context():
    """Restoring a compressed context should return the summary."""
    from core.memory.compressor import Compressor

    with tempfile.TemporaryDirectory() as tmpdir:
        provider = MockProvider(context_window=1500)
        compressor = Compressor(
            threshold=0.85,
            keep_recent_turns=2,
            backup_dir=tmpdir,
        )

        messages = make_long_conversation(6)
        compressor.check_and_compress(
            session_id="restore_test",
            messages=messages,
            provider=provider,
            context_window=1500,
        )

        # Now try to restore
        restored = compressor.load_compressed_context("restore_test")
        assert restored is not None, "Should restore compressed context"
        assert len(restored) > 0
        print(f"  ✅ Context restored: {restored[:80]}...")

        # Non-existent session should return None
        missing = compressor.load_compressed_context("nonexistent")
        assert missing is None
        print(f"  ✅ Missing session returns None")


def test_context_usage_display():
    """Test MemoryManager.get_context_usage() returns correct structure."""
    from core.memory.manager import MemoryManager

    provider = MockProvider(context_window=65536)

    # Create a minimal config
    config = {
        "memory": {
            "enabled": True,
            "storage_path": ":memory:",  # won't be used in this test
        }
    }

    # ignore_cleanup_errors=True: on Windows, ChromaDB SQLite WAL locks
    # may not release instantly even after _system.stop(); this prevents
    # the test from failing on cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # Point all storage paths to tmpdir
        config["memory"]["storage_path"] = tmpdir
        config["memory"]["short_term"] = {"max_messages": 500}
        config["memory"]["compression"] = {"token_threshold": 0.85, "keep_recent_turns": 3}
        config["memory"]["long_term"] = {"chroma_path": f"{tmpdir}/chroma"}
        config["memory"]["semantic"] = {"db_path": f"{tmpdir}/profile.db"}

        manager = MemoryManager(config, provider)

        messages = make_long_conversation(5)
        usage = manager.get_context_usage(messages)

        assert "estimated_tokens" in usage
        assert "context_window" in usage
        assert "percentage" in usage
        assert "threshold_pct" in usage
        assert usage["context_window"] == 65536
        assert usage["percentage"] > 0
        assert usage["threshold_pct"] == 85.0

        print(f"  ✅ Context usage: {usage['percentage']}% "
              f"({usage['estimated_tokens']}/{usage['context_window']} tokens)")

        manager.close()


def test_note_bypasses_llm():
    """
    When session_note is provided, compression MUST NOT call the LLM.
    When session_note is empty/missing, it MUST fall back to the LLM.
    """
    from core.memory.compressor import Compressor

    with tempfile.TemporaryDirectory() as tmpdir:
        # ---- Case 1: with session_note → no LLM call ----
        provider1 = MockProvider(context_window=1500)
        compressor1 = Compressor(
            threshold=0.85, keep_recent_turns=2, backup_dir=tmpdir,
        )

        messages = make_long_conversation(6)
        session_note = (
            "# Session Note\n\n"
            "## Current Status\n- Testing bypass\n\n"
            "## Key Decisions\n- Use session note\n\n"
            "## Open Questions\n(None)\n\n"
            "## Next Steps\n- Verify\n"
        )

        calls_before = provider1.chat_calls
        compressed, was_compressed = compressor1.check_and_compress(
            session_id="bypass_test_1",
            messages=messages,
            provider=provider1,
            context_window=1500,
            session_note=session_note,
        )

        assert was_compressed, "Compression should trigger"
        assert provider1.chat_calls == calls_before, (
            f"LLM should NOT be called when session_note is provided "
            f"(calls: {calls_before} → {provider1.chat_calls})"
        )
        assert "<historical_context>" in compressed[0]["content"]
        assert "Testing bypass" in compressed[0]["content"]
        assert compressor1.last_compression["source"] == "session_note"
        print("  ✅ session_note bypasses LLM (zero API cost)")

        # ---- Case 2: without session_note → fall back to LLM ----
        provider2 = MockProvider(context_window=1500)
        compressor2 = Compressor(
            threshold=0.85, keep_recent_turns=2, backup_dir=tmpdir,
        )

        calls_before2 = provider2.chat_calls
        compressed2, was_compressed2 = compressor2.check_and_compress(
            session_id="bypass_test_2",
            messages=messages,
            provider=provider2,
            context_window=1500,
            session_note=None,  # explicitly None
        )

        assert was_compressed2, "Compression should trigger"
        assert provider2.chat_calls > calls_before2, (
            f"LLM SHOULD be called when no session_note "
            f"(calls: {calls_before2} → {provider2.chat_calls})"
        )
        assert compressor2.last_compression["source"] == "llm"
        print("  ✅ no session_note → LLM fallback works")

        # ---- Case 3: empty session_note → fall back to LLM ----
        provider3 = MockProvider(context_window=1500)
        compressor3 = Compressor(
            threshold=0.85, keep_recent_turns=2, backup_dir=tmpdir,
        )

        calls_before3 = provider3.chat_calls
        compressed3, was_compressed3 = compressor3.check_and_compress(
            session_id="bypass_test_3",
            messages=messages,
            provider=provider3,
            context_window=1500,
            session_note="   ",  # whitespace only
        )

        assert was_compressed3
        assert provider3.chat_calls > calls_before3, (
            "LLM SHOULD be called when session_note is whitespace-only"
        )
        assert compressor3.last_compression["source"] == "llm"
        print("  ✅ whitespace session_note → LLM fallback works")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Mid-term Compression Tests")
    print("=" * 60)
    print()

    tests = [
        ("Token estimation", test_estimate_message_tokens),
        ("Compression triggers above threshold", test_compression_triggers),
        ("No compression below threshold", test_compression_not_triggered_when_below_threshold),
        ("Backup persisted to disk", test_backup_persisted),
        ("Restore compressed context", test_load_compressed_context),
        ("Context usage display", test_context_usage_display),
        ("Session note bypasses LLM", test_note_bypasses_llm),
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
