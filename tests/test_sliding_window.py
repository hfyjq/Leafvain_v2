"""
Unit tests for sliding window safety net (P3).

Validates:
  1. Sliding window NOT triggered when under max_history
  2. Sliding window triggers when message count exceeds max_history
  3. Leading system messages are preserved after truncation
  4. Truncated list length equals max_history
  5. Integration: record_turn applies sliding window after compression

Usage:
    cd D:/OPD/codePRS/Leafvain_v2
    python tests/test_sliding_window.py
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ------------------------------------------------------------------
# Mock Provider
# ------------------------------------------------------------------

class MockProvider:
    """Fake provider for sliding window tests."""

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
            total += 4  # role overhead
        return total

    def chat(self, messages: list[dict], tools=None):
        self.chat_calls += 1

        class FakeResponse:
            content = "Mock summary."
            tool_calls = None
            prompt_tokens = 100
            completion_tokens = 20

            @property
            def total_tokens(self):
                return self.prompt_tokens + self.completion_tokens

        return FakeResponse()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_messages(count: int, with_system: bool = True) -> list[dict]:
    """Build a message list with *count* user↔assistant exchanges."""
    messages = []
    if with_system:
        messages.append({
            "role": "system",
            "content": "You are a document analysis assistant. " * 5,
        })
    for i in range(count):
        messages.append({
            "role": "user",
            "content": f"[Turn {i}] 用户提问内容 " * 3,
        })
        messages.append({
            "role": "assistant",
            "content": f"[Turn {i}] 助手回复内容 " * 3,
        })
    return messages


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_below_limit_untouched():
    """Messages under max_history should not be truncated."""
    from core.memory.manager import MemoryManager

    provider = MockProvider(context_window=65536)
    config = {
        "memory": {
            "enabled": True,
            "storage_path": ":unused:",
            "short_term": {"max_history": 100},
        }
    }

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        config["memory"]["storage_path"] = tmpdir
        config["memory"]["compression"] = {
            "token_threshold": 0.85, "keep_recent_turns": 3,
        }
        config["memory"]["long_term"] = {"chroma_path": f"{tmpdir}/chroma"}
        config["memory"]["semantic"] = {"db_path": f"{tmpdir}/profile.db"}

        manager = MemoryManager(config, provider)

        messages = make_messages(5)  # 1 system + 10 turn msgs = 11 total
        truncated, was_truncated = manager._apply_sliding_window(
            "test", messages,
        )

        assert not was_truncated, (
            f"Should NOT truncate {len(messages)} messages (max_history=100)"
        )
        assert truncated == messages, "Should return original list unchanged"

        print(f"  ✅ Below limit: {len(messages)} messages untouched")
        manager.close()


def test_over_limit_truncated():
    """Messages exceeding max_history should be truncated."""
    from core.memory.manager import MemoryManager

    provider = MockProvider(context_window=65536)
    config = {
        "memory": {
            "enabled": True,
            "storage_path": ":unused:",
            "short_term": {"max_history": 10},
        }
    }

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        config["memory"]["storage_path"] = tmpdir
        config["memory"]["compression"] = {
            "token_threshold": 0.85, "keep_recent_turns": 3,
        }
        config["memory"]["long_term"] = {"chroma_path": f"{tmpdir}/chroma"}
        config["memory"]["semantic"] = {"db_path": f"{tmpdir}/profile.db"}

        manager = MemoryManager(config, provider)

        messages = make_messages(20)  # 1 system + 40 turn msgs = 41 total
        assert len(messages) > 10, "Test requires > 10 messages"

        truncated, was_truncated = manager._apply_sliding_window(
            "test", messages,
        )

        assert was_truncated, (
            f"Should truncate {len(messages)} messages (max_history=10)"
        )
        assert len(truncated) <= 10, (
            f"Truncated list ({len(truncated)}) should be ≤ max_history (10)"
        )
        assert len(truncated) < len(messages), (
            f"Truncated ({len(truncated)}) < original ({len(messages)})"
        )

        print(f"  ✅ Over limit: {len(messages)} → {len(truncated)} messages")
        manager.close()


def test_system_messages_preserved():
    """Leading system messages must survive truncation."""
    from core.memory.manager import MemoryManager

    provider = MockProvider(context_window=65536)
    config = {
        "memory": {
            "enabled": True,
            "storage_path": ":unused:",
            "short_term": {"max_history": 8},
        }
    }

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        config["memory"]["storage_path"] = tmpdir
        config["memory"]["compression"] = {
            "token_threshold": 0.85, "keep_recent_turns": 3,
        }
        config["memory"]["long_term"] = {"chroma_path": f"{tmpdir}/chroma"}
        config["memory"]["semantic"] = {"db_path": f"{tmpdir}/profile.db"}

        manager = MemoryManager(config, provider)

        messages = [
            {"role": "system", "content": "PRIMARY INSTRUCTIONS"},
            {"role": "system", "content": "<historical_context>PAST SUMMARY</historical_context>"},
        ]
        # Add enough turn messages to exceed max_history
        for i in range(20):
            messages.append({"role": "user", "content": f"Q{i}"})
            messages.append({"role": "assistant", "content": f"A{i}"})

        truncated, was_truncated = manager._apply_sliding_window(
            "test", messages,
        )

        assert was_truncated
        assert truncated[0]["role"] == "system", "First message must be system"
        assert truncated[0]["content"] == "PRIMARY INSTRUCTIONS", (
            "Primary instructions must be preserved"
        )
        assert truncated[1]["role"] == "system", "Second message must be system"
        assert "<historical_context>" in truncated[1]["content"], (
            "Historical context must be preserved"
        )

        print(f"  ✅ System messages preserved: "
              f"head={truncated[0]['content'][:30]}... + "
              f"head={truncated[1]['content'][:30]}...")
        manager.close()


def test_single_system_message_truncation():
    """With one leading system message, keep it + last (N-1) messages."""
    from core.memory.manager import MemoryManager

    provider = MockProvider(context_window=65536)
    config = {
        "memory": {
            "enabled": True,
            "storage_path": ":unused:",
            "short_term": {"max_history": 6},
        }
    }

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        config["memory"]["storage_path"] = tmpdir
        config["memory"]["compression"] = {
            "token_threshold": 0.85, "keep_recent_turns": 3,
        }
        config["memory"]["long_term"] = {"chroma_path": f"{tmpdir}/chroma"}
        config["memory"]["semantic"] = {"db_path": f"{tmpdir}/profile.db"}

        manager = MemoryManager(config, provider)

        messages = make_messages(10, with_system=True)  # 1 system + 20 turn = 21

        truncated, was_truncated = manager._apply_sliding_window(
            "test", messages,
        )

        assert was_truncated
        assert len(truncated) == 6, (
            f"Expected exactly 6 messages, got {len(truncated)}"
        )
        assert truncated[0]["role"] == "system", "First message must be system"
        assert truncated[0] == messages[0], (
            "System message content must be identical to original"
        )

        print(f"  ✅ Single system: {len(messages)} → {len(truncated)}, "
              f"system msg intact")
        manager.close()


def test_default_max_history():
    """When config omits max_history, default to 100."""
    from core.memory.manager import MemoryManager

    provider = MockProvider(context_window=65536)
    config = {
        "memory": {
            "enabled": True,
            "storage_path": ":unused:",
            # short_term present but no max_history key
            "short_term": {"max_messages": 500},
        }
    }

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        config["memory"]["storage_path"] = tmpdir
        config["memory"]["compression"] = {
            "token_threshold": 0.85, "keep_recent_turns": 3,
        }
        config["memory"]["long_term"] = {"chroma_path": f"{tmpdir}/chroma"}
        config["memory"]["semantic"] = {"db_path": f"{tmpdir}/profile.db"}

        manager = MemoryManager(config, provider)
        assert manager._max_history == 100, (
            f"Default max_history should be 100, got {manager._max_history}"
        )
        print(f"  ✅ Default max_history: {manager._max_history}")
        manager.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Sliding Window Safety Net Tests (P3)")
    print("=" * 60)
    print()

    tests = [
        ("Below limit untouched", test_below_limit_untouched),
        ("Over limit truncated", test_over_limit_truncated),
        ("System messages preserved", test_system_messages_preserved),
        ("Single system message truncation", test_single_system_message_truncation),
        ("Default max_history = 100", test_default_max_history),
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
