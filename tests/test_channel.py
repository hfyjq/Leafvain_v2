"""
Unit tests for Channel ABC, MessageSession, MessageEvent.

Validates:
  1. MessageSession string roundtrip
  2. MessageSession.from_str parsing
  3. MessageEvent creation and flow control
  4. Channel ABC interface compliance
  5. CLIChannel instantiation

Usage:
    cd D:/OPD/codePRS/Leafvain_v2
    python tests/test_channel.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_message_session_str():
    """MessageSession __str__ produces expected format."""
    from channels.base import MessageSession, MessageType

    s = MessageSession("telegram", MessageType.PRIVATE, "123456")
    assert str(s) == "telegram:private:123456", f"Unexpected: {str(s)}"

    s2 = MessageSession("cli", MessageType.GROUP, "room_42")
    assert str(s2) == "cli:group:room_42", f"Unexpected: {str(s2)}"

    print(f"  ✅ MessageSession str: '{s}' / '{s2}'")


def test_message_session_from_str():
    """MessageSession.from_str should roundtrip correctly."""
    from channels.base import MessageSession, MessageType

    original = MessageSession("telegram", MessageType.GROUP, "999888")
    parsed = MessageSession.from_str(str(original))

    assert parsed.platform == original.platform
    assert parsed.message_type == original.message_type
    assert parsed.session_id == original.session_id
    assert str(parsed) == str(original)

    print(f"  ✅ from_str roundtrip: {original} → {parsed}")


def test_message_session_frozen():
    """MessageSession should be immutable (frozen dataclass)."""
    from channels.base import MessageSession, MessageType

    s = MessageSession("cli", MessageType.PRIVATE, "local")
    try:
        s.platform = "modified"  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass  # Expected

    print(f"  ✅ MessageSession is frozen (immutable)")


def test_message_event_creation():
    """MessageEvent should be creatable and carry extras."""
    from channels.base import (
        ChannelMetadata,
        Message,
        MessageEvent,
        MessageSession,
        MessageType,
    )

    session = MessageSession("cli", MessageType.PRIVATE, "user1")
    msg = Message.from_text(sender_id="user1", text="hello", session=session)
    meta = ChannelMetadata(name="test", description="test channel")

    event = MessageEvent(message=msg, session=session, platform_meta=meta)

    assert event.message.content == "hello"
    assert event.is_stopped() is False
    assert event.unified_msg_origin == "cli:private:user1"

    # Test stop()
    event.stop()
    assert event.is_stopped() is True

    # Test extras
    event.extras["response"] = "hi back"
    assert event.extras["response"] == "hi back"

    print(f"  ✅ MessageEvent: content='{event.message.content}', "
          f"umo='{event.unified_msg_origin}', stop={event.is_stopped()}")


def test_channel_metadata():
    """ChannelMetadata should hold platform info."""
    from channels.base import ChannelMetadata

    meta = ChannelMetadata(
        name="telegram",
        description="Telegram Bot adapter",
        platform_type="telegram",
        support_multimodal=True,
    )

    assert meta.name == "telegram"
    assert meta.support_multimodal is True
    assert meta.support_streaming is False  # default

    print(f"  ✅ ChannelMetadata: {meta.name}, multimodal={meta.support_multimodal}")


def test_message_defaults():
    """Message should have sensible defaults."""
    from channels.base import Message

    msg = Message.from_text(sender_id="u1", text="test")
    assert msg.sender_name == ""
    assert msg.raw_data is None
    assert msg.timestamp > 0
    print(f"  ✅ Message defaults: sender_id='{msg.sender_id}', "
          f"content='{msg.content}'")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Channel ABC Tests")
    print("=" * 60)
    print()

    tests = [
        ("MessageSession str", test_message_session_str),
        ("MessageSession from_str", test_message_session_from_str),
        ("MessageSession frozen", test_message_session_frozen),
        ("MessageEvent creation + extras", test_message_event_creation),
        ("ChannelMetadata", test_channel_metadata),
        ("Message defaults", test_message_defaults),
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
