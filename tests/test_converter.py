"""
Unit tests for channels.converter — MessageConverter ABC + concrete converters.

Tests cover:
  - CLIMessageConverter: to_internal(str) → MessageChain
  - CLIMessageConverter: to_platform(MessageChain) → str
  - TelegramMessageConverter (skipped if python-telegram-bot absent)
  - MessageConverter ABC cannot be instantiated directly
"""

from __future__ import annotations

import pytest

from channels.converter import MessageConverter
from channels.message import MessageChain

# Detect optional telegram dependency
try:
    import telegram  # noqa: F401
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


# ==================================================================
# CLI Converter
# ==================================================================


def test_cli_converter_to_internal():
    from channels.cli_channel.handler import CLIMessageConverter
    conv = CLIMessageConverter()
    chain = conv.to_internal("hello world")
    assert chain.content == "hello world"
    assert len(chain.components) == 1


def test_cli_converter_to_internal_strips():
    from channels.cli_channel.handler import CLIMessageConverter
    conv = CLIMessageConverter()
    chain = conv.to_internal("  spaced  ")
    assert chain.content == "spaced"


def test_cli_converter_to_platform():
    from channels.cli_channel.handler import CLIMessageConverter
    conv = CLIMessageConverter()
    chain = MessageChain.from_text("response")
    result = conv.to_platform(chain)
    assert isinstance(result, str)
    assert result == "response"


def test_cli_converter_empty():
    from channels.cli_channel.handler import CLIMessageConverter
    conv = CLIMessageConverter()
    chain = conv.to_internal("")
    assert chain.content == ""
    assert chain.is_empty is False  # empty string is still a Plain component


# ==================================================================
# Telegram Converter
# ==================================================================


class _MockTgMessage:
    """Minimal mock of telegram.Message for converter tests."""
    text: str | None = None
    caption: str | None = None
    photo: list | None = None
    document: object | None = None


class _MockPhotoSize:
    file_id: str
    width: int
    height: int
    def __init__(self, file_id: str, w: int = 800, h: int = 600):
        self.file_id = file_id
        self.width = w
        self.height = h


class _MockDocument:
    file_id: str
    file_name: str | None
    mime_type: str | None
    file_size: int | None
    def __init__(self, file_id: str, file_name: str | None = None,
                 mime_type: str | None = None, file_size: int | None = None):
        self.file_id = file_id
        self.file_name = file_name
        self.mime_type = mime_type
        self.file_size = file_size


def _make_update(tg_msg):
    """Wrap a mock tg message in a fake Update-like object."""
    class Update:
        message = tg_msg
    return Update()


@pytest.mark.skipif(not HAS_TELEGRAM, reason="python-telegram-bot not installed")
def test_telegram_converter_text():
    from channels.telegram_channel.handler import TelegramMessageConverter
    conv = TelegramMessageConverter()
    tg_msg = _MockTgMessage()
    tg_msg.text = "hello from telegram"
    update = _make_update(tg_msg)
    chain = conv.to_internal(update)
    assert chain.content == "hello from telegram"


@pytest.mark.skipif(not HAS_TELEGRAM, reason="python-telegram-bot not installed")
def test_telegram_converter_photo():
    from channels.telegram_channel.handler import TelegramMessageConverter
    from channels.message import Image
    conv = TelegramMessageConverter()
    tg_msg = _MockTgMessage()
    tg_msg.photo = [_MockPhotoSize("file_001", 100, 80),
                    _MockPhotoSize("file_002", 800, 600)]
    tg_msg.caption = "check this out"
    update = _make_update(tg_msg)
    chain = conv.to_internal(update)
    # Should have text + image
    assert "check this out" in chain.content
    assert len(chain.images) == 1


@pytest.mark.skipif(not HAS_TELEGRAM, reason="python-telegram-bot not installed")
def test_telegram_converter_document():
    from channels.telegram_channel.handler import TelegramMessageConverter
    from channels.message import File
    conv = TelegramMessageConverter()
    tg_msg = _MockTgMessage()
    tg_msg.document = _MockDocument("doc_001", "report.pdf", "application/pdf", 1024)
    update = _make_update(tg_msg)
    chain = conv.to_internal(update)
    assert len(chain.files) == 1
    assert chain.files[0].name == "report.pdf"


def test_telegram_converter_to_platform():
    from channels.telegram_channel.handler import TelegramMessageConverter
    conv = TelegramMessageConverter()
    chain = MessageChain.from_text("response text")
    result = conv.to_platform(chain)
    assert isinstance(result, dict)
    assert result["text"] == "response text"


@pytest.mark.skipif(not HAS_TELEGRAM, reason="python-telegram-bot not installed")
def test_telegram_converter_empty():
    from channels.telegram_channel.handler import TelegramMessageConverter
    conv = TelegramMessageConverter()
    tg_msg = _MockTgMessage()
    update = _make_update(tg_msg)
    chain = conv.to_internal(update)
    assert chain.is_empty


# ==================================================================
# ABC guard
# ==================================================================


def test_converter_abc_cannot_instantiate():
    """MessageConverter ABC should prevent direct instantiation."""
    with pytest.raises(TypeError):
        MessageConverter()  # type: ignore[abstract]
