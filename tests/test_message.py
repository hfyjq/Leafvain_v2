"""
Unit tests for channels.message — MessageChain, MessageComponent subclasses.

Tests cover:
  - Plain text component and MessageChain.from_text
  - Image / File / Voice / At component construction
  - MessageChain.content (backward-compat plain-text extraction)
  - MessageChain.images, MessageChain.files filters
  - Empty chain behaviour
  - Multi-component chains
  - to_plain_text for each component type
  - to_dict serialisation
  - Fluency: MessageChain.add()
  - __bool__ / __len__ / __repr__
"""

from __future__ import annotations

import pytest

from channels.message import (
    At,
    ComponentType,
    File,
    Image,
    MessageChain,
    MessageComponent,
    Plain,
    Voice,
)


# ==================================================================
# Plain text
# ==================================================================


def test_plain_component():
    """Plain should wrap text and set ComponentType.TEXT."""
    c = Plain("hello world")
    assert c.type == ComponentType.TEXT
    assert c.text == "hello world"
    assert c.to_plain_text() == "hello world"
    assert c.to_dict() == {"type": "text", "text": "hello world"}


def test_message_chain_from_text():
    """from_text factory should produce a single Plain component."""
    chain = MessageChain.from_text("hi")
    assert len(chain.components) == 1
    assert isinstance(chain.components[0], Plain)
    assert chain.content == "hi"


def test_message_chain_content_empty():
    """content should return '' for an empty chain."""
    chain = MessageChain()
    assert chain.content == ""
    assert chain.is_empty


# ==================================================================
# Image
# ==================================================================


def test_image_url():
    img = Image(url="https://example.com/a.png")
    assert img.type == ComponentType.IMAGE
    assert img.url == "https://example.com/a.png"
    assert img.mime_type == "image/png"
    assert "[Image" in img.to_plain_text()


def test_image_local_path():
    img = Image(local_path="/tmp/photo.jpg", mime_type="image/jpeg")
    assert img.local_path == "/tmp/photo.jpg"
    assert "photo.jpg" in img.to_plain_text()


def test_image_to_dict():
    img = Image(url="https://x.com/i.png", mime_type="image/png")
    d = img.to_dict()
    assert d["type"] == "image"
    assert d["url"] == "https://x.com/i.png"


def test_message_chain_from_image():
    chain = MessageChain.from_image(url="https://example.com/img.png")
    assert len(chain.images) == 1
    assert chain.images[0].url == "https://example.com/img.png"


# ==================================================================
# File
# ==================================================================


def test_file_component():
    f = File(name="report.pdf", mime_type="application/pdf", size_bytes=42)
    assert f.type == ComponentType.FILE
    assert f.name == "report.pdf"
    assert f.size_bytes == 42
    assert "report.pdf" in f.to_plain_text()
    assert f.to_dict()["name"] == "report.pdf"


def test_message_chain_from_file():
    chain = MessageChain.from_file(name="doc.txt", local_path="/tmp/doc.txt")
    assert len(chain.files) == 1
    assert chain.files[0].name == "doc.txt"


# ==================================================================
# Voice / At (reserved)
# ==================================================================


def test_voice_reserved():
    v = Voice(url="https://x.com/voice.ogg", duration_seconds=12)
    assert v.type == ComponentType.VOICE
    assert v.to_plain_text() == "[Voice]"


def test_at_reserved():
    a = At(target_id="user_42", display_name="Alice")
    assert a.type == ComponentType.AT
    assert "Alice" in a.to_plain_text()
    assert a.to_dict()["target_id"] == "user_42"


# ==================================================================
# Multi-component chains
# ==================================================================


def test_multi_component_chain():
    """A chain with text + image + file should extract content correctly."""
    chain = MessageChain(components=[
        Plain("Look at this: "),
        Image(url="https://example.com/pic.png"),
        Plain(" and this file: "),
        File(name="data.csv", mime_type="text/csv"),
    ])
    assert len(chain.components) == 4
    assert len(chain.images) == 1
    assert len(chain.files) == 1
    assert not chain.is_empty
    # content should include all to_plain_text outputs
    content = chain.content
    assert "Look at this:" in content
    assert "pic.png" in content
    assert "data.csv" in content


def test_chain_add_fluent():
    chain = MessageChain.from_text("a")
    chain.add(Plain("b")).add(Image(url="x"))
    assert len(chain.components) == 3
    assert chain.content == "ab[Image: x]"


# ==================================================================
# Edge cases
# ==================================================================


def test_empty_chain_bool():
    assert not MessageChain()
    assert MessageChain.from_text("x")


def test_chain_repr():
    chain = MessageChain.from_text("hello")
    r = repr(chain)
    assert "hello" in r
    assert "MessageChain" in r


def test_plain_text_preserves_non_ascii():
    chain = MessageChain.from_text("你好世界 🌍")
    assert "你好世界" in chain.content
    assert "🌍" in chain.content
