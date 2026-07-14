"""
WeChat iLink MessageConverter — iLink JSON ↔ MessageChain.

iLink item types (from OpenClaw/Molio reverse-engineering):
  - type 1: text    → Plain
  - type 2: image   → Image   (aeskey / media.aes_key)
  - type 3: voice   → Voice
  - type 4: file    → File    (media.full_url + media.aes_key)
  - type 5: video   → (unsupported)

Media is stored encrypted on CDN.  The ``media`` dict contains
``full_url`` (download URL) and ``aes_key`` (AES-128-ECB key).
"""

from __future__ import annotations

from typing import Any

from channels.converter import MessageConverter
from channels.message import File, Image, MessageChain, Plain, Voice


class WeChatMessageConverter(MessageConverter):
    """Translate between iLink JSON messages and MessageChain."""

    # iLink item type constants (corrected per reverse-engineering)
    ITEM_TEXT = 1
    ITEM_IMAGE = 2
    ITEM_VOICE = 3
    ITEM_FILE = 4
    ITEM_VIDEO = 5

    # -- platform → internal -----------------------------------------

    def to_internal(self, platform_message: dict[str, Any]) -> MessageChain:
        """Convert an iLink message dict into a MessageChain.

        *platform_message* is one element of ``resp["msgs"]`` from getUpdates.
        """
        # Raw message is a flat dict (not nested under "msg")
        msg = platform_message.get("msg", platform_message)
        item_list: list[dict] = msg.get("item_list", [])

        components: list = []
        for item in item_list:
            item_type = item.get("type", 0)

            if item_type == self.ITEM_TEXT:
                text = item.get("text_item", {}).get("text", "")
                if text:
                    components.append(Plain(text))

            elif item_type == self.ITEM_IMAGE:
                img = item.get("image_item", {})
                media = img.get("media", {})
                if isinstance(media, str):
                    import ast
                    try:
                        media = ast.literal_eval(media)
                    except (ValueError, SyntaxError):
                        media = {}
                aes_key = img.get("aeskey") or media.get("aes_key", "")
                components.append(
                    Image(
                        url=media.get("full_url", ""),
                        mime_type="image/jpeg",
                    )
                )
                # Store aes_key on the component for download
                if aes_key:
                    components[-1]._aes_key = aes_key

            elif item_type == self.ITEM_FILE:
                f = item.get("file_item", {})
                media = f.get("media", {})
                if isinstance(media, str):
                    import ast
                    try:
                        media = ast.literal_eval(media)
                    except (ValueError, SyntaxError):
                        media = {}
                size_str = f.get("len", "0")
                try:
                    size_bytes = int(size_str)
                except (ValueError, TypeError):
                    size_bytes = 0
                comp = File(
                    name=f.get("file_name", "file"),
                    url=media.get("full_url", ""),
                    mime_type="application/octet-stream",
                    size_bytes=size_bytes,
                )
                aes_key = media.get("aes_key", "")
                if aes_key:
                    comp._aes_key = aes_key
                components.append(comp)

            elif item_type == self.ITEM_VOICE:
                v = item.get("voice_item", {})
                components.append(
                    Voice(url=v.get("cdn_url", ""),
                          duration_seconds=v.get("duration", 0))
                )

            # type 5 (video) — silently ignored for now

        return MessageChain(components) if components else MessageChain()

    # -- internal → platform -----------------------------------------

    def to_platform(self, message_chain: MessageChain, **kwargs: Any) -> dict[str, Any]:
        """Convert a MessageChain into an iLink sendMessage item_list."""
        item_list: list[dict[str, Any]] = []

        for c in message_chain.components:
            from channels.message import At, File as F, Image as I

            if isinstance(c, (Plain, At)):
                item_list.append({
                    "type": self.ITEM_TEXT,
                    "text_item": {"text": c.to_plain_text()},
                })
            elif isinstance(c, I):
                item_list.append({
                    "type": self.ITEM_IMAGE,
                    "image_item": {
                        "cdn_url": c.url or "",
                        "mime_type": c.mime_type,
                    },
                })
            elif isinstance(c, F):
                item_list.append({
                    "type": self.ITEM_FILE,
                    "file_item": {
                        "file_name": c.name,
                        "cdn_url": c.url or c.local_path or "",
                        "mime_type": c.mime_type,
                        "size": c.size_bytes,
                    },
                })

        return {"item_list": item_list}
