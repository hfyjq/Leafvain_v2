"""
Telegram Channel — multi-user messaging via Telegram Bot API.

Uses a manual ``getUpdates`` polling loop instead of PTB's
``Updater`` / ``run_polling()``.  This avoids all the threading and
event-loop conflicts that prevent message dispatch when running
inside an existing asyncio context.

v0.7.0: ``TelegramMessageConverter`` translates between Telegram
messages and ``MessageChain``.  Document/Photo handlers added —
fixes the "file sent to bot → no response" bug.

Architecture::

    while running:
        updates = await bot.get_updates(offset=offset, timeout=30)
        for update in updates:
            await app.process_update(update)   # dispatches to handlers
            offset = update.update_id + 1
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from channels.base import (
    Channel,
    ChannelMetadata,
    Message,
    MessageEvent,
    MessageSession,
    MessageType,
)
from channels.capabilities import FileReceiveCapable
from channels.converter import MessageConverter
from channels.message import File, Image, MessageChain, Plain

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
_TEMP_DIR = Path("data/workspaces/telegram_files")


# ------------------------------------------------------------------
# TelegramMessageConverter
# ------------------------------------------------------------------

class TelegramMessageConverter(MessageConverter):
    """Translate between Telegram ``Update`` and ``MessageChain``.

    ``to_internal`` extracts text, caption, photo, and document from
    a ``telegram.Update`` (the raw message from PTB).

    ``to_platform`` converts a ``MessageChain`` back to a dict
    representation that the channel's send methods understand.
    """

    # -- platform → internal -----------------------------------------

    def to_internal(
        self,
        platform_message: Any,
        photo_path: str | None = None,
        doc_path: str | None = None,
    ) -> MessageChain:
        """Convert a ``telegram.Update`` into a MessageChain.

        *photo_path* and *doc_path* are pre-downloaded local file paths
        (set by ``_file_handler`` before calling this).
        """
        try:
            from telegram import Message as TgMessage  # noqa: F811
        except ImportError:
            return MessageChain()

        tg_msg: TgMessage | None = getattr(platform_message, "message", None)
        if tg_msg is None:
            return MessageChain()

        components: list = []

        # Photos (check BEFORE text — photos often have a caption too)
        if tg_msg.photo and len(tg_msg.photo) > 0:
            components.append(
                Image(local_path=photo_path, mime_type="image/jpeg")
            )

        # Documents
        if tg_msg.document:
            doc = tg_msg.document
            components.append(
                File(
                    name=doc.file_name or "document",
                    local_path=doc_path,
                    mime_type=doc.mime_type or "application/octet-stream",
                    size_bytes=doc.file_size or 0,
                )
            )

        # Text / caption (always add if present)
        text = tg_msg.text or tg_msg.caption or ""
        if text:
            components.insert(0, Plain(text))

        return MessageChain(components) if components else MessageChain()

    # -- internal → platform -----------------------------------------

    def to_platform(self, message_chain: MessageChain, **kwargs: Any) -> dict[str, Any]:
        """Convert a MessageChain to a dict for Telegram send methods.

        Returns a dict with keys like ``text``, ``photo_path``,
        ``doc_path``, ``doc_name`` — consumed by ``_send_chain()``.
        """
        result: dict[str, Any] = {"text": message_chain.content}

        for img in message_chain.images:
            result["photo_path"] = img.local_path
            if img.mime_type:
                result["photo_mime"] = img.mime_type
            break  # Only one photo per message (Telegram limitation)

        for f in message_chain.files:
            result["doc_path"] = f.local_path
            result["doc_name"] = f.name
            if f.mime_type:
                result["doc_mime"] = f.mime_type
            break  # Only one doc per message

        return result


# ------------------------------------------------------------------
# TelegramChannel
# ------------------------------------------------------------------

class TelegramChannel(Channel):
    """Telegram Bot adapter — manual getUpdates loop, no PTB Updater."""

    def __init__(
        self,
        config: dict | None = None,
        scheduler=None,
        session_pool=None,
    ) -> None:
        super().__init__(config)
        self._converter = TelegramMessageConverter()
        self._scheduler = scheduler
        self._pool = session_pool
        self._running = False
        self._app = None
        self._bot = None

        tg_cfg = (config or {}).get("channel", {}).get("telegram", {})
        self._bot_token = tg_cfg.get(
            "bot_token",
            os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        )
        self._proxy_url = tg_cfg.get("proxy", "") or os.environ.get("HTTPS_PROXY", "")
        self._polling_timeout = tg_cfg.get("polling_timeout", 30)

        _TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Channel ABC ----

    def meta(self) -> ChannelMetadata:
        return ChannelMetadata(
            name="telegram_channel",
            description="Telegram Bot adapter — multi-user messaging",
            platform_type="telegram",
            support_multimodal=True,
        )

    async def run(self) -> None:
        """Manual getUpdates loop — all inside our event loop."""
        if not self._bot_token:
            print("[telegram] ERROR: No bot token configured.")
            return

        # Proxy
        if self._proxy_url:
            os.environ["HTTPS_PROXY"] = self._proxy_url
            os.environ["HTTP_PROXY"] = self._proxy_url
            print(f"[telegram] Using proxy: {self._proxy_url}")

        # Build Application
        from telegram.ext import ApplicationBuilder
        app = ApplicationBuilder().token(self._bot_token).build()

        try:
            await app.initialize()
            await app.start()
        except Exception as e:
            print(f"[telegram] ERROR: Cannot start — {type(e).__name__}: {e}")
            print("[telegram] Check your token, proxy, and network.")
            return

        self._app = app
        self._bot = app.bot

        # Verify connection
        try:
            me = await app.bot.get_me()
            print(f"[telegram] Connected as @{me.username} (id={me.id})")
        except Exception as e:
            print(f"[telegram] WARNING: get_me failed — {type(e).__name__}: {e}")

        # ---- Register handlers on the application ----
        from telegram.ext import CommandHandler, MessageHandler, filters

        channel = self  # closure capture

        async def _process(update: Any,
                           photo_path: str | None = None,
                           doc_path: str | None = None) -> str | None:
            """Core processing: convert update → pipeline → response.

            Returns the response text (or None on error / empty).
            """
            if not update.message or not update.effective_user:
                return None

            user = update.effective_user

            # Convert platform message → MessageChain
            chain = channel._converter.to_internal(
                update, photo_path=photo_path, doc_path=doc_path,
            )

            # Skip completely empty chains (silent messages)
            if chain.is_empty:
                return None

            log_text = chain.content[:80] if chain.content else "[media]"
            print(f"[telegram] 📩 {user.full_name or user.username}: {log_text}")

            session = channel._make_session(update)
            message = Message(
                sender_id=str(user.id),
                sender_name=user.full_name or user.username or "",
                message_chain=chain,
                message_type=session.message_type,
                session=session,
            )
            event = channel.commit_event(message)

            # Typing indicator
            try:
                await update.message.chat.send_action("typing")
            except Exception:
                pass

            # Pipeline
            if channel._scheduler:
                try:
                    response = await channel._scheduler.execute(event)
                except Exception as e:
                    print(f"[telegram] Pipeline error: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    response = f"Error: {type(e).__name__}: {e}"
                return response
            return None

        async def _handler(update: Any, context: Any) -> None:
            """Text handler."""
            response = await _process(update)
            if response:
                await channel._send_long(update.message, response)

        async def _file_handler(update: Any, context: Any) -> None:
            """Document / Photo handler — download, then process."""
            if not update.message or not update.effective_user:
                return

            photo_path: str | None = None
            doc_path: str | None = None

            # Download photo
            if update.message.photo:
                try:
                    file_id = update.message.photo[-1].file_id
                    tg_file = await channel._bot.get_file(file_id)
                    temp_path = _TEMP_DIR / f"photo_{file_id}.jpg"
                    await tg_file.download_to_drive(temp_path)
                    photo_path = str(temp_path)
                    print(f"[telegram] Photo downloaded: {photo_path}")
                except Exception as e:
                    print(f"[telegram] Photo download error: {e}")

            # Download document
            if update.message.document:
                try:
                    doc = update.message.document
                    tg_file = await channel._bot.get_file(doc.file_id)
                    ext = Path(doc.file_name or "").suffix or ".bin"
                    temp_path = _TEMP_DIR / f"doc_{doc.file_id}{ext}"
                    await tg_file.download_to_drive(temp_path)
                    doc_path = str(temp_path)
                    print(f"[telegram] Document downloaded: {doc_path} "
                          f"({doc.file_name}, {doc.file_size or 0} bytes)")
                except Exception as e:
                    print(f"[telegram] Document download error: {e}")

            response = await _process(
                update, photo_path=photo_path, doc_path=doc_path,
            )
            if response:
                await channel._send_long(update.message, response)

        # Handlers: text, documents, photos, commands
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, _handler,
        ))
        app.add_handler(MessageHandler(
            filters.PHOTO, _file_handler,
        ))
        app.add_handler(MessageHandler(
            filters.Document.ALL, _file_handler,
        ))
        app.add_handler(CommandHandler("start", _handler))
        app.add_handler(CommandHandler("clear", _handler))
        app.add_handler(CommandHandler("compact", _handler))

        # ---- Main polling loop ----
        print("[telegram] Bot is running (manual getUpdates loop).")
        self._running = True
        offset = 0

        while self._running:
            retries = 0
            try:
                updates = await app.bot.get_updates(
                    offset=offset,
                    timeout=0,
                    allowed_updates=["message"],
                )
                retries = 0  # reset on success
                await asyncio.sleep(0.8)  # rate-limit
            except Exception as e:
                err_name = type(e).__name__
                if "TimedOut" in err_name or "Timeout" in str(e):
                    await asyncio.sleep(1)
                    continue
                retries += 1
                wait = min(2 ** retries, 30)
                print(f"[telegram] {err_name}, retry in {wait}s (attempt {retries})")
                await asyncio.sleep(wait)
                continue

            for update in updates:
                offset = update.update_id + 1
                try:
                    await app.process_update(update)
                except Exception as e:
                    print(f"[telegram] process_update error: "
                          f"{type(e).__name__}: {e}")

    async def terminate(self) -> None:
        self._running = False
        if self._app:
            try:
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                pass
        print("[telegram] Bot stopped.")

    async def send(self, session: MessageSession, content: str | MessageChain) -> None:
        if self._bot:
            chat_id = int(session.session_id)
            text = content if isinstance(content, str) else content.content
            await self._send_long_via_bot(chat_id, text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_session(self, update: Any) -> MessageSession:
        from telegram.constants import ChatType
        chat = update.effective_chat
        if chat.type == ChatType.PRIVATE:
            return MessageSession("telegram", MessageType.PRIVATE, str(chat.id))
        return MessageSession("telegram", MessageType.GROUP, str(chat.id))

    async def _send_long(self, message: Any, content: str) -> None:
        """Send, splitting long messages."""
        if len(content) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            await message.reply_text(content)
            return
        chunks = _split_text(content, TELEGRAM_MAX_MESSAGE_LENGTH)
        for i, chunk in enumerate(chunks):
            prefix = f"({i + 1}/{len(chunks)})\n" if len(chunks) > 1 else ""
            await message.reply_text(prefix + chunk)

    async def _send_long_via_bot(self, chat_id: int, content: str) -> None:
        """Send via bot (for proactive messages)."""
        if len(content) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            await self._bot.send_message(chat_id=chat_id, text=content)
            return
        chunks = _split_text(content, TELEGRAM_MAX_MESSAGE_LENGTH)
        for i, chunk in enumerate(chunks):
            prefix = f"({i + 1}/{len(chunks)})\n" if len(chunks) > 1 else ""
            await self._bot.send_message(chat_id=chat_id, text=prefix + chunk)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _split_text(text: str, max_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        split_at = remaining.rfind("\n", 0, max_len)
        if split_at == -1 or split_at < max_len // 2:
            split_at = max_len
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks
