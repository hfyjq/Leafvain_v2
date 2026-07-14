"""
WeChat Channel — iLink protocol adapter via Tencent's official API.

Architecture::

    while running:
        msgs = await _get_updates()   # long-poll POST /ilink/bot/getupdates
        for msg in msgs:
            event = commit_event(message)   # → MessageEvent
            response = await scheduler.execute(event)   # → Pipeline → LLM
            await _send_message(to, response, context_token)   # → POST sendMessage

Key differences from Telegram Channel:
  - Login via QR code scan (auth.py), not static bot token
  - Pure HTTP/JSON — no SDK dependency (just httpx)
  - context_token must be extracted from received messages and echoed back
  - from_user_id must be "" (empty string) — omitting it causes silent failure
"""

from __future__ import annotations

import asyncio
import base64
import os
import secrets
import uuid
from pathlib import Path
from typing import Any

import httpx

from channels.base import (
    Channel,
    ChannelMetadata,
    Message,
    MessageEvent,
    MessageSession,
    MessageType,
)
from channels.capabilities import EditStateCapable
from channels.message import MessageChain, Plain
from channels.wechat_channel.auth import do_login, load_credentials
from channels.wechat_channel.converter import WeChatMessageConverter


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_WECHAT_TEMP_DIR = Path("data/workspaces/wechat_files")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _random_wechat_uin() -> str:
    """Generate a random X-WECHAT-UIN header value (base64-encoded uint32)."""
    raw = secrets.randbits(32).to_bytes(4, "big")
    return base64.b64encode(raw).decode("ascii")


def _split_long_text(text: str, max_len: int = 2048) -> list[str]:
    """Split long text at paragraph boundaries for WeChat delivery."""
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


# ------------------------------------------------------------------
# WeChatChannel
# ------------------------------------------------------------------

class WeChatChannel(Channel):
    """WeChat bot adapter via Tencent iLink protocol."""

    def __init__(
        self,
        config: dict | None = None,
        scheduler=None,
        session_pool=None,
    ) -> None:
        super().__init__(config)
        self._converter = WeChatMessageConverter()
        self._scheduler = scheduler
        self._pool = session_pool
        self._running = False

        wx_cfg = (config or {}).get("channel", {}).get("wechat", {})
        self._base_url = wx_cfg.get("base_url", "https://ilinkai.weixin.qq.com")
        self._credential_path = wx_cfg.get(
            "credentials_path", "data/wechat_credentials.json",
        )
        self._polling_timeout = wx_cfg.get("polling_timeout", 35)
        self._aggregation_delay: float = float(
            wx_cfg.get("aggregation_delay", 1.5)
        )

        # State set after login
        self._bot_token: str = ""
        self._ilink_bot_id: str = ""
        self._ilink_user_id: str = ""
        self._get_updates_buf: str = ""  # cursor for incremental sync

        self._http: httpx.AsyncClient | None = None

        # Message aggregation: buffer rapid-fire messages per user,
        # merge them into one MessageChain before pipeline processing.
        # This prevents file+text arriving as two separate turns.
        self._agg_buffers: dict[str, dict[str, Any]] = {}
        self._agg_lock = asyncio.Lock()

    # ---- Channel ABC ----

    def meta(self) -> ChannelMetadata:
        return ChannelMetadata(
            name="wechat_channel",
            description="WeChat Bot via Tencent iLink protocol",
            platform_type="wechat",
            support_multimodal=True,
        )

    async def run(self) -> None:
        """Main loop: login → poll → process → reply."""
        # 1. Ensure logged in
        try:
            await self._ensure_login()
        except Exception as e:
            print(f"[wechat] Login failed: {e}")
            return

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(self._polling_timeout + 10))
        self._running = True

        print(f"[wechat] Bot is running (iLink long-poll loop). "
              f"bot_id={self._ilink_bot_id[:12]}...")

        # 2. Main polling loop
        while self._running:
            try:
                updates = await self._get_updates()
            except httpx.TimeoutException:
                # Long-poll timeout is normal — restart poll immediately
                continue
            except httpx.HTTPError as e:
                print(f"[wechat] Poll error: {type(e).__name__}: {e}")
                await asyncio.sleep(2)
                continue

            msgs = updates.get("msgs", []) or []
            for raw_msg in msgs:
                try:
                    await self._buffer_message(raw_msg)
                except Exception as e:
                    print(f"[wechat] Buffer error: {type(e).__name__}: {e}")

        # Cleanup
        if self._http:
            await self._http.aclose()
            self._http = None

    async def terminate(self) -> None:
        self._running = False
        if self._http:
            await self._http.aclose()
            self._http = None
        print("[wechat] Bot stopped.")

    async def send(self, session: MessageSession, content: str | MessageChain) -> None:
        """Send a message to a WeChat user (used for proactive messages)."""
        text = content if isinstance(content, str) else content.content
        to_user = session.session_id  # ilink_user_id
        ctx_token = ""  # proactive messages have no context_token
        await self._send_text(to_user, text, ctx_token)

    # ---- Login ------------------------------------------------------------

    async def _ensure_login(self) -> None:
        """Load cached credentials or run QR code login flow."""
        creds = load_credentials(self._credential_path)
        if creds:
            print(f"[wechat] Loaded cached credentials "
                  f"(bot_id={creds.get('ilink_bot_id', '')[:12]}...)")
        else:
            print("[wechat] No cached credentials — starting QR login...")
            creds = do_login(
                base_url=self._base_url,
                credential_path=self._credential_path,
            )

        self._bot_token = creds["bot_token"]
        self._ilink_bot_id = creds.get("ilink_bot_id", "")
        self._ilink_user_id = creds.get("ilink_user_id", "")
        if creds.get("base_url"):
            self._base_url = creds["base_url"]

    # ---- Message aggregation + handling -----------------------------------

    async def _buffer_message(self, raw_msg: dict[str, Any]) -> None:
        """Buffer a message for aggregation, or flush immediately if disabled.

        When a user sends rapid-fire messages (e.g. file + instruction),
        WeChat delivers them as separate messages.  Aggregation merges
        them into a single MessageChain so the LLM sees the file AND
        the instruction in one turn.
        """
        msg = raw_msg.get("msg", raw_msg)
        from_user = msg.get("from_user_id", "")
        context_token = msg.get("context_token", "")

        if not from_user:
            return

        chain = self._converter.to_internal({"msg": msg})
        if chain.is_empty:
            return

        delay = self._aggregation_delay
        if delay <= 0:
            # Aggregation disabled — process immediately
            await self._process_aggregated(from_user, chain, context_token)
            return

        async with self._agg_lock:
            if from_user in self._agg_buffers:
                buf = self._agg_buffers[from_user]
                # Cancel existing timer
                if buf["timer"] is not None:
                    buf["timer"].cancel()
                # Merge: text + file → clearly separated so LLM
                # doesn't confuse the instruction with file content.
                has_file = any(
                    c.type.value in ("image", "file") for c in chain.components
                )
                if has_file:
                    buf["chain"].components.append(Plain("\n📎 附件: "))
                else:
                    buf["chain"].components.append(Plain("\n"))
                buf["chain"].components.extend(chain.components)
                buf["context_token"] = context_token  # latest wins
            else:
                buf = {
                    "chain": chain,
                    "context_token": context_token,
                    "timer": None,
                }
                self._agg_buffers[from_user] = buf

            # Start new timer
            buf["timer"] = asyncio.create_task(
                self._delayed_flush(from_user, delay)
            )

    async def _delayed_flush(self, user_id: str, delay: float) -> None:
        """Wait *delay* seconds, then flush the buffer for *user_id*."""
        try:
            await asyncio.sleep(delay)
            async with self._agg_lock:
                buf = self._agg_buffers.pop(user_id, None)
            if buf is not None:
                await self._process_aggregated(
                    user_id, buf["chain"], buf["context_token"],
                )
        except asyncio.CancelledError:
            pass  # timer reset by a new message
        except Exception as e:
            print(f"[wechat] Flush error for {user_id}: {type(e).__name__}: {e}")

    async def _process_aggregated(
        self,
        from_user: str,
        chain: MessageChain,
        context_token: str,
    ) -> None:
        """Process a (possibly aggregated) MessageChain through the pipeline."""
        # Download + decrypt files/images before pipeline processing
        await self._download_files(chain)

        # Append local paths to the message text so the LLM knows
        # files have been materialised (per Molio Prompt 适配).
        for c in chain.components:
            from channels.message import File as F, Image as Img
            if isinstance(c, (F, Img)) and c.local_path:
                chain.components.append(
                    Plain(f"\n[{('Image' if isinstance(c, Img) else 'File')}] "
                          f"已下载到本地：{c.local_path}")
                )

        log_text = chain.content[:60] if chain.content else "[media]"
        print(f"[wechat] 📩 {from_user}: {log_text}")

        session = self._make_session(from_user)
        message = Message(
            sender_id=from_user,
            sender_name=from_user,
            message_chain=chain,
            message_type=session.message_type,
            session=session,
            raw_data={"context_token": context_token},
        )
        event = self.commit_event(message)

        # Typing indicator
        try:
            await self._send_typing(from_user, context_token)
        except Exception:
            pass

        # Pipeline
        response = ""
        if self._scheduler:
            try:
                response = await self._scheduler.execute(event)
            except Exception as e:
                print(f"[wechat] Pipeline error: {type(e).__name__}: {e}")
                response = f"Error: {type(e).__name__}: {e}"

        if response:
            await self._send_text(from_user, response, context_token)

    async def _download_files(self, chain: MessageChain) -> None:
        """Download + AES-128-ECB decrypt files/images from iLink CDN."""
        _WECHAT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        for c in chain.components:
            from channels.message import File as F, Image as Img
            if not isinstance(c, (F, Img)):
                continue

            url = getattr(c, "url", None)
            aes_key = getattr(c, "_aes_key", None)
            if not url:
                continue

            file_name = getattr(c, "name", None) or Path(str(url)).name or "download"
            safe_name = "".join(
                ch if ch.isalnum() or ch in "._-" else "_" for ch in file_name
            )
            local_path = _WECHAT_TEMP_DIR / safe_name
            if local_path.exists():
                c.local_path = str(local_path)
                continue

            print(f"[wechat] Downloading: {safe_name} "
                  f"(aes={'yes' if aes_key else 'no'})...")
            try:
                # 1. Download encrypted bytes from CDN
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=60)
                    resp.raise_for_status()
                    cipher_bytes = resp.content

                if not cipher_bytes:
                    print(f"[wechat] Download: empty response from CDN")
                    continue

                # 2. AES-128-ECB decrypt if key is present
                if aes_key:
                    try:
                        from cryptography.hazmat.primitives.ciphers import (
                            Cipher, algorithms, modes,
                        )
                    except ImportError:
                        print("[wechat] cryptography not installed — "
                              "cannot decrypt files.  pip install cryptography")
                        plain_bytes = cipher_bytes  # store encrypted as-is
                    else:
                        key_bytes = _derive_aes_key(aes_key)
                        if key_bytes and len(key_bytes) == 16:
                            decipher = Cipher(
                                algorithms.AES(key_bytes),
                                modes.ECB(),
                            ).decryptor()
                            plain_bytes = decipher.update(cipher_bytes) + decipher.finalize()
                            # Strip PKCS7 padding
                            pad = plain_bytes[-1]
                            if 1 <= pad <= 16:
                                if all(plain_bytes[-i] == pad for i in range(1, pad + 1)):
                                    plain_bytes = plain_bytes[:-pad]
                        else:
                            print(f"[wechat] Download: cannot derive AES key, "
                                  f"using raw bytes")
                            plain_bytes = cipher_bytes
                else:
                    # No encryption — store raw bytes
                    plain_bytes = cipher_bytes

                local_path.write_bytes(plain_bytes)
                c.local_path = str(local_path)
                if isinstance(c, F):
                    c.size_bytes = local_path.stat().st_size
                print(f"[wechat] Downloaded: {safe_name} "
                      f"({local_path.stat().st_size} bytes)")
            except Exception as e:
                print(f"[wechat] Download failed: {type(e).__name__}: {e}")


    # ---- MessageSession ----------------------------------------------------

    def _make_session(self, user_id: str) -> MessageSession:
        """Create a session key for a WeChat user.  Always private chat."""
        return MessageSession("wechat", MessageType.PRIVATE, user_id)

    # ---- iLink API calls --------------------------------------------------

    async def _api_post(
        self,
        path: str,
        body: dict[str, Any],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """POST to iLink API with correct headers."""
        assert self._http is not None
        headers = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "Authorization": f"Bearer {self._bot_token}",
            "X-WECHAT-UIN": _random_wechat_uin(),
        }
        resp = await self._http.post(
            f"{self._base_url}/{path}",
            json=body,
            headers=headers,
            timeout=timeout or self._polling_timeout + 10,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def _get_updates(self) -> dict[str, Any]:
        """Long-poll for new messages."""
        body: dict[str, Any] = {
            "get_updates_buf": self._get_updates_buf,
            "base_info": {"channel_version": "1.0.3"},
        }
        resp = await self._api_post(
            "ilink/bot/getupdates",
            body,
            timeout=self._polling_timeout,
        )
        # Update cursor for incremental sync
        new_buf = resp.get("get_updates_buf", "")
        if new_buf:
            self._get_updates_buf = new_buf
        return resp

    async def _send_text(
        self, to_user: str, text: str, context_token: str,
    ) -> None:
        """Send a text message (possibly split into multiple chunks)."""
        chunks = _split_long_text(text)
        for i, chunk in enumerate(chunks):
            prefix = f"({i + 1}/{len(chunks)})\n" if len(chunks) > 1 else ""
            body = {
                "msg": {
                    "from_user_id": "",           # MUST be empty string
                    "to_user_id": to_user,
                    "client_id": uuid.uuid4().hex[:16],
                    "message_type": 2,            # BOT message
                    "message_state": 2,           # FINISH
                    "item_list": [
                        {"type": 1, "text_item": {"text": prefix + chunk}},
                    ],
                    "context_token": context_token,
                },
                "base_info": {"channel_version": "1.0.3"},
            }
            await self._api_post("ilink/bot/sendmessage", body)

    async def _send_typing(self, to_user: str, context_token: str) -> None:
        """Send 'typing' indicator."""
        body = {
            "to_user_id": to_user,
            "context_token": context_token,
        }
        try:
            await self._api_post("ilink/bot/sendtyping", body, timeout=5)
        except Exception:
            pass  # typing indicator is best-effort


# ------------------------------------------------------------------
# AES key derivation (module-level — used by _download_files)
# ------------------------------------------------------------------

def _derive_aes_key(aes_key: str) -> bytes | None:
    """Derive a 16-byte AES-128 key from *aes_key* (3 formats).

    Per Molio/CowAgent reverse-engineering:
      1. 32-char hex string → decode as hex → 16 bytes
      2. Base64 of a 32-char hex string → decode → ascii hex → bytes
      3. Base64 of raw 16 bytes → decode directly
    """
    import base64 as _b64
    import re

    key = aes_key.strip()
    if not key:
        return None

    # Format 1: 32-char hex string
    if re.fullmatch(r"[0-9a-fA-F]{32}", key):
        return bytes.fromhex(key)

    # Format 2/3: base64-encoded
    try:
        decoded = _b64.b64decode(key)
    except Exception:
        return None

    if len(decoded) == 32:
        as_hex = decoded.decode("ascii", errors="replace")
        if re.fullmatch(r"[0-9a-fA-F]{32}", as_hex):
            return bytes.fromhex(as_hex)

    if len(decoded) == 16:
        return decoded

    return None
