"""
WeChat iLink authentication — QR code login + credential cache.

Login flow:
  1. GET  /ilink/bot/get_bot_qrcode?bot_type=3  → {qrcode, qrcode_img_content}
  2. Display QR code in terminal
  3. GET  /ilink/bot/get_qrcode_status?qrcode=xxx  (long poll 40s)
     status: wait → scaned → confirmed → {bot_token, ilink_bot_id, ilink_user_id}
  4. Persist credentials to disk (0o600) for reuse on restart

Credentials format::

    {
      "bot_token": "...",
      "ilink_bot_id": "...",
      "ilink_user_id": "xxx@im.wechat",
      "base_url": "https://ilinkai.weixin.qq.com"
    }
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

BASE_URL = "https://ilinkai.weixin.qq.com"
QR_POLL_TIMEOUT = 40  # seconds
MAX_QR_RETRIES = 3


# ------------------------------------------------------------------
# Credential persistence
# ------------------------------------------------------------------

def load_credentials(path: str | Path) -> dict[str, Any] | None:
    """Load cached credentials from disk.  Returns None if absent or corrupt."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "bot_token" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def save_credentials(path: str | Path, creds: dict[str, Any]) -> None:
    """Persist credentials to disk with restricted permissions."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(creds, ensure_ascii=False, indent=2), encoding="utf-8")
    # Restrict to owner-only on POSIX; on Windows this is a no-op
    try:
        os.chmod(str(p), 0o600)
    except OSError:
        pass


# ------------------------------------------------------------------
# QR code login
# ------------------------------------------------------------------

def fetch_qrcode(base_url: str = BASE_URL) -> dict[str, str]:
    """Get a fresh QR code from the server.

    Returns ``{"qrcode": "<key>", "qrcode_img_content": "<url>"}``.
    """
    resp = httpx.get(f"{base_url}/ilink/bot/get_bot_qrcode?bot_type=3")
    resp.raise_for_status()
    data = resp.json()
    return {"qrcode": data["qrcode"], "qrcode_img_content": data.get("qrcode_img_content", "")}


def poll_qrcode_status(
    qrcode_key: str,
    base_url: str = BASE_URL,
    timeout: int = QR_POLL_TIMEOUT,
) -> dict[str, Any]:
    """Long-poll the QR code status until the user acts or it expires.

    Returns the full status dict.  Caller should inspect ``status["status"]``:
    ``"wait"`` | ``"scaned"`` | ``"confirmed"`` | ``"expired"``.
    """
    resp = httpx.get(
        f"{base_url}/ilink/bot/get_qrcode_status?qrcode={qrcode_key}",
        headers={"iLink-App-ClientVersion": "1"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def display_qrcode_terminal(qrcode_url: str) -> None:
    """Render a QR code URL as ASCII art in the terminal."""
    try:
        import qrcode as qr_module
        qr = qr_module.QRCode(border=1)
        qr.add_data(qrcode_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ImportError:
        print(f"[wechat] QR code URL: {qrcode_url}")
        print("[wechat] (install 'qrcode' package for terminal QR display)")


def do_login(
    base_url: str = BASE_URL,
    credential_path: str | Path = "data/wechat_credentials.json",
) -> dict[str, Any]:
    """Full QR code login flow.  Blocks until user confirms on phone.

    Returns the credentials dict ready for persistence and use.
    May retry up to *max_retries* times if the QR code expires.
    """
    for attempt in range(1, MAX_QR_RETRIES + 1):
        print(f"[wechat] Fetching QR code (attempt {attempt}/{MAX_QR_RETRIES})...")
        qr = fetch_qrcode(base_url)
        qrcode_key = qr["qrcode"]

        display_qrcode_terminal(qr["qrcode_img_content"])
        print("[wechat] Scan the QR code with WeChat to log in.")

        while True:
            try:
                status = poll_qrcode_status(qrcode_key, base_url)
            except httpx.HTTPError as e:
                print(f"[wechat] QR poll error: {e}")
                break

            state = status.get("status", "")
            if state == "wait":
                continue  # keep polling
            elif state == "scaned":
                print("[wechat] Scanned! Please confirm on your phone...")
            elif state == "confirmed":
                token = status.get("bot_token", "")
                if not token:
                    raise RuntimeError("Login confirmed but no bot_token returned")
                creds = {
                    "bot_token": token,
                    "ilink_bot_id": status.get("ilink_bot_id", ""),
                    "ilink_user_id": status.get("ilink_user_id", ""),
                    "base_url": status.get("baseurl") or base_url,
                }
                save_credentials(credential_path, creds)
                print(f"[wechat] ✅ Login successful! "
                      f"bot_id={creds['ilink_bot_id'][:12]}...")
                return creds
            elif state == "expired":
                print("[wechat] QR code expired, regenerating...")
                break
            else:
                print(f"[wechat] Unknown QR status: {state}")
                time.sleep(1)

    raise RuntimeError(
        f"Login failed after {MAX_QR_RETRIES} QR code attempts. "
        "Please check your network and try again."
    )
