"""
Mid-term memory compression.

When total tokens in a session exceed 85% of the provider's context
window, the oldest messages are compressed into a summary by the LLM
and injected back into the conversation as a <historical_context> tag.

The compressed summary is also persisted to disk so it survives restarts.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Prompt for compressing old conversation history
COMPRESSION_PROMPT = """You are a conversation summarizer. Summarize the following conversation history concisely.

Focus on:
1. Key topics discussed
2. Decisions made
3. Important facts shared by the user
4. Actions taken or agreed upon

Write a single paragraph summary (max 500 characters). Use the same language as the conversation."""


class Compressor:
    """
    Detects when a session exceeds the token threshold and triggers
    LLM-based compression of old conversation turns.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        keep_recent_turns: int = 3,
        backup_dir: str = "data/memory/compressed",
    ) -> None:
        self._threshold = threshold
        self._keep_recent = keep_recent_turns
        self._backup_dir = Path(backup_dir)
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_compress(
        self, total_tokens: int, context_window: int
    ) -> bool:
        """Return True if the session should be compressed."""
        if context_window <= 0:
            return False
        return total_tokens > int(context_window * self._threshold)

    def check_and_compress(
        self,
        session_id: str,
        messages: List[Dict],
        provider,  # BaseProvider for LLM compression call
        context_window: int,
    ) -> Tuple[List[Dict], bool]:
        """
        If the message list exceeds the token threshold, compress the
        oldest turns into a summary.

        Args:
            session_id: Session identifier (used for backup filename).
            messages: Current message list.
            provider: LLM provider for generating the summary.
            context_window: Provider's context window size.

        Returns:
            (compressed_messages, was_compressed)
        """
        # Estimate total tokens from ChatResponse history is more accurate,
        # but as a quick pre-check we use the provider's token counter.
        total = sum(provider.count_tokens(m.get("content") or "") for m in messages)

        if not self.should_compress(total, context_window):
            return messages, False

        # Identify the split point: keep last N turns intact
        # A "turn" = user + assistant (+ optional tool messages)
        turn_boundaries = _find_turn_boundaries(messages)
        if len(turn_boundaries) <= self._keep_recent:
            return messages, False  # not enough turns to compress

        split_idx = turn_boundaries[-(self._keep_recent)]
        old_messages = messages[:split_idx]
        recent_messages = messages[split_idx:]

        if not old_messages:
            return messages, False

        # Generate summary via LLM
        summary = self._summarize(old_messages, provider)
        if not summary:
            return messages, False  # LLM call failed, skip compression

        # Build the compressed message list
        compressed: List[Dict] = []

        # Inject summary as a system-level context message
        compressed.append({
            "role": "system",
            "content": f"<historical_context>\n{summary}\n</historical_context>",
        })

        compressed.extend(recent_messages)

        # Backup to disk
        self._save_backup(session_id, summary, old_messages)

        return compressed, True

    def load_compressed_context(self, session_id: str) -> Optional[str]:
        """
        Return the most recent compressed summary for *session_id*,
        or None if none exists.
        """
        backup_file = self._backup_dir / f"{session_id}.json"
        if not backup_file.exists():
            return None

        try:
            data = json.loads(backup_file.read_text(encoding="utf-8"))
            return data.get("summary")
        except (json.JSONDecodeError, KeyError):
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _summarize(self, messages: List[Dict], provider) -> str:
        """Ask the LLM to compress old messages into a summary."""
        # Build a compact representation of the old conversation
        transcript_lines: List[str] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                label = "User" if role == "user" else "Assistant"
                transcript_lines.append(f"{label}: {content[:500]}")

        if not transcript_lines:
            return ""

        transcript = "\n".join(transcript_lines)

        try:
            response = provider.chat([
                {"role": "system", "content": COMPRESSION_PROMPT},
                {"role": "user", "content": transcript},
            ])
            return (response.content or "").strip()
        except Exception:
            return ""

    def _save_backup(
        self,
        session_id: str,
        summary: str,
        old_messages: List[Dict],
    ) -> None:
        """Persist compressed summary to disk as JSON."""
        backup_file = self._backup_dir / f"{session_id}.json"
        data = {
            "session_id": session_id,
            "summary": summary,
            "compressed_turn_count": len(old_messages),
        }
        backup_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_turn_boundaries(messages: List[Dict]) -> List[int]:
    """
    Return list of indices where new user turns begin.

    A turn starts with a ``role: "user"`` message (excluding system
    messages that come after user/assistant pairs).
    """
    boundaries: List[int] = []
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            boundaries.append(i)
    return boundaries
