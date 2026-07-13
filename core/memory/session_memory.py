"""
L3 Session Memory: structured Markdown session notes.

Maintains a file at ``data/memory/sessions/<session_id>.md`` that is
incrementally updated after every conversation turn via an LLM call.
This pre-built note replaces the LLM-generated compression summary,
making compression a zero-API-cost operation.

Template::

    # Session Note — <session_id>
    ## Current Status
    ## Key Decisions
    ## Open Questions
    ## Next Steps
"""

import os
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------
# Prompt for updating the session note
# ------------------------------------------------------------------

SESSION_NOTE_PROMPT = """You maintain a structured session note for an ongoing AI conversation. You will receive the CURRENT session note and the LATEST user-assistant exchange. Update the note to reflect the new exchange.

## Sections (always keep all four)
- **Current Status**: What is the user doing right now? What task or topic are they working on?
- **Key Decisions**: Concrete decisions made by the user or agreed upon during the conversation.
- **Open Questions**: Questions the user has asked that remain unanswered, or items they said they'd return to.
- **Next Steps**: What should happen next? What did the user say they will do or want the assistant to do?

## Rules
1. Keep each section concise — bullet points preferred, max 5 items per section.
2. Update Current Status to reflect the MOST RECENT activity.
3. Add new decisions, questions, and next steps without losing old ones (unless resolved).
4. If a question was answered, move it out of Open Questions and note the resolution in Key Decisions or Current Status.
5. Do NOT summarize the entire conversation verbatim — capture the essential state.
6. Write in a continuous, present-tense style.  Do NOT use third-person
   recollection phrases like "the user previously mentioned" or "the assistant
   suggested" — this note tracks an ongoing conversation, not a historical record.
7. Output ONLY the full updated Markdown note, nothing else.
8. If the current note is mostly empty (new session), fill all sections based on the latest exchange.
9. Use the same language as the user's message. Output ONLY valid Markdown — no code fences, no explanations."""


# ------------------------------------------------------------------
# Template for brand-new sessions
# ------------------------------------------------------------------

def _new_session_template(session_id: str) -> str:
    """Return the initial (mostly empty) session note for a new session."""
    return (
        f"# Session Note — {session_id}\n\n"
        "## Current Status\n"
        "(New session — no activity yet)\n\n"
        "## Key Decisions\n"
        "(None yet)\n\n"
        "## Open Questions\n"
        "(None yet)\n\n"
        "## Next Steps\n"
        "(None yet)\n"
    )


# ------------------------------------------------------------------
# SessionMemory
# ------------------------------------------------------------------

class SessionMemory:
    """Manages persistent, structured Markdown session notes on disk."""

    NOTES_DIR = "data/memory/sessions"

    def __init__(self, sessions_dir: str | None = None) -> None:
        self._sessions_dir = Path(sessions_dir or self.NOTES_DIR)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> Optional[str]:
        """Read the session note from disk.  Returns ``None`` if not found."""
        note_path = self.get_note_path(session_id)
        try:
            return note_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def save(self, session_id: str, note: str) -> None:
        """
        Write the session note to disk atomically.

        Writes to a temporary sibling file first, then replaces the
        target path so that a crash mid-write never corrupts the
        existing note.
        """
        note_path = self.get_note_path(session_id)
        tmp_path = note_path.with_suffix(".md.tmp")

        tmp_path.write_text(note, encoding="utf-8")
        os.replace(tmp_path, note_path)  # atomic on Windows + POSIX

    def update(
        self,
        session_id: str,
        current_note: str,
        user_message: str,
        assistant_message: str,
        provider,  # BaseProvider for the LLM call
    ) -> str:
        """
        Call the LLM to produce an updated session note based on the
        latest user-assistant exchange.

        Args:
            session_id: Session identifier (used for logging).
            current_note: The existing session note (empty string for new).
            user_message: The user's latest message text.
            assistant_message: The assistant's latest response text.
            provider: LLM provider for the note-update call.

        Returns:
            The updated session note as a Markdown string.
        """
        # Seed new sessions with the empty template
        if not current_note or not current_note.strip():
            current_note = _new_session_template(session_id)

        # Build a compact prompt with only the current note and latest
        # exchange — the note itself is the cumulative summary.
        user_prompt = (
            f"## CURRENT SESSION NOTE\n\n{current_note}\n\n"
            f"## LATEST EXCHANGE\n\n"
            f"User: {user_message[:800]}\n\n"
            f"Assistant: {assistant_message[:800]}"
        )

        response = provider.chat([
            {"role": "system", "content": SESSION_NOTE_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        updated = (response.content or "").strip()

        # Strip markdown code fences if the LLM wrapped its output
        if updated.startswith("```"):
            # Remove opening fence line (e.g. ```markdown or ```)
            first_nl = updated.find("\n")
            if first_nl != -1:
                updated = updated[first_nl + 1:]
            # Remove closing fence
            if updated.endswith("```"):
                updated = updated[:-3]
            updated = updated.strip()

        return updated

    def exists(self, session_id: str) -> bool:
        """Return ``True`` if a ``.md`` file exists for this session."""
        return self.get_note_path(session_id).is_file()

    def get_note_path(self, session_id: str) -> Path:
        """Return the filesystem path to the session note ``.md`` file."""
        return self._sessions_dir / f"{session_id}.md"
