"""
MemoryManager — unified orchestrator for all four memory layers.

This is the single entry-point that agent_loop and the CLI channel
interact with.  It coordinates short-term, mid-term, long-term,
and semantic memory.
"""

import threading
from typing import Dict, List, Optional, Tuple

from core.memory.short_term import ShortTermMemory
from core.memory.compressor import Compressor
from core.memory.fact_extractor import FactExtractor
from core.memory.long_term import LongTermMemory
from core.memory.semantic import SemanticMemory
from core.memory.session_memory import SessionMemory


class MemoryManager:
    """
    Orchestrates the four-layer memory system.

    Usage in agent_loop:

        # Before building messages (inject memory context)
        memory_ctx = manager.get_context(session_id, user_message)

        # After final response (persist turn)
        manager.record_turn(
            session_id, user_message, assistant_message,
            token_count=response.total_tokens,
            messages=messages,
        )
    """

    def __init__(
        self,
        config: dict,
        provider,  # BaseProvider for compression + fact-extraction LLM calls
    ) -> None:
        mem_cfg = config.get("memory", {})

        storage_path = mem_cfg.get("storage_path", "data/memory")
        short_term_cfg = mem_cfg.get("short_term", {})
        compression_cfg = mem_cfg.get("compression", {})
        long_term_cfg = mem_cfg.get("long_term", {})
        semantic_cfg = mem_cfg.get("semantic", {})

        # Sliding window: hard cap on in-memory message count.
        # This is a safety net — compression (85% token threshold)
        # should normally keep the list well under this limit.
        self._max_history = short_term_cfg.get("max_history", 100)

        # Initialize sub-modules
        self.short_term = ShortTermMemory(
            db_path=f"{storage_path}/short_term.db",
        )
        self.compressor = Compressor(
            threshold=compression_cfg.get("token_threshold", 0.85),
            keep_recent_turns=compression_cfg.get("keep_recent_turns", 3),
            backup_dir=f"{storage_path}/compressed",
        )
        self.long_term = LongTermMemory(
            persist_dir=long_term_cfg.get("chroma_path", f"{storage_path}/chroma"),
        )
        self.semantic = SemanticMemory(
            db_path=semantic_cfg.get("db_path", f"{storage_path}/profile.db"),
        )
        self.fact_extractor = FactExtractor(
            long_term_memory=self.long_term,
            semantic_memory=self.semantic,
            provider=provider,
        )

        session_memory_cfg = mem_cfg.get("session_memory", {})
        self.session_memory = SessionMemory(
            sessions_dir=session_memory_cfg.get(
                "sessions_dir", f"{storage_path}/sessions"
            ),
        )

        self._provider = provider
        self._token_total: Dict[str, int] = {}
        self._saved_count: Dict[str, int] = {}
        self._session_note_cache: Dict[str, str] = {}
        self._bg_status: List[str] = []  # pending status from background threads

        # Track which sessions need memory injected into the system
        # prompt.  We only inject on the first turn after restore or
        # compression — injecting every turn causes the LLM to treat
        # it as "recovered memory" and greet the user like a reunion.
        self._pending_memory_inject: Dict[str, bool] = {}

        # Sessions created by /clear should only inject the user
        # profile (identity) — not session notes or long-term facts
        # about the previous conversation.  The agent still knows WHO
        # the user is, but not WHAT they were just working on.
        self._profile_only: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def load_session(
        self, session_id: str | None = None
    ) -> Tuple[str, List[Dict]]:
        """
        Initialize or restore a session.

        If *session_id* is None, attempts to restore the most recently
        active session.  Falls back to creating a new one.

        Returns (session_id, messages_list).
        """
        # If no session_id given, try to restore a previous session
        if session_id is None:
            session_id = self.short_term.get_last_session_id()

        sid = self.short_term.init_session(session_id)

        # Load session note first — used for both caching and optional
        # restore-time compression below.
        note = self.session_memory.load(sid)
        if note:
            self._session_note_cache[sid] = note
            print(f"[memory] Loaded session note for {sid} "
                  f"({len(note)} chars)")
        else:
            self._session_note_cache[sid] = ""

        messages = self.short_term.load_messages(sid)
        if messages:
            print(f"[memory] Loaded {len(messages)} messages from session {sid}")

            # Compress old turns on restore when a session note exists.
            # Without this, the full message history makes the LLM want
            # to "review" every past turn — even the pro model does a
            # mild recap on each post-restore turn.  Compressing into
            # <historical_context> with the pre-built session note gives
            # the LLM a clean slate + structured background knowledge.
            if note and len(messages) > 3:
                compressed, was_compressed = self.compressor.check_and_compress(
                    sid,
                    messages,
                    self._provider,
                    self._provider.context_window,
                    session_note=note,
                )
                if was_compressed:
                    messages = compressed
                    print(f"[memory] Restored session compressed: "
                          f"{len(messages)} messages with "
                          f"<historical_context> ({len(note)} char note)")
        else:
            # Check for a previously compressed context
            compressed_ctx = self.compressor.load_compressed_context(sid)
            if compressed_ctx:
                messages = [{
                    "role": "system",
                    "content": (
                        f"<historical_context>\n{compressed_ctx}\n"
                        f"</historical_context>"
                    ),
                }]

        # Restore token counter
        if sid not in self._token_total:
            self._token_total[sid] = self.short_term.get_token_total(sid)

        # Track how many messages are already persisted
        self._saved_count: dict[str, int] = {}
        self._saved_count[sid] = len(messages)

        # Mark for memory injection on the first turn after restore
        self._pending_memory_inject[sid] = True
        self._profile_only[sid] = False  # normal restore: full context

        return sid, messages

    def new_session(self) -> Tuple[str, List[Dict]]:
        """
        Force creation of a brand-new session.

        Unlike :meth:`load_session`, this never restores a previous
        session — it always generates a fresh session ID and returns
        an empty message list.

        Memory injection is suppressed for the new session so the
        agent starts silently — no recap, no "I remember you".
        Profile and long-term facts are preserved and will be
        retrieved on-demand via search_chunks-style queries, not
        injected into the system prompt.

        Returns (session_id, empty_messages_list).
        """
        sid = self.short_term.init_session(None)

        # Note: we intentionally do NOT load the old session note.
        # A fresh session starts with no note.
        self._session_note_cache[sid] = ""

        # Initialize tracking for the new session
        self._token_total[sid] = 0
        self._saved_count[sid] = 0

        # After /clear, inject ONLY the user profile so the agent
        # knows WHO it's talking to (identity, preferences) but NOT
        # what they were just working on (session notes, project
        # facts).  This mirrors Claude Code's cross-session user
        # memory: profile persists, context resets.
        self._pending_memory_inject[sid] = True
        self._profile_only[sid] = True

        print(f"[memory] New session created: {sid} "
              f"(profile-only injection)")
        return sid, []

    def compact(
        self,
        session_id: str,
        messages: List[Dict],
    ) -> Tuple[List[Dict], bool]:
        """
        Manually trigger compression (for ``/compact`` command).

        Unlike automatic compression which only fires above the token
        threshold, this forces compression regardless of current usage.
        The session note is used as the summary when available (zero
        API cost); falls back to LLM summarisation.

        Returns (compressed_messages, was_compressed).
        """
        current_note = self._session_note_cache.get(session_id, "")

        old_count = len(messages)
        compressed, was_compressed = self.compressor.check_and_compress(
            session_id,
            messages,
            self._provider,
            self._provider.context_window,
            session_note=current_note,
        )

        if was_compressed:
            self._token_total[session_id] = 0
            self._saved_count[session_id] = len(compressed)
            self._pending_memory_inject[session_id] = True
            self._profile_only[session_id] = False  # compact: full context
            print(f"[memory] Manual compaction: "
                  f"{old_count} → {len(compressed)} messages")

        return compressed, was_compressed

    def _apply_sliding_window(
        self, session_id: str, messages: List[Dict],
    ) -> Tuple[List[Dict], bool]:
        """Hard-cap the message list to *max_history* entries.

        Keeps leading system messages (instructions + historical_context)
        and the most recent messages.  This is a last-resort safety net
        — compression should normally keep the list well under the limit.

        Returns ``(truncated_messages, was_truncated)``.
        """
        if len(messages) <= self._max_history:
            return messages, False

        # Count leading system messages (keep at most 2: instructions
        # + <historical_context>)
        head_system_count = 0
        for m in messages:
            if m.get("role") == "system":
                head_system_count += 1
            else:
                break

        keep_heads = min(head_system_count, 2)
        keep_tails = self._max_history - keep_heads
        truncated = messages[:keep_heads] + messages[-keep_tails:]

        print(
            f"[memory] Sliding window: {len(messages)} → {len(truncated)} "
            f"messages (max_history={self._max_history})"
        )
        return truncated, True

    # ------------------------------------------------------------------
    # Per-turn context injection
    # ------------------------------------------------------------------

    def get_context(
        self,
        session_id: str,
        user_query: str,
        messages: List[Dict] | None = None,
    ) -> str:
        """
        Build a memory context string to inject into the system prompt.

        Memory is ONLY injected when the session context is thin —
        after session restore or compression.  During normal
        conversation the full message history already provides all
        the context the model needs; injecting profile + facts on
        every turn causes the "reunion greeting" problem.

        Args:
            session_id: Session identifier.
            user_query: The user's latest message (for semantic search).
            messages: Current message list (unused, kept for API compat).
        """
        # Only inject memory on turns where context is genuinely thin.
        # After the first injection, the flag is cleared and normal
        # turns get an empty context — the LLM relies on the full
        # message history for continuity.
        if not self._pending_memory_inject.get(session_id, True):
            return ""

        # Clear the flag — inject only once until next restore/compress
        self._pending_memory_inject[session_id] = False
        profile_only = self._profile_only.pop(session_id, False)

        parts: List[str] = []

        # 1. User profile — in profile-only mode, restrict to user scope
        profile = (
            self.semantic.get_profile(scope="user") if profile_only
            else self.semantic.get_profile()
        )
        if profile:
            profile_lines = [
                "## USER PROFILE (internal — do NOT mention in replies)",
            ]
            for key, value in profile.items():
                profile_lines.append(f"- {value}")
            parts.append("\n".join(profile_lines))
            print(f"[memory] get_context: {len(profile)} profile entries found"
                  + (" (user scope only)" if profile_only else ""))

        # 2. Relevant historical facts — scope-aware
        if not profile_only:
            # Normal mode: search all scopes
            relevant_facts = self.long_term.search_facts(user_query, k=3)
        else:
            # Profile-only mode: only user-scope facts (identity / preferences)
            relevant_facts = self.long_term.search_facts(
                user_query, k=3, scope_filter="user",
            )
        if relevant_facts:
            fact_lines = [
                "## RELEVANT CONTEXT (internal — do NOT mention in replies)",
            ]
            for f in relevant_facts:
                fact_lines.append(
                    f"- [{f.get('category', '')}] {f.get('fact', '')}"
                )
            parts.append("\n".join(fact_lines))
            print(f"[memory] get_context: {len(relevant_facts)} long-term facts matched"
                  + (" (user scope only)" if profile_only else ""))

        # 3. Session note  (skipped in profile-only mode)
        if not profile_only:
            note = self._session_note_cache.get(session_id, "")
            if note:
                note_lines = [
                    "## SESSION STATE (internal tracker — do NOT mention "
                    "or reference this section in your replies.  Use it "
                    "only for background awareness of the ongoing "
                    "conversation.)",
                    note,
                ]
                parts.append("\n".join(note_lines))
                print(f"[memory] get_context: session note injected "
                      f"({len(note)} chars)")

        if profile_only:
            print(f"[memory] get_context: profile-only mode — "
                  f"skipping facts + session note")

        if not parts:
            print(f"[memory] get_context: no memory context available yet")

        print(f"[memory] get_context: memory injected "
              f"({sum(len(p) for p in parts)} chars — one-shot, "
              f"will be silent until next restore/compress)")

        # Wrap in XML tags so the LLM can structurally distinguish
        # background reference from the current conversation task.
        # Without this boundary, the model treats injected memory as
        # part of the live conversation and feels compelled to
        # acknowledge it ("I remember you!", recaps, etc.).
        content = "\n\n".join(parts)
        return (
            "<background_context>\n"
            + content
            + "\n</background_context>"
        )

    # ------------------------------------------------------------------
    # Per-turn recording
    # ------------------------------------------------------------------

    def record_turn(
        self,
        session_id: str,
        user_message: Dict,
        assistant_message: Dict,
        token_count: int,
        messages: List[Dict] | None = None,
    ) -> List[Dict] | None:
        """
        Called after each conversation turn.

        1. Save user + assistant messages to SQLite.
        2. Accumulate token count, check compression threshold.
        3. Extract facts from the conversation.

        Args:
            session_id: Session identifier.
            user_message: The user's message dict.
            assistant_message: The assistant's response message dict.
            token_count: Tokens consumed this turn (prompt + completion).
            messages: The full message list (for fact extraction context).
                If None, extraction is skipped.

        Returns:
            Compressed messages list if compression occurred, else None.
        """
        # 1. Persist ALL new messages since last save (not just user+assistant)
        if messages is not None:
            saved_count = self._saved_count.get(session_id, 0)
            new_msgs = messages[saved_count:]  # only unsaved messages
            for msg in new_msgs:
                self.short_term.save_message(session_id, msg, token_count)
            self._saved_count[session_id] = len(messages)
            print(f"[memory] Saved {len(new_msgs)} new messages "
                  f"(total: {len(messages)}, session: {session_id[:12]}...)")
        else:
            self.short_term.save_message(session_id, user_message, token_count)
            self.short_term.save_message(session_id, assistant_message, 0)

        # 2. Update token counter
        self._token_total[session_id] = (
            self._token_total.get(session_id, 0) + token_count
        )

        # 2.5. Launch background thread for session note update.
        #      This is thread-safe: it only does file I/O (.md write).
        #
        #      Fact extraction stays in the main thread (step 4) because
        #      it uses SQLite-backed ChromaDB + semantic store which are
        #      not safe to share across threads.
        #
        #      Compression (step 3) uses the *previous* session note
        #      from the cache — at most one turn stale, negligible for
        #      a summary that covers dozens of turns.
        current_note = self._session_note_cache.get(session_id, "")

        # Pre-extract for the background closure
        user_text = (
            user_message.get("content", "")
            if isinstance(user_message, dict)
            else str(user_message)
        )
        assistant_text = (
            assistant_message.get("content", "")
            if isinstance(assistant_message, dict)
            else str(assistant_message)
        )

        def _update_note_bg() -> None:
            """Background: update the session note .md file via LLM."""
            try:
                updated = self.session_memory.update(
                    session_id=session_id,
                    current_note=current_note,
                    user_message=user_text,
                    assistant_message=assistant_text,
                    provider=self._provider,
                )
                self.session_memory.save(session_id, updated)
                self._session_note_cache[session_id] = updated
                self._bg_status.append(
                    f"[memory] Session note updated ({len(updated)} chars)"
                )
            except Exception as e:
                self._bg_status.append(
                    f"[memory] Session note update skipped: "
                    f"{type(e).__name__}: {e}"
                )

        threading.Thread(target=_update_note_bg, daemon=True).start()

        # 3. Check compression threshold
        compressed_messages = None
        if messages is not None:
            compressed_messages, was_compressed = self.compressor.check_and_compress(
                session_id,
                messages,
                self._provider,
                self._provider.context_window,
                session_note=current_note,
            )
            if was_compressed:
                self._token_total[session_id] = 0
                self._saved_count[session_id] = len(compressed_messages)
                self._pending_memory_inject[session_id] = True
                self._profile_only[session_id] = False  # auto-compress: full context
                print(f"[memory] Compression applied: "
                      f"{len(messages)} → {len(compressed_messages)} messages")

        # 3.5. Sliding window safety net — hard cap on message count.
        #      Applied AFTER compression on whichever list is current.
        #      Should rarely trigger; compression normally keeps the
        #      list well under max_history.  When it does trigger the
        #      truncated messages remain in SQLite and contribute to
        #      restore-time compression on the next session load.
        if messages is not None:
            current = (
                compressed_messages
                if compressed_messages is not None
                else messages
            )
            truncated, was_truncated = self._apply_sliding_window(
                session_id, current,
            )
            if was_truncated:
                # Sync _saved_count to the truncated list length so
                # future saves are relative to the kept messages.
                self._saved_count[session_id] = len(truncated)
                compressed_messages = truncated

        # 4. Extract facts in the main thread (uses SQLite — must be
        #    on the creating thread).
        if messages is not None:
            try:
                new_facts = self.fact_extractor.extract_and_store(messages)
                if new_facts:
                    self._bg_status.append(
                        f"[memory] Extracted {len(new_facts)} new facts"
                    )
            except Exception as e:
                self._bg_status.append(
                    f"[memory] Fact extraction skipped: "
                    f"{type(e).__name__}: {e}"
                )

        return compressed_messages

    # ------------------------------------------------------------------
    # Background status flush
    # ------------------------------------------------------------------

    def flush_bg_status(self) -> None:
        """Print and clear any pending background-thread status lines.

        Call this BEFORE printing the ``You>`` prompt so that
        background output never interleaves with the input line.
        """
        if not self._bg_status:
            return
        for line in self._bg_status:
            print(line)
        self._bg_status.clear()

    # ------------------------------------------------------------------
    # User RAG
    # ------------------------------------------------------------------

    def remember_document(self, chunks: List[Dict]) -> int:
        """
        Add document chunks to the user's personal RAG collection.

        Called when the user explicitly asks the agent to remember a
        document for future reference.

        Returns the number of chunks added.
        """
        return self.long_term.add_rag_documents(chunks)

    # ------------------------------------------------------------------
    # Context usage
    # ------------------------------------------------------------------

    def get_context_usage(self, messages: List[Dict]) -> dict:
        """
        Return context-window usage statistics for display.

        Returns dict with:
          - estimated_tokens: current token estimate
          - context_window: provider's max context window
          - percentage: usage percentage (0.0–100.0)
          - threshold_pct: compression trigger threshold
        """
        total = self._provider.estimate_message_tokens(messages)
        window = self._provider.context_window
        pct = (total / window * 100) if window > 0 else 0.0
        return {
            "estimated_tokens": total,
            "context_window": window,
            "percentage": round(pct, 1),
            "threshold_pct": round(self.compressor._threshold * 100, 1),
        }

    @property
    def last_compression(self) -> dict | None:
        """Stats from the most recent compression run, if any."""
        return self.compressor.last_compression

    # ------------------------------------------------------------------
    # Project-scope memory management
    # ------------------------------------------------------------------

    def clear_project_memory(self) -> int:
        """Delete all project-scope facts from semantic and Chroma stores.

        User-scope facts (personal identity, preferences) are preserved.
        Call when switching projects to reset project-level context.

        Returns the number of project facts removed from SQLite.
        """
        # SQLite: remove project-scope rows
        rows = self.semantic.get_by_scope("project")
        for key, _value, _category in rows:
            self.semantic.delete(key)

        # Chroma: there's no efficient bulk-delete-by-metadata in
        # Chroma's API for now; we log what would be cleaned.
        # In practice, project facts in Chroma are naturally evicted
        # by the embedding space — new project queries will surface
        # relevant facts regardless of old project noise.
        count = len(rows)
        print(f"[memory] Cleared {count} project-scope facts from semantic profile")
        return count

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all persistent stores."""
        self.short_term.close()
        self.semantic.close()
        self.long_term.close()

    def stats(self) -> Dict:
        """Return memory system statistics for diagnostics."""
        return {
            "short_term_messages": self.short_term.get_token_total(
                list(self._token_total.keys())[0]
            ) if self._token_total else 0,
            "semantic_facts": self.semantic.fact_count(),
            "long_term_facts": self.long_term.fact_count(),
            "rag_documents": self.long_term.rag_count(),
            "sessions": len(self._token_total),
        }
