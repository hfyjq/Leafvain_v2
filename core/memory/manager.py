"""
MemoryManager — unified orchestrator for all four memory layers.

This is the single entry-point that agent_loop and the CLI channel
interact with.  It coordinates short-term, mid-term, long-term,
and semantic memory.
"""

from typing import Dict, List, Optional, Tuple

from core.memory.short_term import ShortTermMemory
from core.memory.compressor import Compressor
from core.memory.fact_extractor import FactExtractor
from core.memory.long_term import LongTermMemory
from core.memory.semantic import SemanticMemory


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

        self._provider = provider
        self._token_total: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def load_session(
        self, session_id: str | None = None
    ) -> Tuple[str, List[Dict]]:
        """
        Initialize or restore a session.

        Returns (session_id, messages_list).  If the session already
        exists in the database, its messages are restored.
        """
        sid = self.short_term.init_session(session_id)

        messages = self.short_term.load_messages(sid)

        # Check for a previously compressed context
        if not messages:
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

        return sid, messages

    # ------------------------------------------------------------------
    # Per-turn context injection
    # ------------------------------------------------------------------

    def get_context(
        self, session_id: str, user_query: str
    ) -> str:
        """
        Build a memory context string to inject into the system prompt.

        Includes:
          - User profile from semantic memory
          - Relevant historical facts from long-term memory
          - Compressed conversation context (if any)
        """
        parts: List[str] = []

        # 1. User profile
        profile = self.semantic.get_profile()
        if profile:
            profile_lines = ["## User Profile"]
            for key, value in profile.items():
                profile_lines.append(f"- {value}")
            parts.append("\n".join(profile_lines))

        # 2. Relevant historical facts (search long-term memory)
        relevant_facts = self.long_term.search_facts(user_query, k=3)
        if relevant_facts:
            fact_lines = ["## Relevant Context (from past conversations)"]
            for f in relevant_facts:
                fact_lines.append(f"- [{f.get('category', '')}] {f.get('fact', '')}")
            parts.append("\n".join(fact_lines))

        # 3. Compressed context is injected by compressor.check_and_compress
        #    directly into the messages list, not here.

        return "\n\n".join(parts) if parts else ""

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
        # 1. Persist messages
        self.short_term.save_message(session_id, user_message, token_count)
        self.short_term.save_message(session_id, assistant_message, 0)

        # 2. Update token counter
        self._token_total[session_id] = (
            self._token_total.get(session_id, 0) + token_count
        )

        # 3. Check compression threshold
        compressed_messages = None
        if messages is not None:
            compressed_messages, was_compressed = self.compressor.check_and_compress(
                session_id,
                messages,
                self._provider,
                self._provider.context_window,
            )
            if was_compressed:
                self._token_total[session_id] = 0  # Reset counter after compression

        # 4. Extract facts (async-optional; skip if no messages provided)
        if messages is not None:
            try:
                self.fact_extractor.extract_and_store(messages)
            except Exception:
                pass  # Fact extraction is best-effort; never crash the agent loop

        return compressed_messages

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
    # Maintenance
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all persistent stores."""
        self.short_term.close()
        self.semantic.close()
        # Chroma client does not need explicit close

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
