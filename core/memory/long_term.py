"""
Long-term memory: Chroma vector store for conversation facts and user RAG.

Two collections:
  - conversation_facts: structured facts extracted from conversations
  - user_rag:           user's personal document knowledge base

Embeddings are generated locally via Chroma's default
sentence-transformers model (all-MiniLM-L6-v2) — no API key required.

For better Chinese text retrieval in production, swap the embedding
function to a multilingual model (e.g. paraphrase-multilingual-MiniLM-L12-v2).
"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings


class LongTermMemory:
    """Chroma-backed vector store for long-term memory."""

    COLLECTION_FACTS = "conversation_facts"
    COLLECTION_RAG = "user_rag"

    def __init__(self, persist_dir: str = "data/memory/chroma") -> None:
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Ensure both collections exist
        self._facts = self._client.get_or_create_collection(
            name=self.COLLECTION_FACTS,
            metadata={"description": "Conversation-extracted facts"},
        )
        self._rag = self._client.get_or_create_collection(
            name=self.COLLECTION_RAG,
            metadata={"description": "User personal RAG documents"},
        )

    # ------------------------------------------------------------------
    # Public API — facts
    # ------------------------------------------------------------------

    def add_facts(self, facts: List[Dict]) -> int:
        """
        Store extracted facts in the conversation_facts collection.

        Each fact dict should have: ``fact`` (str), ``category`` (str),
        ``scope`` (str: "user" | "project"), and optionally ``confidence`` (float).

        Returns the number of facts added.
        """
        if not facts:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for f in facts:
            fact_text = f.get("fact", "")
            if not fact_text.strip():
                continue
            fact_id = f.get("id") or f"fact_{uuid.uuid4().hex[:16]}"
            ids.append(fact_id)
            documents.append(fact_text)
            metadatas.append({
                "category": f.get("category", "general"),
                "confidence": f.get("confidence", 1.0),
                "scope": f.get("scope", "project"),
            })

        if ids:
            self._facts.add(ids=ids, documents=documents, metadatas=metadatas)

        return len(ids)

    def search_facts(
        self, query: str, k: int = 3, scope_filter: str | None = None,
    ) -> List[Dict]:
        """
        Search conversation facts by semantic similarity.

        If *scope_filter* is given (``"user"`` or ``"project"``),
        only facts with that scope are returned.

        Returns list of {fact, category, scope, confidence, distance}.
        """
        where = None
        if scope_filter:
            where = {"scope": scope_filter}

        results = self._facts.query(
            query_texts=[query], n_results=k,
            where=where,
        )
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        output: List[Dict] = []
        ids_list = results["ids"][0]
        docs_list = results["documents"][0]
        meta_list = results["metadatas"][0]
        dist_list = results["distances"][0]

        for i in range(len(ids_list)):
            output.append({
                "id": ids_list[i],
                "fact": docs_list[i] if docs_list else "",
                "category": meta_list[i].get("category", "") if meta_list else "",
                "scope": meta_list[i].get("scope", "project") if meta_list else "project",
                "confidence": meta_list[i].get("confidence", 1.0) if meta_list else 1.0,
                "distance": dist_list[i] if dist_list else 0.0,
            })

        return output

    def delete_fact(self, fact_id: str) -> None:
        """Remove a single fact from the collection."""
        try:
            self._facts.delete(ids=[fact_id])
        except Exception:
            pass  # already deleted or never existed

    # ------------------------------------------------------------------
    # Public API — user RAG
    # ------------------------------------------------------------------

    def add_rag_documents(self, chunks: List[Dict]) -> int:
        """
        Add document chunks to the user's personal RAG collection.

        Each chunk dict should have: ``text`` (str) and optionally
        ``source`` (str), ``page`` (int).
        """
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for c in chunks:
            chunk_text = c.get("text", "")
            if not chunk_text.strip():
                continue
            chunk_id = f"rag_{uuid.uuid4().hex[:16]}"
            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append({
                "source": c.get("source", "unknown"),
                "page": c.get("page", 0),
            })

        if ids:
            self._rag.add(ids=ids, documents=documents, metadatas=metadatas)

        return len(ids)

    def search_rag(self, query: str, k: int = 3) -> List[Dict]:
        """Search user RAG collection by semantic similarity."""
        results = self._rag.query(query_texts=[query], n_results=k)
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        output: List[Dict] = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i] if results["documents"] else "",
                "source": results["metadatas"][0][i].get("source", "")
                if results["metadatas"] else "",
                "page": results["metadatas"][0][i].get("page", 0)
                if results["metadatas"] else 0,
            })
        return output

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def fact_count(self) -> int:
        return self._facts.count()

    def rag_count(self) -> int:
        return self._rag.count()

    def close(self) -> None:
        """Release ChromaDB resources.

        On Windows, SQLite WAL mode holds file locks until the
        underlying segment server is stopped.  Without this, the
        database files cannot be deleted while the process lives.
        """
        # 1. Stop the internal segment server (releases SQLite handles)
        try:
            if hasattr(self._client, "_system") and self._client._system is not None:
                self._client._system.stop()
        except Exception:
            pass

        # 2. Drop collection references
        self._facts = None
        self._rag = None

        # 3. Drop the client reference so GC can clean up
        self._client = None
