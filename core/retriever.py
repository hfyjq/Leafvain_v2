"""
BM25 keyword retriever with jieba tokenization for Chinese text.

Provides lightweight document chunk retrieval WITHOUT a vector database.
In Phase 1 this replaces Chroma — all indexes are in-memory.
"""

from typing import Dict, List

import jieba
from rank_bm25 import BM25Okapi


def jieba_tokenizer(text: str) -> List[str]:
    """
    Tokenize *text* with jieba.

    For Chinese, jieba performs dictionary-based word segmentation.
    For pure-ASCII / English text, jieba effectively falls back to
    whitespace-delimited tokenization, which is correct for BM25.
    """
    return list(jieba.cut(text))


class BM25Retriever:
    """
    A lightweight in-memory retriever that indexes chunk dicts and
    returns the top-K most relevant chunks for a query using BM25.

    **Hard constraint**: retrieve() returns at most *top_k* chunks.
    The remaining chunks are never exposed. This enforces context isolation.
    """

    def __init__(self) -> None:
        self._chunks: List[Dict] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: List[List[str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, chunks: List[Dict]) -> None:
        """
        Build / rebuild the BM25 index from *chunks*.

        Each chunk dict MUST have a ``"text"`` key.  Page/position
        metadata is preserved but not used for ranking.

        Replaces any existing index — each call is a full re-index.
        (Fine for Phase 1 since chunks are collected once per document.)
        """
        self._chunks = chunks
        self._tokenized_corpus = [jieba_tokenizer(c["text"]) for c in chunks]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Return up to *top_k* most relevant chunks for *query*.

        Returns an empty list if no chunks have been indexed or if
        no chunk has a positive BM25 score for the query.

        **Hard constraint**: only chunks with score > 0 are returned.
        This prevents irrelevant chunks from polluting the LLM context.
        """
        if self._bm25 is None:
            return []

        tokenized_query = jieba_tokenizer(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Rank indices by descending score
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        # Take top_k, but only those with positive scores
        result = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            result.append(self._chunks[idx])
            if len(result) >= top_k:
                break

        return result

    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)
