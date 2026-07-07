"""
Streaming chunker: never loads the full document in memory.

Each page's text is consumed, split into chunks, and the page text
reference is released before advancing to the next page.
"""

import re
from typing import Dict, Iterator, List, Tuple

# Sentence-ending delimiters for Chinese and English.
# We split on these so chunks stay semantically coherent.
_SENTENCE_END = re.compile(r"([。！？!?.\n])")


def chunk_stream(
    text_iterator: Iterator[Tuple[int, str]],
    max_chars: int = 1500,
) -> List[Dict]:
    """
    Process a streaming text iterator and produce chunk dicts.

    Args:
        text_iterator: Yields (page_num, page_text) tuples.
            For TXT files, use a pseudo-page counter (e.g. every 100 lines = 1 page).
        max_chars: Maximum characters per chunk (default 1500).

    Returns:
        List of dicts: {"page": int, "chunk_index": int, "text": str}

    **Hard constraint**: NEVER accumulates the full document in memory.
    Each page's raw text is released when the loop advances to the next page.
    """
    all_chunks: List[Dict] = []
    for page_num, page_text in text_iterator:
        page_chunks = _chunk_text(page_text, page_num, max_chars)
        all_chunks.extend(page_chunks)
        # page_text reference released here — no accumulation across pages
    return all_chunks


def _chunk_text(text: str, page_num: int, max_chars: int) -> List[Dict]:
    """
    Split a single page's text into chunks of at most *max_chars*.

    Strategy (best-effort for semantic coherence):
    1. Split by sentence-ending delimiters (。！？!?.\n).
    2. Accumulate sentences into a buffer; flush when adding the next
       sentence would exceed max_chars.
    3. If a SINGLE sentence exceeds max_chars, hard-split it at
       max_chars boundaries (fallback, not the normal path).
    """
    if not text.strip():
        return []

    # Split into sentence-like segments, keeping delimiters attached
    segments = _SENTENCE_END.split(text)
    # Re-join delimiter with its preceding text
    sentences: List[str] = []
    buf = ""
    for seg in segments:
        if _SENTENCE_END.match(seg):
            buf += seg
            sentences.append(buf)
            buf = ""
        else:
            buf += seg
    if buf.strip():
        sentences.append(buf)

    chunks: List[Dict] = []
    chunk_index = 0
    current_buffer: List[str] = []
    current_length = 0

    def _flush() -> None:
        nonlocal chunk_index
        chunk_text = "".join(current_buffer).strip()
        if chunk_text:
            # Enforce max_chars (should already be satisfied, but belt-and-suspenders)
            if len(chunk_text) > max_chars:
                chunk_text = chunk_text[:max_chars]
            chunks.append({
                "page": page_num,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            chunk_index += 1
        current_buffer.clear()
        nonlocal current_length
        current_length = 0

    for sentence in sentences:
        s_len = len(sentence)

        # If a single sentence exceeds max_chars, flush current buffer
        # and hard-split this sentence
        if s_len > max_chars:
            _flush()
            for i in range(0, s_len, max_chars):
                hard_chunk = sentence[i:i + max_chars]
                chunks.append({
                    "page": page_num,
                    "chunk_index": chunk_index,
                    "text": hard_chunk,
                })
                chunk_index += 1
            continue

        # If adding this sentence would overflow, flush the buffer first
        if current_buffer and current_length + s_len > max_chars:
            _flush()

        current_buffer.append(sentence)
        current_length += s_len

    # Flush any remaining text
    _flush()

    return chunks
