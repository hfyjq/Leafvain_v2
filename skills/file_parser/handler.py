"""
File parser skill handlers: parse_pdf, parse_txt.

Hard constraints enforced here:
  1. Streaming: fitz page iterator (PDF) / line iterator (TXT) — no read().
  2. Security: validate_path() BEFORE any file is opened.
  3. Chunking: all text flows through core.chunker.

Note: search_chunks and summarize_document are handled by agent_loop.py
directly (they operate on session_chunks, not on files), but their tool
definitions live in manifest.yaml so the LLM sees them.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from core.chunker import chunk_stream
from core.security import SecurityGuard

# Injected by skill_loader at startup
_security: SecurityGuard | None = None


def init(security: SecurityGuard) -> None:
    """Called by skill_loader after module import. Sets the shared security guard."""
    global _security
    _security = security


# ------------------------------------------------------------------
# Tool handlers
# ------------------------------------------------------------------

def parse_pdf(file_path: str) -> List[Dict]:
    """
    Stream-parse a PDF file.

    - Opens with fitz (PyMuPDF) using a context manager.
    - Iterates pages one at a time: page.get_text().
    - Each page's text is fed into chunk_stream() and then released.
    - The full document text is NEVER accumulated in memory.

    Args:
        file_path: Path to the PDF file. Tries as-is, then under data/workspaces/.

    Returns:
        List of chunk dicts: {"page", "chunk_index", "text"}.
    """
    _require_security()

    resolved_path = _resolve_path(file_path)

    # Security: validate BEFORE touching the file
    resolved = _security.validate_path(str(resolved_path))  # type: ignore[union-attr]

    def page_iterator() -> Iterator[Tuple[int, str]]:
        with fitz.open(str(resolved)) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                yield (page_num, text)
                # page reference released at end of loop body

    return chunk_stream(page_iterator())


def parse_txt(file_path: str, encoding: str = "utf-8") -> List[Dict]:
    """
    Stream-parse a TXT file.

    - Opens with built-in open() using a context manager.
    - Reads ONE LINE AT A TIME (never .read() or .readlines()).
    - Accumulates lines into pseudo-pages (every 100 lines = 1 "page").
    - Each pseudo-page is fed into chunk_stream() and released.

    Args:
        file_path: Path to the TXT file. Tries as-is, then under data/workspaces/.
        encoding: File encoding (default "utf-8").

    Returns:
        List of chunk dicts: {"page", "chunk_index", "text"}.
    """
    _require_security()

    resolved_path = _resolve_path(file_path)

    # Security: validate BEFORE touching the file
    resolved = _security.validate_path(str(resolved_path))  # type: ignore[union-attr]

    def line_iterator() -> Iterator[Tuple[int, str]]:
        with open(str(resolved), "r", encoding=encoding) as f:
            page_num = 1
            buffer: List[str] = []
            line_count = 0
            for line in f:
                buffer.append(line)
                line_count += 1
                if line_count >= 100:
                    yield (page_num, "".join(buffer))
                    page_num += 1
                    buffer = []
                    line_count = 0
            if buffer:  # flush remainder
                yield (page_num, "".join(buffer))

    return chunk_stream(line_iterator())


# ------------------------------------------------------------------
# Placeholder handlers for search_chunks / summarize_document
# These exist for registry lookup only — real logic is in agent_loop.
# ------------------------------------------------------------------

def search_chunks(query: str = "", file_path: str = "") -> List[Dict]:
    """
    Search indexed chunks (handled by agent_loop via BM25Retriever).
    This function exists for registry lookup only — the real logic is in agent_loop.
    """
    raise NotImplementedError("search_chunks is handled by agent_loop")


def summarize_document(file_path: str = "") -> str:
    """
    Summarize indexed document (handled by agent_loop via iterative_summarize).
    This function exists for registry lookup only — the real logic is in agent_loop.
    """
    raise NotImplementedError("summarize_document is handled by agent_loop")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _require_security() -> None:
    if _security is None:
        raise RuntimeError(
            "SecurityGuard not initialized. "
            "The skill_loader must call init() before any handler is invoked."
        )


def _resolve_path(file_path: str) -> Path:
    """Try file_path as-is; if it doesn't exist, try under data/workspaces/."""
    p = Path(file_path)
    if p.exists():
        return p.resolve()
    # Try under workspaces
    alt = Path("data/workspaces") / p.name
    if alt.exists():
        return alt.resolve()
    # Neither exists — return the original for security to reject clearly
    return p.resolve()
