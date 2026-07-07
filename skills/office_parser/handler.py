"""
Office document parser skill handlers: parse_docx, parse_pptx, parse_xlsx.

Hard constraints enforced here:
  1. Streaming: paragraph/slide/row iterators — no bulk reads.
  2. Security: validate_path() BEFORE any file is opened.
  3. Chunking: all text flows through core.chunker (≤1500 chars, metadata).

Dependencies:
  - python-docx  (Word)
  - python-pptx  (PowerPoint)
  - openpyxl     (Excel — read_only mode)
"""

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from core.chunker import chunk_stream
from core.security import SecurityGuard

# Injected by skill_loader at startup
_security: SecurityGuard | None = None


def init(security: SecurityGuard) -> None:
    """Called by skill_loader after module import."""
    global _security
    _security = security


# ------------------------------------------------------------------
# Word (.docx) — paragraph iterator
# ------------------------------------------------------------------

def parse_docx(file_path: str) -> List[Dict]:
    """
    Stream-parse a Word (.docx) document.

    - Uses python-docx: doc.paragraphs iterator (lazy, no bulk load).
    - Tables are extracted and interleaved at their position.
    - Every 50 paragraphs = 1 pseudo-page → fed to chunk_stream().

    Args:
        file_path: Path to the .docx file.

    Returns:
        List of chunk dicts: {"page", "chunk_index", "text"}.
    """
    _require_security()
    resolved_path = _resolve_path(file_path)
    resolved = _security.validate_path(str(resolved_path))  # type: ignore[union-attr]

    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for Word documents. "
            "Install it with: pip install python-docx"
        )

    def paragraph_iterator() -> Iterator[Tuple[int, str]]:
        doc = Document(str(resolved))
        page_num = 1
        buffer: List[str] = []
        para_count = 0

        # Iterate through the document body in order, interleaving
        # paragraphs and tables.
        # python-docx stores paragraphs and tables separately in
        # doc.paragraphs and doc.tables, but we can use iter_inner_content
        # on the body element to walk them in document order.
        from docx.oxml.ns import qn

        body = doc.element.body
        para_idx = 0
        table_idx = 0

        for child in body:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

            if tag == "p":
                # Paragraph
                if para_idx < len(doc.paragraphs):
                    text = doc.paragraphs[para_idx].text
                    if text.strip():
                        buffer.append(text)
                        para_count += 1
                    para_idx += 1
            elif tag == "tbl":
                # Table
                if table_idx < len(doc.tables):
                    table = doc.tables[table_idx]
                    table_lines = []
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        table_lines.append(" | ".join(cells))
                    if table_lines:
                        buffer.append("\n".join(table_lines))
                        para_count += 1
                    table_idx += 1

            if para_count >= 50:
                yield (page_num, "\n".join(buffer))
                page_num += 1
                buffer = []
                para_count = 0

        # Flush remaining
        if buffer:
            yield (page_num, "\n".join(buffer))

    return chunk_stream(paragraph_iterator())


# ------------------------------------------------------------------
# PowerPoint (.pptx) — slide iterator
# ------------------------------------------------------------------

def parse_pptx(file_path: str) -> List[Dict]:
    """
    Extract text from a PowerPoint (.pptx) presentation.

    - Uses python-pptx: prs.slides iterator.
    - Each slide = 1 page. Text from all shapes is collected.
    - Fed to chunk_stream() for enforced ≤1500-char chunks.

    Args:
        file_path: Path to the .pptx file.

    Returns:
        List of chunk dicts: {"page", "chunk_index", "text"}.
    """
    _require_security()
    resolved_path = _resolve_path(file_path)
    resolved = _security.validate_path(str(resolved_path))  # type: ignore[union-attr]

    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError(
            "python-pptx is required for PowerPoint files. "
            "Install it with: pip install python-pptx"
        )

    def slide_iterator() -> Iterator[Tuple[int, str]]:
        prs = Presentation(str(resolved))
        for slide_num, slide in enumerate(prs.slides, start=1):
            texts: List[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            texts.append(t)
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        texts.append(" | ".join(cells))
            yield (slide_num, "\n".join(texts))

    return chunk_stream(slide_iterator())


# ------------------------------------------------------------------
# Excel (.xlsx) — row iterator (streaming via read_only)
# ------------------------------------------------------------------

def parse_xlsx(file_path: str, sheet_name: str = "") -> List[Dict]:
    """
    Stream-read an Excel (.xlsx) spreadsheet.

    - Uses openpyxl with read_only=True for true streaming.
    - Rows are read one at a time via iter_rows().
    - Every 100 rows = 1 pseudo-page → fed to chunk_stream().

    Args:
        file_path: Path to the .xlsx file.
        sheet_name: Optional sheet name. Defaults to the active sheet.

    Returns:
        List of chunk dicts: {"page", "chunk_index", "text"}.
    """
    _require_security()
    resolved_path = _resolve_path(file_path)
    resolved = _security.validate_path(str(resolved_path))  # type: ignore[union-attr]

    try:
        from openpyxl import load_workbook
    except ImportError:
        raise ImportError(
            "openpyxl is required for Excel files. "
            "Install it with: pip install openpyxl"
        )

    def row_iterator() -> Iterator[Tuple[int, str]]:
        wb = load_workbook(str(resolved), read_only=True, data_only=True)

        # Determine which sheet to read
        if sheet_name:
            if sheet_name not in wb.sheetnames:
                wb.close()
                raise ValueError(
                    f"Sheet '{sheet_name}' not found. "
                    f"Available sheets: {wb.sheetnames}"
                )
            ws = wb[sheet_name]
        else:
            ws = wb.active

        page_num = 1
        buffer: List[str] = []
        row_count = 0

        # iter_rows yields one row at a time — true streaming
        for row in ws.iter_rows(values_only=True):
            # Convert row tuple to tab-separated string
            row_str = "\t".join(
                str(cell) if cell is not None else "" for cell in row
            )
            if row_str.strip():  # skip entirely empty rows
                buffer.append(row_str)
                row_count += 1

            if row_count >= 100:
                yield (page_num, "\n".join(buffer))
                page_num += 1
                buffer = []
                row_count = 0

        # Flush remaining
        if buffer:
            yield (page_num, "\n".join(buffer))

        wb.close()

    return chunk_stream(row_iterator())


# ------------------------------------------------------------------
# Placeholder handlers for shared tools
# These are registered by file_parser; listed here only so
# skill_loader doesn't look for them in this module.
# ------------------------------------------------------------------

# search_chunks and summarize_document are NOT defined here —
# they are handled by agent_loop.py directly and registered via
# file_parser/manifest.yaml.


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
    """Try file_path as-is; if not found, try under data/workspaces/."""
    p = Path(file_path)
    if p.exists():
        return p.resolve()
    alt = Path("data/workspaces") / p.name
    if alt.exists():
        return alt.resolve()
    return p.resolve()
