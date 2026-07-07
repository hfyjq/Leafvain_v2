---
name: office-parser
description: Parse Microsoft Office documents — Word (.docx), PowerPoint (.pptx), and Excel (.xlsx) — with streaming extraction and chunking. Use when working with Office documents, spreadsheets, presentations, or when the user mentions docx, pptx, xlsx, Word, Excel, PowerPoint, Office files.
metadata:
  kind: skill
  loader: manifest.yaml
  tools: parse_docx, parse_pptx, parse_xlsx
  mcp_note: >
    Directory name is office_parser (underscore) for Python import compatibility.
    SKILL.md name uses office-parser (hyphen) per agentskills.io standard.
    A thin adapter layer (see MCP_ADAPTATION.md) will map these when
    publishing to the MCP ecosystem.
---

# Office Document Parser Skill

## When to use
Use this skill when the user needs to read, search, or summarize Microsoft Office documents.
Supports Word (.docx), PowerPoint (.pptx), and Excel (.xlsx) formats.

## Tools

### parse_docx
Stream-parse a Word document. Paragraphs are read one at a time via python-docx iterator,
accumulated into pseudo-pages (50 paragraphs each), and fed through the unified chunker.
Tables are also extracted and included inline.

### parse_pptx
Extract text from PowerPoint slides. Each slide is treated as one page. Text from all
shapes on each slide is collected, then chunked by the unified chunker.

### parse_xlsx
Stream-read an Excel spreadsheet using openpyxl read_only mode. Rows are read one at a time
via `iter_rows()`, accumulated into pseudo-pages (100 rows each), and chunked.
Optionally specify a sheet name; defaults to the active sheet.

## Shared tools
After parsing with any tool above, use these shared tools (from file-parser skill):
- **search_chunks**: BM25 keyword search across parsed content
- **summarize_document**: Iterative aggregation summary

## Notes
- File path resolution: bare filenames are auto-resolved under `data/workspaces/`
- All paths validated by SecurityGuard before any file is opened
- openpyxl read_only mode ensures Excel files are never fully loaded into memory
- python-docx and python-pptx iterate one element at a time — no bulk reads
