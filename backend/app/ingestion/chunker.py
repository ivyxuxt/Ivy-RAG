"""
Heading-aware paragraph chunking with token budget and overlap.

Design rationale (see README):
- Section-aware splits improve citation precision over fixed-size windows.
- Token budget (not just char count) keeps embedding context consistent.
- Overlap (75 tokens ≈ 300 chars) prevents answers at chunk boundaries from being missed.
- 500 tokens is a well-established sweet spot for embedding models.
- Never cut mid-sentence; prefer paragraph then sentence boundaries.
"""
import re
from typing import List, Tuple

from app.config import settings

# Rough token estimate: 1 token ≈ 4 characters (English average)
_CHARS_PER_TOKEN = 4
_TARGET_CHARS = settings.CHUNK_TARGET_TOKENS * _CHARS_PER_TOKEN      # ~2000
_OVERLAP_CHARS = settings.CHUNK_OVERLAP_TOKENS * _CHARS_PER_TOKEN    # ~300
_MIN_CHARS = settings.MIN_CHUNK_CHARS                                 # 300


def chunk_pages(
    pages: List[Tuple[int, str]],
    doc_id: str,
    doc_name: str,
) -> List[dict]:
    """
    Convert extracted pages into chunks.

    Args:
        pages:    List of (page_number, text) from parser.py
        doc_id:   UUID of the parent document
        doc_name: Original filename

    Returns:
        List of chunk dicts with full metadata.
    """
    # Step 1: Flatten all lines with page numbers
    all_lines: List[Tuple[int, str]] = []
    for page_num, text in pages:
        for line in text.split("\n"):
            all_lines.append((page_num, line))

    # Step 2: Split into sections at heading boundaries
    sections = _split_into_sections(all_lines)

    # Step 3: Within each section split into paragraphs, then pack into chunks
    chunks: List[dict] = []
    chunk_index = 0

    for section in sections:
        section_title = section["title"]
        paragraphs = _split_paragraphs(section["lines"])

        # Buffer accumulates text until it hits the target size
        buf_text = ""
        buf_pages: List[int] = []

        for para_text, para_pages in paragraphs:
            if not para_text.strip():
                continue

            candidate = (buf_text + "\n\n" + para_text).strip() if buf_text else para_text

            if len(candidate) <= _TARGET_CHARS:
                # Paragraph fits in the current buffer
                buf_text = candidate
                buf_pages = sorted(set(buf_pages + para_pages))
            else:
                # Flush current buffer as a chunk (if large enough)
                if buf_text:
                    chunk = _make_chunk(buf_text, doc_id, doc_name, section_title, buf_pages, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
                    # Carry overlap from end of this buffer
                    overlap = _get_overlap_tail(buf_text)
                    buf_text = (overlap + "\n\n" + para_text).strip() if overlap else para_text
                    buf_pages = para_pages
                else:
                    # Paragraph alone is bigger than target — split it
                    sub_chunks = _split_large_paragraph(
                        para_text, para_pages, doc_id, doc_name,
                        section_title, chunk_index
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    if sub_chunks:
                        overlap = _get_overlap_tail(sub_chunks[-1]["text"])
                        buf_text = overlap
                        buf_pages = para_pages
                    else:
                        buf_text = ""
                        buf_pages = []

        # Flush remaining buffer
        if buf_text.strip():
            if len(buf_text) >= _MIN_CHARS:
                chunk = _make_chunk(buf_text, doc_id, doc_name, section_title, buf_pages, chunk_index)
                chunks.append(chunk)
                chunk_index += 1
            elif chunks:
                # Merge short tail into previous chunk
                merged = (chunks[-1]["text"] + "\n\n" + buf_text).strip()
                if len(merged) <= _TARGET_CHARS * 1.3:
                    chunks[-1]["text"] = merged
                    chunks[-1]["token_count"] = len(merged) // _CHARS_PER_TOKEN
                    if buf_pages:
                        chunks[-1]["page_end"] = max(
                            chunks[-1].get("page_end") or 0,
                            max(buf_pages)
                        )
                else:
                    chunk = _make_chunk(buf_text, doc_id, doc_name, section_title, buf_pages, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                # Only content — add even if short
                chunk = _make_chunk(buf_text, doc_id, doc_name, section_title, buf_pages, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

    return chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_into_sections(all_lines: List[Tuple[int, str]]) -> List[dict]:
    """Group lines into sections separated by detected headings."""
    sections = []
    current = {"title": None, "lines": []}

    for page_num, line in all_lines:
        if _is_heading(line) and current["lines"]:
            sections.append(current)
            current = {"title": line.strip(), "lines": []}
        else:
            current["lines"].append((page_num, line))

    sections.append(current)
    return sections


def _is_heading(line: str) -> bool:
    """
    Heuristic heading detection.
    Covers: ALL-CAPS, numbered sections (3.2), very short standalone lines.
    """
    s = line.strip()
    if not s or len(s) < 2:
        return False
    if len(s) >= 3 and s.isupper() and len(s) < 80:
        return True
    if re.match(r"^(\d+\.)+\d*\s+\S", s):
        return True
    if re.match(r"^(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+\S", s, re.IGNORECASE):
        return True
    # Short title-case line (< 60 chars, no terminal period, starts capitalized)
    if len(s) < 60 and not s.endswith(".") and re.match(r"^[A-Z]", s) and s == s.title():
        return True
    return False


def _split_paragraphs(lines: List[Tuple[int, str]]) -> List[Tuple[str, List[int]]]:
    """Split lines into paragraphs at blank lines."""
    paragraphs: List[Tuple[str, List[int]]] = []
    cur_lines: List[str] = []
    cur_pages: List[int] = []

    for page_num, line in lines:
        if not line.strip():
            if cur_lines:
                paragraphs.append((" ".join(cur_lines), sorted(set(cur_pages))))
                cur_lines = []
                cur_pages = []
        else:
            cur_lines.append(line.strip())
            cur_pages.append(page_num)

    if cur_lines:
        paragraphs.append((" ".join(cur_lines), sorted(set(cur_pages))))

    return paragraphs


def _split_large_paragraph(
    text: str,
    pages: List[int],
    doc_id: str,
    doc_name: str,
    section_title,
    start_index: int,
) -> List[dict]:
    """Split a single oversized paragraph at sentence boundaries."""
    chunks = []
    remaining = text
    idx = start_index
    overlap = ""

    while len(remaining) > _TARGET_CHARS:
        split_at = _sentence_split_point(remaining, _TARGET_CHARS)
        chunk_text = (overlap + " " + remaining[:split_at]).strip() if overlap else remaining[:split_at]
        if chunk_text:
            chunks.append(_make_chunk(chunk_text, doc_id, doc_name, section_title, pages, idx))
            idx += 1
        overlap = _get_overlap_tail(remaining[:split_at])
        remaining = remaining[split_at:].strip()

    if remaining:
        final_text = (overlap + " " + remaining).strip() if overlap else remaining
        chunks.append(_make_chunk(final_text, doc_id, doc_name, section_title, pages, idx))

    return chunks


def _sentence_split_point(text: str, target_chars: int) -> int:
    """Find a sentence boundary near target_chars."""
    if len(text) <= target_chars:
        return len(text)
    window_start = max(0, target_chars - 200)
    window_end = min(len(text), target_chars + 200)
    window = text[window_start:window_end]
    # Look for ". " or "! " or "? " followed by a capital letter
    for m in reversed(list(re.finditer(r"[.!?]\s+(?=[A-Z\"])", window))):
        return window_start + m.end()
    # Fallback: nearest space
    space = text.rfind(" ", 0, target_chars)
    return space if space > 0 else target_chars


def _get_overlap_tail(text: str) -> str:
    """Return the last ~OVERLAP_CHARS of text, starting at a word boundary."""
    if len(text) <= _OVERLAP_CHARS:
        return text
    tail = text[-_OVERLAP_CHARS:]
    space = tail.find(" ")
    return tail[space + 1:].strip() if space >= 0 else tail


def _make_chunk(
    text: str,
    doc_id: str,
    doc_name: str,
    section_title,
    pages: List[int],
    chunk_index: int,
) -> dict:
    page_start = min(pages) if pages else None
    page_end = max(pages) if pages else None
    chunk_id = f"{doc_id}#p{page_start}#c{chunk_index}"
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_start": page_start,
        "page_end": page_end,
        "section_title": section_title,
        "chunk_index": chunk_index,
        "text": text,
        "token_count": len(text) // _CHARS_PER_TOKEN,
    }
