"""Tests for heading-aware chunker."""
import pytest
from app.ingestion.chunker import chunk_pages, _is_heading, _get_overlap_tail


DOC_ID = "test-doc-001"
DOC_NAME = "test.pdf"


def make_pages(text: str, page_num: int = 1):
    return [(page_num, text)]


# ── _is_heading ───────────────────────────────────────────────────────────────

def test_heading_all_caps():
    assert _is_heading("INTRODUCTION") is True


def test_heading_numbered_section():
    assert _is_heading("3.2 Methodology") is True


def test_heading_short_title_case():
    assert _is_heading("Related Work") is True


def test_not_heading_long_sentence():
    assert _is_heading("This is a long sentence that should not be treated as a heading.") is False


def test_not_heading_ends_with_period():
    assert _is_heading("Short Line.") is False


def test_not_heading_empty():
    assert _is_heading("") is False
    assert _is_heading("   ") is False


# ── _get_overlap_tail ─────────────────────────────────────────────────────────

def test_overlap_short_text():
    text = "hello world"
    result = _get_overlap_tail(text)
    assert result == text


def test_overlap_starts_at_word_boundary():
    text = "a " * 200  # very long text
    result = _get_overlap_tail(text)
    assert not result.startswith(" ")


# ── chunk_pages ───────────────────────────────────────────────────────────────

def test_basic_chunking_returns_chunks():
    text = "\n\n".join(["Sentence number %d in the test document." % i for i in range(30)])
    pages = make_pages(text)
    chunks = chunk_pages(pages, DOC_ID, DOC_NAME)
    assert len(chunks) >= 1


def test_chunk_schema():
    text = "\n\n".join(["This is a paragraph about topic %d in the test file." % i for i in range(10)])
    pages = make_pages(text)
    chunks = chunk_pages(pages, DOC_ID, DOC_NAME)
    assert len(chunks) >= 1
    c = chunks[0]
    assert "chunk_id" in c
    assert "doc_id" in c
    assert c["doc_id"] == DOC_ID
    assert "doc_name" in c
    assert c["doc_name"] == DOC_NAME
    assert "text" in c
    assert len(c["text"]) > 0
    assert "token_count" in c
    assert "chunk_index" in c


def test_chunk_ids_unique():
    text = "\n\n".join(["Paragraph %d: " % i + "word " * 50 for i in range(20)])
    pages = make_pages(text)
    chunks = chunk_pages(pages, DOC_ID, DOC_NAME)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_heading_creates_section_boundary():
    # INTRODUCTION goes into the first section's lines (no prior lines → else branch).
    # METHODS comes after content lines, so it triggers a new section with title "METHODS".
    intro_content = "First section content with many words. " * 10
    methods_content = "Second section content describing methods. " * 10
    text = f"INTRODUCTION\n\n{intro_content}\n\nMETHODS\n\n{methods_content}"
    pages = make_pages(text)
    chunks = chunk_pages(pages, DOC_ID, DOC_NAME)
    sections = {c.get("section_title") for c in chunks}
    assert "METHODS" in sections


def test_short_document_still_produces_chunk():
    text = "Short document."
    pages = make_pages(text)
    chunks = chunk_pages(pages, DOC_ID, DOC_NAME)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short document."


def test_multipage_page_numbers_recorded():
    pages = [(1, "Content on page one."), (2, "Content on page two."), (3, "Content on page three.")]
    chunks = chunk_pages(pages, DOC_ID, DOC_NAME)
    assert len(chunks) >= 1
    assert chunks[0]["page_start"] is not None
    assert chunks[0]["page_end"] is not None


def test_chunk_index_monotonic():
    text = "\n\n".join(["Para %d: " % i + "text " * 100 for i in range(15)])
    pages = make_pages(text)
    chunks = chunk_pages(pages, DOC_ID, DOC_NAME)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
