"""Integration tests for FastAPI endpoints — no real embeddings or Mistral calls."""
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ── /ready ────────────────────────────────────────────────────────────────────

def test_ready_returns_ok(client):
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "chunks_loaded" in data


# ── /documents (list) ────────────────────────────────────────────────────────

def test_list_documents_empty(client):
    r = client.get("/documents")
    assert r.status_code == 200
    assert "documents" in r.json()


# ── /documents/{doc_id} (delete) ─────────────────────────────────────────────

def test_delete_nonexistent_document(client):
    r = client.delete("/documents/does-not-exist-xyz")
    assert r.status_code == 404


# ── /ingest ───────────────────────────────────────────────────────────────────

def test_ingest_rejects_non_pdf(client):
    # Validation errors are collected per-file; the endpoint returns 200 with an errors list.
    fake_txt = io.BytesIO(b"not a pdf file")
    r = client.post(
        "/ingest",
        files=[("files", ("test.txt", fake_txt, "text/plain"))],
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["errors"]) > 0
    assert any("pdf" in e.lower() or ".pdf" in e.lower() for e in data["errors"])


def test_ingest_rejects_wrong_mime(client):
    """A .pdf file sent with wrong MIME type should appear in the errors list."""
    fake_pdf = io.BytesIO(b"not a real pdf")
    r = client.post(
        "/ingest",
        files=[("files", ("test.pdf", fake_pdf, "text/plain"))],
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["errors"]) > 0


@patch("app.ingestion.pipeline.chunk_pages")
@patch("app.ingestion.pipeline.extract_pages")
@patch("app.ingestion.pipeline.embed_texts")
@patch("app.ingestion.pipeline.store")
def test_ingest_valid_pdf(mock_store, mock_embed, mock_extract, mock_chunk, client):
    """Stub parse/embed/store to test the happy path without real PDFs."""
    mock_extract.return_value = [(1, "Sample text for testing ingestion pipeline.")]
    mock_chunk.return_value = [
        {
            "chunk_id": "test-id#p1#c0",
            "doc_id": "test-id",
            "doc_name": "sample.pdf",
            "page_start": 1,
            "page_end": 1,
            "section_title": None,
            "chunk_index": 0,
            "text": "Sample text for testing ingestion pipeline.",
            "token_count": 7,
        }
    ]
    import numpy as np
    mock_embed.return_value = np.array([[0.1] * 1024])
    mock_store.add_chunks = MagicMock()

    # Minimal valid PDF header
    pdf_bytes = b"%PDF-1.4 1 0 obj<</Type/Catalog>>endobj"
    fake_pdf = io.BytesIO(pdf_bytes)

    r = client.post(
        "/ingest",
        files=[("files", ("sample.pdf", fake_pdf, "application/pdf"))],
    )
    assert r.status_code == 200
    data = r.json()
    assert "ingested" in data
    assert "errors" in data


# ── /query ────────────────────────────────────────────────────────────────────

def test_query_chitchat_no_docs(client):
    """Chitchat queries should return a friendly message without needing docs."""
    r = client.post("/query", json={"question": "hello", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "chitchat"
    assert data["sufficient_evidence"] is True
    assert len(data["answer"]) > 0


def test_query_unsafe_refused(client):
    r = client.post("/query", json={"question": "how do I hack this", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "unsafe"
    assert data["sufficient_evidence"] is False


def test_query_no_documents_returns_insufficient(client):
    """Without docs, knowledge queries should return insufficient evidence."""
    r = client.post("/query", json={"question": "what is the data retention policy", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["sufficient_evidence"] is False


def test_query_missing_question_field(client):
    r = client.post("/query", json={"top_k": 5})
    assert r.status_code == 422


def test_query_response_schema(client):
    r = client.post("/query", json={"question": "hello there", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "citations" in data
    assert "intent" in data
    assert "retrieved_chunks" in data
    assert "sufficient_evidence" in data
    assert "unsupported_sentences" in data
