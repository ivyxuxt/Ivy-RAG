from pydantic import BaseModel
from typing import List, Optional


# ── Ingestion ──────────────────────────────────────────────────────────────────

class IngestedDoc(BaseModel):
    doc_id: str
    name: str
    n_pages: int
    n_chunks: int


class IngestResponse(BaseModel):
    ingested: List[IngestedDoc]
    errors: List[str]


# ── Documents list ─────────────────────────────────────────────────────────────

class DocInfo(BaseModel):
    doc_id: str
    name: str
    n_chunks: int


class DocumentsResponse(BaseModel):
    documents: List[DocInfo]


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Citation(BaseModel):
    chunk_id: str
    doc_name: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: Optional[str] = None
    score: float
    snippet: str


class UnsupportedSentence(BaseModel):
    index: int
    text: str
    status: str  # "possibly_unsupported" | "web_contradicted"


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    intent: str
    retrieved_chunks: int
    sufficient_evidence: bool
    unsupported_sentences: List[UnsupportedSentence]


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    ok: bool


class ReadyResponse(BaseModel):
    ok: bool
    chunks_loaded: int
