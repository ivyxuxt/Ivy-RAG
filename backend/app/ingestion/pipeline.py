"""
Ingestion pipeline orchestrator.
POST /ingest: receive PDFs → parse → chunk → embed → persist.
GET /documents and DELETE /documents/{doc_id} live in llm.py (shared router).
"""
import os
import uuid
import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.parser import extract_pages
from app.models import IngestedDoc, IngestResponse
from app.retrieval.embedder import embed_texts
from app.storage.store import store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])
limiter = Limiter(key_func=get_remote_address)

_MAX_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files for ingestion.

    Validates:
    - Extension is .pdf
    - MIME type is application/pdf
    - File size ≤ MAX_UPLOAD_MB
    - File count ≤ MAX_UPLOAD_FILES
    """
    if len(files) > settings.MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum is {settings.MAX_UPLOAD_FILES} per request.",
        )

    ingested: List[IngestedDoc] = []
    errors: List[str] = []

    for upload in files:
        filename = upload.filename or "unnamed.pdf"
        try:
            _validate_file(upload, filename)
            content = await upload.read()

            if len(content) > _MAX_BYTES:
                errors.append(f"{filename}: exceeds {settings.MAX_UPLOAD_MB} MB limit")
                continue

            doc_id = _save_and_process(content, filename)

            # Count chunks for this doc
            n_chunks = sum(1 for c in store.chunks if c["doc_id"] == doc_id)
            n_pages = len(set(c["page_start"] for c in store.chunks if c["doc_id"] == doc_id and c["page_start"]))

            ingested.append(IngestedDoc(
                doc_id=doc_id,
                name=filename,
                n_pages=n_pages,
                n_chunks=n_chunks,
            ))

        except HTTPException as e:
            errors.append(f"{filename}: {e.detail}")
        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {e}", exc_info=True)
            errors.append(f"{filename}: internal error — {str(e)[:100]}")

    return IngestResponse(ingested=ingested, errors=errors)


def _validate_file(upload: UploadFile, filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail=f"Only .pdf files are accepted (got {ext})")

    content_type = upload.content_type or ""
    if content_type and "pdf" not in content_type.lower():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {content_type}. Expected application/pdf.",
        )


def _save_and_process(content: bytes, original_filename: str) -> str:
    """Save PDF to disk, extract, chunk, embed, and persist."""
    doc_id = str(uuid.uuid4())
    os.makedirs(settings.raw_dir, exist_ok=True)
    raw_path = os.path.join(settings.raw_dir, f"{doc_id}.pdf")

    with open(raw_path, "wb") as f:
        f.write(content)

    # Parse
    pages = extract_pages(raw_path)

    # Chunk
    chunks = chunk_pages(pages, doc_id, original_filename)

    if not chunks:
        raise ValueError("No text could be extracted from this PDF")

    # Embed
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    # Persist
    store.add_chunks(chunks, vectors)

    logger.info(f"Ingested {original_filename} → {len(chunks)} chunks (doc_id={doc_id})")
    return doc_id
