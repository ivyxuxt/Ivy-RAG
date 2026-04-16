"""
Mistral chat completions client + POST /query endpoint.

Full query pipeline:
    1. policy check → refuse if PII/unsafe
    2. intent classification
    3. chitchat shortcut (no retrieval)
    4. query transformation (HyDE-lite for dense, cleaned for BM25)
    5. dense search + BM25 search
    6. RRF fusion → MMR rerank + neighbor expansion
    7. evidence gate (refuse if scores too low)
    8. build prompt by intent
    9. call Mistral chat
    10. post-hoc hallucination check
    11. return structured response
"""
import logging
import time
from typing import List

import httpx
from fastapi import APIRouter, Request

from app.config import settings
from app.generation.prompts import SYSTEM_PROMPT, build_context_block, get_user_prompt
from app.generation.verifier import verify
from app.models import Citation, QueryRequest, QueryResponse, UnsupportedSentence
from app.query import intent as intent_mod
from app.query import policy as policy_mod
from app.query import transform as transform_mod
from app.retrieval import bm25 as bm25_mod
from app.retrieval import vector_store as vs_mod
from app.retrieval.hybrid import rrf_merge
from app.retrieval.reranker import rerank
from app.retrieval.embedder import embed_single
from app.storage.store import store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


# ── /documents endpoints ────────────────────────────────────────────────────────

from app.models import DocumentsResponse, DocInfo
from fastapi import APIRouter, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)


@router.get("/documents", response_model=DocumentsResponse)
def list_documents():
    docs = store.list_docs()
    return DocumentsResponse(documents=[DocInfo(**d) for d in docs])


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    removed = store.remove_doc(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"deleted": doc_id}


# ── POST /query ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
def query(request: Request, body: QueryRequest):
    question = body.question.strip()
    top_k = body.top_k

    # 1. Policy check
    policy = policy_mod.check(question)
    if policy.refuse:
        return QueryResponse(
            answer=policy.reason,
            citations=[],
            intent="unsafe",
            retrieved_chunks=0,
            sufficient_evidence=False,
            unsupported_sentences=[],
        )

    # 2. Intent
    intent = intent_mod.classify(question)

    # 3. Chitchat shortcut
    if intent == "chitchat":
        answer = _chitchat_reply(question)
        return QueryResponse(
            answer=answer,
            citations=[],
            intent="chitchat",
            retrieved_chunks=0,
            sufficient_evidence=True,
            unsupported_sentences=[],
        )

    if intent == "unsafe":
        return QueryResponse(
            answer="I cannot help with that request.",
            citations=[],
            intent="unsafe",
            retrieved_chunks=0,
            sufficient_evidence=False,
            unsupported_sentences=[],
        )

    # 4. Query transformation
    transformed = transform_mod.transform(question)

    # 5. Retrieval
    if store.embeddings is None or len(store.chunks) == 0:
        return _no_docs_response(intent)

    query_vec = embed_single(transformed.dense_query)

    dense_results = vs_mod.topk_dense(store.embeddings, query_vec, k=settings.DENSE_TOPK)

    bm25_results = []
    if store.bm25_index:
        bm25_results = bm25_mod.search(store.bm25_index, transformed.keyword_query, k=settings.BM25_TOPK)

    # 6. Fusion + reranking
    fused = rrf_merge(dense_results, bm25_results, store.chunks, top_n=20)

    query_embed_for_rerank = embed_single(question)  # rerank against original question
    reranked_chunks = rerank(
        fused,
        query_embed_for_rerank,
        store.embeddings,
        store.chunks,
        question,
        top_k=top_k,
    )

    # 7. Evidence gate
    if not reranked_chunks:
        return _insufficient_evidence_response(intent)

    top_scores = [c.get("_score", 0.0) for c in reranked_chunks[:3]]
    top1_score = top_scores[0] if top_scores else 0.0
    avg3_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

    if top1_score < settings.SIMILARITY_THRESHOLD_TOP1 or avg3_score < settings.SIMILARITY_THRESHOLD_AVG3:
        return _insufficient_evidence_response(intent)

    # 8. Build prompt
    context = build_context_block(reranked_chunks)
    user_prompt = get_user_prompt(intent, question, context)

    # 9. Generate
    raw_answer = _call_mistral(user_prompt)

    # Prepend disclaimer if flagged by policy
    if policy.disclaimer:
        raw_answer = policy.disclaimer + "\n\n" + raw_answer

    # 10. Hallucination check
    chunk_row_map = {c["chunk_id"]: i for i, c in enumerate(store.chunks)}
    chunk_rows = [chunk_row_map[c["chunk_id"]] for c in reranked_chunks if c["chunk_id"] in chunk_row_map]
    unsupported = verify(raw_answer, reranked_chunks, store.embeddings, chunk_rows)

    # 11. Build citations
    citations = _build_citations(reranked_chunks)

    return QueryResponse(
        answer=raw_answer,
        citations=citations,
        intent=intent,
        retrieved_chunks=len(reranked_chunks),
        sufficient_evidence=True,
        unsupported_sentences=[UnsupportedSentence(**u) for u in unsupported],
    )


def _call_mistral(user_prompt: str) -> str:
    """Call Mistral chat completions with retry."""
    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.MISTRAL_CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": settings.LLM_TEMPERATURE,
                    "max_tokens": 1024,
                },
                timeout=30.0,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                wait = [2.0, 4.0, 8.0][min(attempt, 2)]
                logger.warning(f"Mistral chat HTTP {response.status_code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.TimeoutException:
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
    return "I was unable to generate a response at this time. Please try again."


def _build_citations(chunks: List[dict]) -> List[Citation]:
    citations = []
    for chunk in chunks:
        if chunk.get("_is_neighbor"):
            continue  # skip auto-expanded neighbors from citation list
        snippet = chunk["text"][:250].strip()
        if len(chunk["text"]) > 250:
            snippet += "…"
        citations.append(Citation(
            chunk_id=chunk["chunk_id"],
            doc_name=chunk["doc_name"],
            page_start=chunk.get("page_start"),
            page_end=chunk.get("page_end"),
            section_title=chunk.get("section_title"),
            score=round(chunk.get("_score", 0.0), 4),
            snippet=snippet,
        ))
    return citations


def _chitchat_reply(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["hello", "hi", "hey"]):
        return "Hello! I'm a document assistant. Upload some PDFs and ask me anything about them."
    if any(w in q for w in ["thanks", "thank you"]):
        return "You're welcome! Let me know if you have more questions about your documents."
    if "who are you" in q or "what can you do" in q:
        return (
            "I'm a RAG-powered document assistant. Upload PDFs using the panel on the left, "
            "then ask me questions about their content. I'll answer with citations and evidence."
        )
    return "I'm here to help. Upload a PDF and ask me anything about it!"


def _insufficient_evidence_response(intent: str) -> QueryResponse:
    return QueryResponse(
        answer="Insufficient evidence in the uploaded documents to answer this question.",
        citations=[],
        intent=intent,
        retrieved_chunks=0,
        sufficient_evidence=False,
        unsupported_sentences=[],
    )


def _no_docs_response(intent: str) -> QueryResponse:
    return QueryResponse(
        answer="No documents have been uploaded yet. Please upload PDFs using the /ingest endpoint first.",
        citations=[],
        intent=intent,
        retrieved_chunks=0,
        sufficient_evidence=False,
        unsupported_sentences=[],
    )
