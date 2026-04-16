"""
Query transformation before retrieval.

Dense search — HyDE-lite:
    Ask Mistral to write a one-sentence hypothetical answer.
    Embed that sentence instead of the raw question.
    Rationale: when the question uses different vocabulary than the document
    (e.g. "how do I reduce latency?" vs document text: "performance optimization
    techniques"), embedding the hypothetical answer brings the query vector
    closer to the relevant chunks.

BM25 keyword search:
    Use the original question. HyDE text adds noise for exact-match keyword search.
    Optionally: strip WH-words and filler phrases to focus on content terms.
"""
import re
import time
from dataclasses import dataclass

import httpx

from app.config import settings

_WH_FILLER = re.compile(
    r"^(what|how|why|when|where|who|which|whose|whom|can you|could you|"
    r"please|tell me|give me|show me|i want to know|i need to know)\s+",
    re.IGNORECASE,
)

_FILLER_PHRASES = [
    "according to the document", "according to the file", "based on the document",
    "based on the uploaded", "in the uploaded files", "from the pdf",
    "the document says", "as stated in", "per the policy",
]


@dataclass
class TransformedQuery:
    dense_query: str    # text to embed for semantic search
    keyword_query: str  # text to tokenize for BM25


def transform(question: str) -> TransformedQuery:
    """
    Produce a dense query (HyDE) and a keyword query (cleaned original).
    """
    keyword_query = _clean_for_keyword(question)
    dense_query = _hyde_lite(question)

    return TransformedQuery(
        dense_query=dense_query,
        keyword_query=keyword_query,
    )


def _clean_for_keyword(question: str) -> str:
    """Remove filler phrases and WH-words, preserve content terms."""
    q = question
    for phrase in _FILLER_PHRASES:
        q = q.replace(phrase, " ")
    q = _WH_FILLER.sub("", q.strip())
    return q.strip() or question


def _hyde_lite(question: str) -> str:
    """
    Ask Mistral for a one-sentence hypothetical answer, return that text.
    Falls back to the original question on error.
    """
    prompt = (
        f"Write exactly one sentence that would directly answer the following question. "
        f"Do not introduce yourself. Do not explain. Just write the answer sentence.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    for attempt in range(2):
        try:
            response = httpx.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.MISTRAL_CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 80,
                },
                timeout=15.0,
            )
            response.raise_for_status()
            hypothetical = response.json()["choices"][0]["message"]["content"].strip()
            if hypothetical:
                return hypothetical
        except Exception:
            if attempt == 0:
                time.sleep(2)

    return question  # fallback: use original question
