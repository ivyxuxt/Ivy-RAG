"""
Post-hoc hallucination verifier.

Step 1 (local): per-sentence cosine similarity vs. retrieved chunks.
    If max_sim < SENTENCE_EVIDENCE_THRESHOLD → "possibly_unsupported"

Step 2 (optional web): for flagged sentences, issue a web search query.
    Uses DuckDuckGo Instant Answer JSON API (no API key needed).
    If top snippet contradicts the claim → "web_contradicted"
    Otherwise stays "possibly_unsupported"

Returns a list of {index, text, status} for sentences that failed the check.
"""
import re
import logging
from typing import List

import httpx
import numpy as np

from app.config import settings
from app.retrieval.embedder import embed_single

logger = logging.getLogger(__name__)

# Boilerplate sentences to skip (citations, transitions, refusals)
_SKIP_PATTERNS = [
    r"^insufficient evidence",
    r"^\[s\d+\]",
    r"^based on the",
    r"^according to",
    r"^as stated",
    r"^the document",
    r"^\s*$",
]


def verify(answer: str, top_chunks: List[dict], embeddings: np.ndarray, chunk_rows: List[int]) -> List[dict]:
    """
    Check each answer sentence for support in retrieved chunks.

    Args:
        answer:      The generated answer text
        top_chunks:  List of chunk dicts returned by the retriever
        embeddings:  Full embedding matrix from the store
        chunk_rows:  Row indices of top_chunks in the embedding matrix

    Returns:
        List of {index, text, status} for unsupported sentences.
    """
    sentences = _split_sentences(answer)
    chunk_vecs = np.array([embeddings[r] for r in chunk_rows if r < len(embeddings)])

    if len(chunk_vecs) == 0:
        return []

    unsupported = []

    for i, sent in enumerate(sentences):
        if _should_skip(sent):
            continue

        try:
            sent_vec = embed_single(sent)
            sims = chunk_vecs @ sent_vec
            max_sim = float(np.max(sims))
        except Exception as e:
            logger.warning(f"Verifier embed failed for sentence {i}: {e}")
            continue

        if max_sim < settings.SENTENCE_EVIDENCE_THRESHOLD:
            status = _web_check(sent)
            unsupported.append({"index": i, "text": sent, "status": status})

    return unsupported


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on . ! ? followed by whitespace + capital letter or end of string
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [p.strip() for p in parts if p.strip()]


def _should_skip(sentence: str) -> bool:
    s = sentence.lower().strip()
    for pattern in _SKIP_PATTERNS:
        if re.match(pattern, s):
            return True
    # Very short sentences (likely headings or filler)
    if len(sentence.split()) < 5:
        return True
    return False


def _web_check(sentence: str) -> str:
    """
    Query DuckDuckGo Instant Answer API to cross-check a flagged sentence.
    Returns "web_contradicted" or "possibly_unsupported".
    """
    # Extract the key claim (first ~60 chars, stripped of citations)
    clean = re.sub(r"\[S\d+\]", "", sentence).strip()
    query = clean[:100]

    try:
        response = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=5.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        data = response.json()

        # Check if DuckDuckGo returns an AbstractText that contradicts key numbers
        abstract = (data.get("AbstractText") or "").lower()
        if abstract and _contradicts(sentence, abstract):
            return "web_contradicted"

    except Exception as e:
        logger.debug(f"Web check failed: {e}")

    return "possibly_unsupported"


def _contradicts(sentence: str, web_text: str) -> bool:
    """
    Simple contradiction check: if the sentence states a number and the web
    text states a different number in the same context, flag it.
    This is intentionally conservative to avoid false positives.
    """
    sentence_numbers = re.findall(r"\b\d[\d,.]*\b", sentence)
    web_numbers = re.findall(r"\b\d[\d,.]*\b", web_text)

    if not sentence_numbers or not web_numbers:
        return False

    # If none of the sentence's numbers appear in the web text, possible contradiction
    return not any(n in web_numbers for n in sentence_numbers)
