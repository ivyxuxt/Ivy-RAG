"""
Post-hoc hallucination verifier.

Step 1 (local): per-sentence cosine similarity vs. retrieved chunks.
    If max_sim < SENTENCE_EVIDENCE_THRESHOLD → sentence is suspicious.

Step 2 (LLM): for each suspicious sentence, call Mistral with the
    best-matching source chunk and ask whether it SUPPORTS, CONTRADICTS,
    or does NOT MENTION the sentence.

    This is more accurate than regex/number matching (the previous approach),
    catching name swaps, inverted causality, and misattributed claims — not
    just misquoted numbers.

    Returns:
        "web_contradicted"   — Mistral says the source contradicts it  (red in UI)
        "possibly_unsupported" — Mistral says the source doesn't mention it (amber in UI)
        (sentence removed from list) — Mistral says it is supported (cosine false alarm)

    Cost: one Mistral call per flagged sentence, gated behind the cosine filter.
"""
import re
import time
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

_MISTRAL_VERIFY_PROMPT = (
    "You are a fact-checking assistant. Given a source passage and a sentence "
    "from an AI-generated answer, determine whether the source SUPPORTS, CONTRADICTS, "
    "or does NOT MENTION the sentence.\n\n"
    "Source:\n{source}\n\n"
    "Sentence:\n{sentence}\n\n"
    "Reply with exactly one word: SUPPORTED, CONTRADICTED, or NOT_MENTIONED."
)


def verify(answer: str, top_chunks: List[dict], embeddings: np.ndarray, chunk_rows: List[int]) -> List[dict]:
    """
    Check each answer sentence for support in retrieved chunks.

    Args:
        answer:      The generated answer text
        top_chunks:  List of chunk dicts returned by the retriever
        embeddings:  Full embedding matrix from the store
        chunk_rows:  Row indices of top_chunks in the embedding matrix

    Returns:
        List of {index, text, status} for unsupported or contradicted sentences.
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
            # Find the best-matching chunk to use as the source for LLM verification
            best_chunk_idx = int(np.argmax(sims))
            best_chunk_text = top_chunks[best_chunk_idx]["text"] if best_chunk_idx < len(top_chunks) else ""

            status = _mistral_check(sent, best_chunk_text)
            if status is not None:
                unsupported.append({"index": i, "text": sent, "status": status})

    return unsupported


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [p.strip() for p in parts if p.strip()]


def _should_skip(sentence: str) -> bool:
    s = sentence.lower().strip()
    for pattern in _SKIP_PATTERNS:
        if re.match(pattern, s):
            return True
    if len(sentence.split()) < 5:
        return True
    return False


def _mistral_check(sentence: str, source_chunk: str):
    """
    Ask Mistral whether the source chunk supports, contradicts, or doesn't
    mention the flagged sentence.

    Returns:
        "web_contradicted"     if Mistral says CONTRADICTED
        "possibly_unsupported" if Mistral says NOT_MENTIONED
        None                   if Mistral says SUPPORTED (cosine was a false alarm)
    """
    if not source_chunk:
        return "possibly_unsupported"

    prompt = _MISTRAL_VERIFY_PROMPT.format(
        source=source_chunk[:1500],  # cap to stay within context
        sentence=sentence,
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
                    "max_tokens": 10,
                },
                timeout=15.0,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == 0:
                    time.sleep(2.0)
                continue
            response.raise_for_status()
            verdict = response.json()["choices"][0]["message"]["content"].strip().upper()

            if "CONTRADICT" in verdict:
                return "web_contradicted"
            if "NOT_MENTIONED" in verdict or "NOT MENTIONED" in verdict:
                return "possibly_unsupported"
            # SUPPORTED or anything else → cosine false alarm, don't flag
            return None

        except Exception as e:
            logger.debug(f"Mistral verifier call failed (attempt {attempt}): {e}")
            if attempt == 0:
                time.sleep(2.0)

    # If both attempts fail, fall back to amber (conservative)
    return "possibly_unsupported"
