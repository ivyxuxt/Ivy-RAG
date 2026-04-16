"""
Custom BM25 implementation — no external library.

Okapi BM25 formula:
    score(q, d) = Σ IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))

Where:
    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    tf(t,d) = term frequency of t in document d
    |d|     = document length (tokens)
    avgdl   = average document length across corpus
    k1, b   = tuning constants (1.5, 0.75)

Design note:
    BM25 is the workhorse of keyword retrieval. It captures exact terminology,
    identifiers, names, and rare terms that embedding models often miss.
    We implement it from scratch to satisfy the "no external search library" constraint.
"""
import math
import re
from typing import Dict, List, Tuple

from app.config import settings

# Basic English stopwords — small enough to maintain inline, effective enough for BM25
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "i", "you", "he", "she", "we", "they",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "as",
    "if", "than", "then", "when", "where", "which", "who", "whom", "what",
    "how", "why", "all", "any", "each", "few", "more", "most", "other",
    "some", "such", "about", "up", "out", "into", "through", "during",
    "before", "after", "above", "below", "between", "while",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def build_index(chunks: List[dict]) -> dict:
    """
    Build a BM25 index from a list of chunk dicts.

    Returns a dict with keys:
        inverted_index: {term: {chunk_id: tf}}
        doc_lengths:    {chunk_id: int}
        avg_doc_len:    float
        df:             {term: int}  — document frequency
        n_docs:         int
        chunk_ids:      [chunk_id, ...]  — ordered list for score()
    """
    inverted: Dict[str, Dict[str, int]] = {}
    doc_lengths: Dict[str, int] = {}

    for chunk in chunks:
        cid = chunk["chunk_id"]
        tokens = tokenize(chunk["text"])
        doc_lengths[cid] = len(tokens)

        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        for term, count in tf.items():
            if term not in inverted:
                inverted[term] = {}
            inverted[term][cid] = count

    n_docs = len(chunks)
    avg_doc_len = sum(doc_lengths.values()) / n_docs if n_docs > 0 else 1.0

    df = {term: len(postings) for term, postings in inverted.items()}

    return {
        "inverted_index": inverted,
        "doc_lengths": doc_lengths,
        "avg_doc_len": avg_doc_len,
        "df": df,
        "n_docs": n_docs,
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }


def search(
    index: dict,
    query: str,
    k: int = None,
) -> List[Tuple[str, float]]:
    """
    Score all chunks against the query using BM25.

    Returns:
        List of (chunk_id, score) sorted descending, top k.
    """
    if k is None:
        k = settings.BM25_TOPK

    if not index or not index.get("chunk_ids"):
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    inverted = index["inverted_index"]
    doc_lengths = index["doc_lengths"]
    avg_doc_len = index["avg_doc_len"]
    df = index["df"]
    n_docs = index["n_docs"]
    k1 = settings.BM25_K1
    b = settings.BM25_B

    scores: Dict[str, float] = {}

    for term in query_terms:
        if term not in inverted:
            continue

        # IDF with smoothing (Robertson-Walker variant)
        df_t = df.get(term, 0)
        idf = math.log((n_docs - df_t + 0.5) / (df_t + 0.5) + 1)

        for chunk_id, tf in inverted[term].items():
            doc_len = doc_lengths.get(chunk_id, avg_doc_len)
            norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * norm_tf

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:k]
