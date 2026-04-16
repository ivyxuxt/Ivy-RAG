"""
Reciprocal Rank Fusion (RRF) for combining dense and BM25 results.

Why RRF over weighted score fusion:
    BM25 scores and cosine similarities live in different numerical ranges.
    Calibrating a weighted average requires per-corpus tuning.
    RRF uses rank position only — robust, parameter-light, and well-studied.

Formula:
    RRF(d) = Σ 1 / (k + rank(d, list_i))
    Standard k=60 smooths the rank differences.
"""
from typing import Dict, List, Tuple

from app.config import settings


def rrf_merge(
    dense_ranked: List[Tuple[int, float]],    # (row_index, score)
    bm25_ranked: List[Tuple[str, float]],     # (chunk_id, score)
    chunks: list,
    top_n: int = 20,
) -> List[Tuple[int, float]]:
    """
    Merge dense and BM25 results using RRF.

    Args:
        dense_ranked: Top-k from vector_store.topk_dense (row index, cosine score)
        bm25_ranked:  Top-k from bm25.search (chunk_id, BM25 score)
        chunks:       Full chunk list (to map chunk_id ↔ row index)
        top_n:        Number of merged results to return

    Returns:
        List of (row_index, rrf_score) sorted descending, length top_n.
    """
    k = settings.RRF_K

    # Build chunk_id → row_index map for BM25 results
    chunk_id_to_row = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    rrf_scores: Dict[int, float] = {}

    for rank, (row_idx, _) in enumerate(dense_ranked):
        rrf_scores[row_idx] = rrf_scores.get(row_idx, 0.0) + 1.0 / (k + rank)

    for rank, (chunk_id, _) in enumerate(bm25_ranked):
        row_idx = chunk_id_to_row.get(chunk_id)
        if row_idx is None:
            continue
        rrf_scores[row_idx] = rrf_scores.get(row_idx, 0.0) + 1.0 / (k + rank)

    merged = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return merged[:top_n]
