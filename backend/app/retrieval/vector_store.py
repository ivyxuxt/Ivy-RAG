"""
NumPy-based dense vector search.
- Cosine similarity = dot product of L2-normalized vectors
- Uses argpartition for O(N) top-k instead of full sort
- All vectors are pre-normalized at storage time (embedder.py)
"""
from typing import List, Tuple

import numpy as np

from app.config import settings


def topk_dense(
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    k: int = None,
) -> List[Tuple[int, float]]:
    """
    Find the top-k most similar chunks by cosine similarity.

    Args:
        embeddings: (N, D) L2-normalized float32 matrix from the store
        query_vec:  (D,) L2-normalized query embedding
        k:          Number of results (defaults to DENSE_TOPK from config)

    Returns:
        List of (row_index, score) sorted descending by score.
    """
    if k is None:
        k = settings.DENSE_TOPK

    if embeddings is None or len(embeddings) == 0:
        return []

    n = len(embeddings)
    k = min(k, n)

    # Cosine similarity = dot product (both are L2-normalized)
    sims = embeddings @ query_vec  # shape (N,)

    # argpartition requires kth < array size; when k == n just argsort all
    if k == n:
        top_indices = np.argsort(-sims)
    else:
        top_indices = np.argpartition(-sims, k)[:k]

    results = [(int(i), float(sims[i])) for i in top_indices]
    results.sort(key=lambda x: -x[1])
    return results
