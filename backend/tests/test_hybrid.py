"""Tests for RRF hybrid fusion."""
import pytest
from app.retrieval.hybrid import rrf_merge


def _make_chunks(n):
    return [{"chunk_id": f"doc#c{i}", "text": f"text {i}"} for i in range(n)]


def test_rrf_merge_basic():
    chunks = _make_chunks(5)
    dense = [(0, 0.9), (1, 0.8), (2, 0.7)]
    bm25 = [("doc#c1", 5.0), ("doc#c0", 4.0), ("doc#c3", 3.0)]
    merged = rrf_merge(dense, bm25, chunks, top_n=5)
    assert len(merged) <= 5
    row_indices = [r for r, _ in merged]
    assert len(row_indices) == len(set(row_indices)), "No duplicate row indices"


def test_rrf_merge_consensus_ranks_higher():
    """A chunk that appears in both lists should outscore one appearing in only one."""
    chunks = _make_chunks(4)
    # chunk 0: dense rank 1, bm25 rank 1 — strong consensus
    # chunk 2: dense rank 2, not in bm25
    dense = [(0, 0.95), (2, 0.80)]
    bm25 = [("doc#c0", 8.0), ("doc#c3", 3.0)]
    merged = rrf_merge(dense, bm25, chunks, top_n=4)
    scores = {row: score for row, score in merged}
    assert scores.get(0, 0) > scores.get(2, 0)


def test_rrf_merge_respects_top_n():
    chunks = _make_chunks(10)
    dense = [(i, 1.0 / (i + 1)) for i in range(10)]
    bm25 = [(f"doc#c{i}", 10.0 / (i + 1)) for i in range(10)]
    merged = rrf_merge(dense, bm25, chunks, top_n=5)
    assert len(merged) == 5


def test_rrf_merge_scores_positive():
    chunks = _make_chunks(3)
    dense = [(0, 0.9), (1, 0.7)]
    bm25 = [("doc#c0", 5.0)]
    merged = rrf_merge(dense, bm25, chunks, top_n=3)
    assert all(score > 0 for _, score in merged)


def test_rrf_merge_sorted_descending():
    chunks = _make_chunks(5)
    dense = [(i, 1.0 / (i + 1)) for i in range(5)]
    bm25 = [(f"doc#c{i}", 5.0 / (i + 1)) for i in range(5)]
    merged = rrf_merge(dense, bm25, chunks, top_n=5)
    scores = [s for _, s in merged]
    assert scores == sorted(scores, reverse=True)


def test_rrf_merge_unknown_bm25_chunk_id_skipped():
    """BM25 chunk IDs not in the chunk list should be silently ignored."""
    chunks = _make_chunks(2)
    dense = [(0, 0.9)]
    bm25 = [("ghost#c99", 10.0)]  # doesn't exist in chunks
    merged = rrf_merge(dense, bm25, chunks, top_n=5)
    row_indices = [r for r, _ in merged]
    assert 99 not in row_indices


def test_rrf_merge_empty_dense():
    chunks = _make_chunks(3)
    bm25 = [("doc#c0", 5.0), ("doc#c1", 3.0)]
    merged = rrf_merge([], bm25, chunks, top_n=5)
    assert len(merged) >= 1


def test_rrf_merge_empty_bm25():
    chunks = _make_chunks(3)
    dense = [(0, 0.9), (1, 0.7)]
    merged = rrf_merge(dense, [], chunks, top_n=5)
    assert len(merged) >= 1
