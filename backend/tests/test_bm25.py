"""Tests for custom BM25 implementation."""
import math
import pytest
from app.retrieval.bm25 import tokenize, build_index, search


# ── tokenize ──────────────────────────────────────────────────────────────────

def test_tokenize_lowercases():
    tokens = tokenize("Hello World")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_removes_stopwords():
    tokens = tokenize("the cat sat on the mat")
    assert "the" not in tokens
    assert "on" not in tokens


def test_tokenize_splits_on_punctuation():
    tokens = tokenize("word1,word2.word3")
    assert "word1" in tokens
    assert "word2" in tokens
    assert "word3" in tokens


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("the the the") == []


def test_tokenize_removes_single_chars():
    tokens = tokenize("a b c hello")
    assert "a" not in tokens
    assert "b" not in tokens


# ── build_index ───────────────────────────────────────────────────────────────

def _make_chunks(texts):
    return [{"chunk_id": f"doc#c{i}", "text": t} for i, t in enumerate(texts)]


def test_build_index_basic_structure():
    chunks = _make_chunks(["hello world test", "another test document"])
    index = build_index(chunks)
    assert "inverted_index" in index
    assert "doc_lengths" in index
    assert "avg_doc_len" in index
    assert "df" in index
    assert "n_docs" in index
    assert index["n_docs"] == 2


def test_build_index_term_appears_in_correct_docs():
    # Use a single-token word (no underscores) that won't appear in doc 1
    chunks = _make_chunks(["xyzquux present here", "completely different content"])
    index = build_index(chunks)
    inv = index["inverted_index"]
    assert "xyzquux" in inv
    assert "doc#c0" in inv["xyzquux"]
    assert "doc#c1" not in inv.get("xyzquux", {})


def test_build_index_df_counts():
    chunks = _make_chunks(["shared term present", "shared term here also", "no match"])
    index = build_index(chunks)
    assert index["df"]["shared"] == 2
    assert index["df"]["term"] == 2


def test_build_index_empty():
    index = build_index([])
    assert index["n_docs"] == 0
    assert index["avg_doc_len"] == 1.0


# ── search ────────────────────────────────────────────────────────────────────

def test_search_returns_relevant_doc_first():
    chunks = _make_chunks([
        "machine learning neural network deep learning",
        "cooking recipes baking bread flour yeast",
    ])
    index = build_index(chunks)
    results = search(index, "neural network deep learning", k=2)
    assert len(results) > 0
    top_chunk_id, top_score = results[0]
    assert top_chunk_id == "doc#c0"


def test_search_score_positive():
    chunks = _make_chunks(["retrieval augmented generation language model"])
    index = build_index(chunks)
    results = search(index, "retrieval generation", k=1)
    assert len(results) == 1
    assert results[0][1] > 0


def test_search_no_match_returns_empty():
    chunks = _make_chunks(["apple orange banana fruit"])
    index = build_index(chunks)
    results = search(index, "quantum physics relativity", k=5)
    assert results == []


def test_search_empty_query():
    chunks = _make_chunks(["some document text here"])
    index = build_index(chunks)
    results = search(index, "the and or", k=5)  # all stopwords
    assert results == []


def test_search_respects_k():
    chunks = _make_chunks([f"document {i} contains keyword test topic" for i in range(10)])
    index = build_index(chunks)
    results = search(index, "keyword test", k=3)
    assert len(results) <= 3


def test_search_higher_tf_scores_higher():
    chunks = _make_chunks([
        "keyword keyword keyword in this document",
        "keyword appears once here",
    ])
    index = build_index(chunks)
    results = search(index, "keyword", k=2)
    scores = {cid: score for cid, score in results}
    assert scores["doc#c0"] > scores["doc#c1"]
