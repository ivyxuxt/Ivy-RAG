"""
Post-fusion reranking using MMR + metadata signals + neighbor expansion.

MMR (Maximal Marginal Relevance):
    Selects chunks that are relevant to the query but diverse from each other.
    MMR(c) = λ · sim(c, query) − (1−λ) · max sim(c, already_selected)

    λ=0.7: favors relevance, still penalizes near-duplicate paragraphs.

Metadata signals layered on top of MMR:
    +0.10 if section heading contains query terms
    +0.05 if chunk contains exact numeric/entity match with query
    +0.05 if chunk's neighbor (prev/next) is also high-scoring

Neighbor expansion:
    For each final selected chunk, also include the prev/next chunk.
    Repairs answers that straddle chunk boundaries.
"""
import re
from typing import List, Tuple

import numpy as np

from app.config import settings
from app.retrieval.bm25 import tokenize


def rerank(
    candidates: List[Tuple[int, float]],   # (row_index, rrf_score)
    query_vec: np.ndarray,                  # L2-normalized query embedding
    embeddings: np.ndarray,                 # full embedding matrix
    chunks: list,                           # full chunk list
    query_text: str,
    top_k: int = None,
) -> List[dict]:
    """
    Rerank top-N RRF candidates using MMR + metadata signals.
    Apply neighbor expansion to final selection.

    Returns:
        List of chunk dicts (enriched with score), length up to top_k * 2
        (after neighbor expansion).
    """
    if top_k is None:
        top_k = settings.TOP_K

    if not candidates:
        return []

    lam = settings.MMR_LAMBDA
    query_terms = set(tokenize(query_text))

    # Build candidate pool with metadata scores
    pool = []
    candidate_set = {row_idx for row_idx, _ in candidates}

    for row_idx, rrf_score in candidates:
        chunk = chunks[row_idx]
        vec = embeddings[row_idx]
        cosine = float(np.dot(vec, query_vec))

        # Metadata bonus
        meta_bonus = 0.0
        if chunk.get("section_title"):
            heading_terms = set(tokenize(chunk["section_title"]))
            if query_terms & heading_terms:
                meta_bonus += 0.10
        if _has_entity_match(query_text, chunk["text"]):
            meta_bonus += 0.05

        # Neighbor support bonus: if adjacent chunk is also in the candidate pool
        idx = chunk.get("chunk_index", row_idx)
        for adj in (idx - 1, idx + 1):
            adj_row = _find_chunk_row(chunks, chunk["doc_id"], adj)
            if adj_row is not None and adj_row in candidate_set:
                meta_bonus += 0.05
                break

        pool.append({
            "row_idx": row_idx,
            "chunk": chunk,
            "vec": vec,
            "cosine": cosine,
            "meta_bonus": meta_bonus,
        })

    # MMR greedy selection
    selected_vecs: List[np.ndarray] = []
    selected_rows: List[int] = []

    for _ in range(min(top_k, len(pool))):
        best_score = -1e9
        best_item = None

        for item in pool:
            if item["row_idx"] in selected_rows:
                continue

            relevance = item["cosine"] + item["meta_bonus"]

            if selected_vecs:
                max_sim = max(float(np.dot(item["vec"], sv)) for sv in selected_vecs)
                mmr = lam * relevance - (1 - lam) * max_sim
            else:
                mmr = relevance

            if mmr > best_score:
                best_score = mmr
                best_item = item

        if best_item is None:
            break

        best_item["final_score"] = best_score
        selected_rows.append(best_item["row_idx"])
        selected_vecs.append(best_item["vec"])

    # Build result list in selection order
    row_to_item = {item["row_idx"]: item for item in pool}
    results = []
    for row_idx in selected_rows:
        item = row_to_item[row_idx]
        chunk = dict(item["chunk"])
        chunk["_score"] = item.get("final_score", item["cosine"])
        results.append(chunk)

    # Neighbor expansion: add prev/next chunks for each selected chunk
    results = _expand_neighbors(results, chunks, selected_rows)

    return results


def _expand_neighbors(
    selected: List[dict],
    all_chunks: list,
    selected_rows: List[int],
) -> List[dict]:
    """Add prev/next chunk for each selected chunk if not already present."""
    selected_ids = {c["chunk_id"] for c in selected}
    additions = []

    for chunk in selected:
        doc_id = chunk["doc_id"]
        idx = chunk.get("chunk_index", 0)

        for adj_idx in (idx - 1, idx + 1):
            adj_row = _find_chunk_row(all_chunks, doc_id, adj_idx)
            if adj_row is None:
                continue
            adj_chunk = all_chunks[adj_row]
            if adj_chunk["chunk_id"] not in selected_ids:
                neighbor = dict(adj_chunk)
                neighbor["_score"] = chunk["_score"] * 0.8  # slightly lower score
                neighbor["_is_neighbor"] = True
                additions.append(neighbor)
                selected_ids.add(adj_chunk["chunk_id"])

    return selected + additions


def _has_entity_match(query: str, chunk_text: str) -> bool:
    """Check if any numbers or capitalized entities in the query appear in the chunk."""
    numbers = re.findall(r"\b\d[\d,.]*\b", query)
    for num in numbers:
        if num in chunk_text:
            return True

    # Capitalized multi-word phrases (proper nouns / identifiers)
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", query)
    for ent in entities:
        if ent in chunk_text:
            return True

    return False


def _find_chunk_row(chunks: list, doc_id: str, chunk_index: int):
    """Find the row index of a chunk by doc_id and chunk_index."""
    for i, c in enumerate(chunks):
        if c["doc_id"] == doc_id and c.get("chunk_index") == chunk_index:
            return i
    return None
