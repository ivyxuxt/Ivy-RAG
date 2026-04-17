"""
In-memory index: loads chunks.jsonl + embeddings.npy + bm25.pkl on startup.
All retrieval modules read from this singleton.
"""
import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

from app.config import settings


class IndexStore:
    def __init__(self):
        self.chunks: List[dict] = []                  # row-aligned with embeddings
        self.embeddings: Optional[np.ndarray] = None  # shape (N, D), L2-normalized
        self.bm25_index: Optional[dict] = None        # built by bm25.py
        self._doc_rows: Dict[str, List[int]] = {}     # doc_id → list of row indices

    # ── Persistence ────────────────────────────────────────────────────────────

    def load(self):
        """Load all three artifacts from disk if they exist."""
        if os.path.exists(settings.chunks_path):
            with open(settings.chunks_path) as f:
                self.chunks = [json.loads(line) for line in f if line.strip()]

        if os.path.exists(settings.embeddings_path) and self.chunks:
            # Memory-map the embeddings file so the OS pages in only accessed rows,
            # keeping RAM usage proportional to what retrieval actually touches
            # rather than loading the full matrix upfront.
            self.embeddings = np.load(settings.embeddings_path, mmap_mode="r")

        if os.path.exists(settings.bm25_path):
            with open(settings.bm25_path, "rb") as f:
                self.bm25_index = pickle.load(f)

        self._rebuild_doc_map()

    def _save_chunks(self):
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        with open(settings.chunks_path, "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk) + "\n")

    def _save_embeddings(self):
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        np.save(settings.embeddings_path, self.embeddings)

    def _save_bm25(self):
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        with open(settings.bm25_path, "wb") as f:
            pickle.dump(self.bm25_index, f)

    # ── Ingest ─────────────────────────────────────────────────────────────────

    def add_chunks(self, new_chunks: List[dict], new_embeddings: np.ndarray):
        """Append new chunks + embeddings, update BM25 incrementally, persist everything."""
        start_idx = len(self.chunks)

        self.chunks.extend(new_chunks)

        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack(
                [self.embeddings, new_embeddings.astype(np.float32)]
            )

        self._rebuild_doc_map()
        self._update_bm25(new_chunks)
        self._save_chunks()
        self._save_embeddings()
        self._save_bm25()

        return start_idx

    def remove_doc(self, doc_id: str) -> bool:
        """Remove all chunks and embeddings for a document, repack."""
        if doc_id not in self._doc_rows:
            return False

        removed_chunks = [c for c in self.chunks if c["doc_id"] == doc_id]
        keep = [i for i in range(len(self.chunks)) if self.chunks[i]["doc_id"] != doc_id]
        self.chunks = [self.chunks[i] for i in keep]

        if self.embeddings is not None and len(self.embeddings) > 0:
            if keep:
                self.embeddings = self.embeddings[keep]
            else:
                self.embeddings = np.empty((0, self.embeddings.shape[1]), dtype=np.float32)

        self._rebuild_doc_map()
        self._remove_from_bm25(removed_chunks)
        self._save_chunks()
        self._save_embeddings()
        self._save_bm25()
        return True

    # ── Documents ──────────────────────────────────────────────────────────────

    def list_docs(self) -> List[dict]:
        seen = {}
        for chunk in self.chunks:
            doc_id = chunk["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {"doc_id": doc_id, "name": chunk["doc_name"], "n_chunks": 0}
            seen[doc_id]["n_chunks"] += 1
        return list(seen.values())

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _rebuild_doc_map(self):
        self._doc_rows = {}
        for i, chunk in enumerate(self.chunks):
            doc_id = chunk["doc_id"]
            self._doc_rows.setdefault(doc_id, []).append(i)

    def _rebuild_bm25(self):
        """Full rebuild — used only on initial load."""
        from app.retrieval.bm25 import build_index
        self.bm25_index = build_index(self.chunks)

    def _update_bm25(self, new_chunks: List[dict]):
        """Incremental update — O(new_docs) instead of O(all_docs)."""
        from app.retrieval.bm25 import update_index
        self.bm25_index = update_index(self.bm25_index or {}, new_chunks)

    def _remove_from_bm25(self, removed_chunks: List[dict]):
        """Remove deleted chunks from index without full rebuild."""
        from app.retrieval.bm25 import remove_from_index
        if self.bm25_index:
            self.bm25_index = remove_from_index(self.bm25_index, removed_chunks)


store = IndexStore()
