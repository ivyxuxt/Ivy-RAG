"""
Threshold calibration script for evidence gate.

Runs two query sets against the saved index:
  - Answerable: queries whose answer clearly exists in the uploaded docs
  - Unanswerable: queries about topics not present in the docs

Reports top-1 and avg-top-3 cosine similarity for each query,
then suggests threshold values at the gap between the two clusters.

Usage (from backend/):
    python scripts/calibrate_thresholds.py
"""
import json
import math
import os
import sys

import numpy as np
import httpx

# ── Load .env from project root ───────────────────────────────────────────────
from pathlib import Path
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_EMBED_MODEL = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

# ── Query sets — update these to match your uploaded documents ────────────────
#
# ANSWERABLE: rephrase something you know is in the docs.
# UNANSWERABLE: topics clearly absent from the docs.
#
ANSWERABLE_QUERIES = [
    "What is the median base salary for MBAn graduates?",
    "Which industries hired the most MBAn students?",
    "What percentage of students received a full-time job offer?",
    "What are the top companies that recruited MBAn graduates?",
    "What is the employment rate of the MBAn class?",
]

UNANSWERABLE_QUERIES = [
    "What is the GDP of Brazil in 2023?",
    "How does quantum computing affect cryptography?",
    "What are the immigration policies for H-1B visa holders?",
    "Explain the mechanism of mRNA vaccines.",
    "What is the market cap of Apple Inc.?",
]


def embed_texts(texts):
    response = httpx.post(
        "https://api.mistral.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
        json={"model": MISTRAL_EMBED_MODEL, "input": texts},
        timeout=30.0,
    )
    response.raise_for_status()
    vecs = np.array([d["embedding"] for d in response.json()["data"]], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


def load_index():
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        print(f"ERROR: Index not found at {DATA_DIR}. Upload at least one PDF first.")
        sys.exit(1)
    embeddings = np.load(str(EMBEDDINGS_PATH))
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks, embeddings shape {embeddings.shape}\n")
    return chunks, embeddings


def score_query(query_vec, embeddings):
    sims = embeddings @ query_vec
    sorted_sims = np.sort(sims)[::-1]
    top1 = float(sorted_sims[0])
    avg3 = float(sorted_sims[:3].mean())
    return top1, avg3


def run(label, queries, embeddings):
    print(f"── {label} ──")
    print(f"{'Query':<52} {'Top-1':>6}  {'Avg-3':>6}")
    print("-" * 68)
    vecs = embed_texts(queries)
    top1s, avg3s = [], []
    for q, vec in zip(queries, vecs):
        t1, a3 = score_query(vec, embeddings)
        top1s.append(t1)
        avg3s.append(a3)
        print(f"{q[:51]:<52} {t1:>6.3f}  {a3:>6.3f}")
    print(f"{'MEAN':<52} {sum(top1s)/len(top1s):>6.3f}  {sum(avg3s)/len(avg3s):>6.3f}")
    print()
    return top1s, avg3s


def main():
    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY not set.")
        sys.exit(1)

    chunks, embeddings = load_index()

    ans_t1, ans_a3 = run("ANSWERABLE queries", ANSWERABLE_QUERIES, embeddings)
    una_t1, una_a3 = run("UNANSWERABLE queries", UNANSWERABLE_QUERIES, embeddings)

    # Suggest thresholds at midpoint between cluster means
    gap_top1 = (sum(ans_t1) / len(ans_t1) + sum(una_t1) / len(una_t1)) / 2
    gap_avg3 = (sum(ans_a3) / len(ans_a3) + sum(una_a3) / len(una_a3)) / 2

    print("── Suggested thresholds ──")
    print(f"  SIMILARITY_THRESHOLD_TOP1 ≈ {gap_top1:.2f}  (midpoint between clusters)")
    print(f"  SIMILARITY_THRESHOLD_AVG3 ≈ {gap_avg3:.2f}  (midpoint between clusters)")
    print()
    print("Set these in .env or keep the defaults if they are close.")


if __name__ == "__main__":
    main()
