"""
Mistral embeddings client.
- Batches chunks in groups of 32
- Exponential backoff on 429/5xx
- L2-normalizes vectors at return time (enables cosine = dot product)
"""
import time
import logging
from typing import List

import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 32
_RETRIES = 3
_BACKOFF = [2.0, 4.0, 8.0]


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using Mistral embeddings API.

    Returns:
        np.ndarray of shape (len(texts), D), L2-normalized (float32).
    """
    all_vectors = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        vectors = _embed_batch_with_retry(batch)
        all_vectors.extend(vectors)

    matrix = np.array(all_vectors, dtype=np.float32)
    return _l2_normalize(matrix)


def embed_single(text: str) -> np.ndarray:
    """Embed a single text, return 1D normalized vector."""
    return embed_texts([text])[0]


def _embed_batch_with_retry(texts: List[str]) -> List[List[float]]:
    last_error = None

    for attempt in range(_RETRIES):
        try:
            response = httpx.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": settings.MISTRAL_EMBED_MODEL, "input": texts},
                timeout=30.0,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                wait = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
                logger.warning(f"Mistral embed HTTP {response.status_code}, retrying in {wait}s")
                time.sleep(wait)
                last_error = f"HTTP {response.status_code}"
                continue

            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]

        except httpx.TimeoutException as e:
            wait = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
            logger.warning(f"Mistral embed timeout, retrying in {wait}s")
            time.sleep(wait)
            last_error = str(e)

    raise RuntimeError(f"Mistral embeddings failed after {_RETRIES} attempts: {last_error}")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so cosine similarity = dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms
