"""Embedding backends.

`hash`                 -- deterministic, dependency-free hashing embedding (CI/offline).
`sentence-transformers`-- real dense embeddings (default for local use; needs the
                          `local` extra and a one-time model download).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from rag.config import Settings, get_settings

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """A bag-of-words hashing embedding with L2 normalisation.

    Not semantically rich, but fully deterministic and download-free, which makes it
    ideal for tests and reproducible CI. Cosine similarity over these vectors behaves
    like a TF weighted lexical match — good enough to validate the plumbing.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    return HashingEmbedder(dim=settings.hash_embedding_dim)
