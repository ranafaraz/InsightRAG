"""Shared fixtures. All tests force the deterministic offline backends so the
suite needs no model downloads, network, or API keys."""
from __future__ import annotations

import pytest

from rag.config import Settings
from rag.pipeline import RAGPipeline

DOCS = [
    "InsightRAG combines BM25 sparse retrieval with dense vector search and reranks "
    "candidates using a cross-encoder before generating a grounded answer.",
    "The capital of France is Paris, which is famous for the Eiffel Tower and the Louvre.",
    "Diversification is a risk-management strategy that mixes many investments to limit "
    "exposure to any single asset.",
    "Spaced repetition increases the interval between reviews to exploit the spacing effect "
    "for better long-term recall.",
]
SOURCES = ["insightrag.md", "paris.md", "finance.md", "learning.md"]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        embedding_backend="hash",
        rerank_backend="lexical",
        llm_backend="stub",
        vector_store="memory",
    )


@pytest.fixture
def pipeline(settings) -> RAGPipeline:
    pipe = RAGPipeline(settings=settings)
    pipe.ingest_texts(DOCS, SOURCES)
    return pipe
