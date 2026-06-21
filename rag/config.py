"""Central configuration loaded from environment / `.env`.

Every heavy component (embeddings, reranker, LLM) has a lightweight deterministic
backend so the full pipeline runs offline with no model downloads or API keys —
this is what keeps CI green and makes the repo trivially reproducible.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Backends
    embedding_backend: Literal["hash", "sentence-transformers"] = "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    rerank_backend: Literal["lexical", "cross-encoder"] = "lexical"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_backend: Literal["stub", "ollama", "openai"] = "stub"
    llm_model: str = "llama3.1:8b"

    # Providers
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Vector store: "memory" (numpy, no deps — used by CI) | "chroma" (persistent)
    vector_store: Literal["memory", "chroma"] = "memory"
    chroma_path: str = ".chroma"
    collection_name: str = "insightrag"

    # Retrieval / generation knobs
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieve_k: int = 20
    rerank_top_n: int = 5
    hybrid_alpha: float = 0.5
    min_rerank_score: float = 0.0
    max_context_chars: int = 8000

    # Embedding dimensionality for the deterministic "hash" backend
    hash_embedding_dim: int = 256


@lru_cache
def get_settings() -> Settings:
    return Settings()
