"""Shared data structures used across the RAG pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk of a source document, embedded and stored in the vector store."""

    id: str
    text: str
    source: str
    chunk_index: int = 0
    metadata: dict = Field(default_factory=dict)


class Scored(BaseModel):
    """A chunk with a retrieval/rerank score attached."""

    chunk: Chunk
    score: float
    # Component scores for transparency / debugging (bm25, dense, rerank).
    components: dict[str, float] = Field(default_factory=dict)


class Citation(BaseModel):
    marker: str          # e.g. "[1]"
    source: str
    chunk_id: str
    snippet: str


class Answer(BaseModel):
    query: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[Scored] = Field(default_factory=list)
    refused: bool = False
    blocked: bool = False          # blocked by a guardrail
    guardrail_flags: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
