"""Ingestion: turn files/directories of documents into stored, embedded chunks."""
from __future__ import annotations

from pathlib import Path

from rag.chunking import SUPPORTED_SUFFIXES, chunk_text, load_and_chunk
from rag.config import Settings, get_settings
from rag.store import VectorStore, get_store
from rag.types import Chunk


def iter_documents(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def ingest_path(
    path: str | Path,
    store: VectorStore | None = None,
    settings: Settings | None = None,
) -> tuple[VectorStore, int]:
    """Ingest a file or directory into the (optionally provided) store.

    Returns the store and the number of chunks added.
    """
    settings = settings or get_settings()
    store = store or get_store(settings)

    all_chunks: list[Chunk] = []
    for doc in iter_documents(path):
        all_chunks.extend(
            load_and_chunk(doc, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        )
    store.add(all_chunks)
    return store, len(all_chunks)


def ingest_texts(
    texts: list[str],
    sources: list[str] | None = None,
    store: VectorStore | None = None,
    settings: Settings | None = None,
) -> VectorStore:
    """Ingest raw in-memory strings — handy for evaluation and tests."""
    settings = settings or get_settings()
    store = store or get_store(settings)
    sources = sources or [f"doc_{i}" for i in range(len(texts))]
    chunks: list[Chunk] = []
    for text, src in zip(texts, sources):
        chunks.extend(
            chunk_text(text, source=src, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        )
    store.add(chunks)
    return store
