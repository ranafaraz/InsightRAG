"""Vector store backends.

`memory` -- numpy cosine-similarity store; zero external deps, used by CI.
`chroma` -- persistent Chroma collection for local/production use.

Both expose the same minimal interface so the retriever is storage-agnostic.
"""
from __future__ import annotations

from typing import Protocol

from rag.config import Settings, get_settings
from rag.providers.embeddings import Embedder, get_embedder
from rag.types import Chunk, Scored


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def query(self, text: str, k: int) -> list[Scored]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def count(self) -> int: ...


class InMemoryStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self.embedder.embed([c.text for c in chunks])
        self._chunks.extend(chunks)
        self._vectors.extend(vecs)

    def query(self, text: str, k: int) -> list[Scored]:
        if not self._chunks:
            return []
        import numpy as np

        q = np.asarray(self.embedder.embed([text])[0])
        mat = np.asarray(self._vectors)
        sims = mat @ q  # vectors are L2-normalised -> dot == cosine
        top = np.argsort(-sims)[:k]
        return [
            Scored(chunk=self._chunks[i], score=float(sims[i]), components={"dense": float(sims[i])})
            for i in top
        ]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def count(self) -> int:
        return len(self._chunks)


class ChromaStore:
    def __init__(self, embedder: Embedder, path: str, collection: str) -> None:
        import chromadb

        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._col.add(
            ids=[c.id for c in chunks],
            embeddings=self.embedder.embed([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        )

    def query(self, text: str, k: int) -> list[Scored]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=self.embedder.embed([text]),
            n_results=min(k, self._col.count()),
        )
        out: list[Scored] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            out.append(
                Scored(
                    chunk=Chunk(
                        id=cid,
                        text=doc,
                        source=meta.get("source", "unknown"),
                        chunk_index=int(meta.get("chunk_index", 0)),
                    ),
                    score=score,
                    components={"dense": score},
                )
            )
        return out

    def all_chunks(self) -> list[Chunk]:
        res = self._col.get()
        return [
            Chunk(
                id=cid,
                text=doc,
                source=meta.get("source", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
            )
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def count(self) -> int:
        return self._col.count()


def get_store(settings: Settings | None = None, embedder: Embedder | None = None) -> VectorStore:
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    if settings.vector_store == "chroma":
        return ChromaStore(embedder, settings.chroma_path, settings.collection_name)
    return InMemoryStore(embedder)
