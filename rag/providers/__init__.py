"""Pluggable provider backends for embeddings, reranking and generation.

Each factory reads the active backend from :class:`rag.config.Settings` and returns
an object implementing a small, stable protocol. A deterministic offline backend is
always available so nothing here requires network access or API keys by default.
"""
