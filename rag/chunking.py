"""Document loading and chunking.

Supports `.pdf`, `.md`/`.markdown`, and `.txt`. Chunking is recursive on paragraph
then sentence boundaries with a character budget and overlap, which keeps related
sentences together far better than a naive fixed-width split.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from rag.types import Chunk

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def load_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix}")


def _split_units(text: str) -> list[str]:
    """Break text into atomic units (paragraphs, falling back to sentences)."""
    units: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= 1200:
            units.append(para)
        else:
            units.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return units


def chunk_text(
    text: str, source: str, chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    units = _split_units(text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    idx = 0

    def flush() -> None:
        nonlocal buf, buf_len, idx
        if not buf:
            return
        body = "\n\n".join(buf).strip()
        if body:
            chunks.append(
                Chunk(
                    id=f"{Path(source).name}::{idx}::{uuid.uuid5(uuid.NAMESPACE_URL, body[:64]).hex[:8]}",
                    text=body,
                    source=source,
                    chunk_index=idx,
                )
            )
            idx += 1

    for unit in units:
        if buf_len + len(unit) > chunk_size and buf:
            flush()
            # carry overlap from the tail of the previous chunk
            if overlap > 0:
                tail = "\n\n".join(buf)[-overlap:]
                buf, buf_len = [tail], len(tail)
            else:
                buf, buf_len = [], 0
        buf.append(unit)
        buf_len += len(unit)
    flush()
    return chunks


def load_and_chunk(
    path: str | Path, chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    text = load_text(path)
    return chunk_text(text, source=str(path), chunk_size=chunk_size, overlap=overlap)
