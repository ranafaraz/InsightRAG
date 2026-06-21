from rag.chunking import chunk_text


def test_chunking_respects_size_budget():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." for i in range(40))
    chunks = chunk_text(text, source="doc.md", chunk_size=200, overlap=20)
    assert len(chunks) > 1
    # allow overlap slack but no runaway chunks
    assert all(len(c.text) <= 200 + 120 for c in chunks)
    assert all(c.source == "doc.md" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_unique():
    text = "\n\n".join(f"Distinct paragraph {i}." for i in range(10))
    chunks = chunk_text(text, source="d.md", chunk_size=50, overlap=0)
    assert len({c.id for c in chunks}) == len(chunks)


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ", source="d.md") == []
