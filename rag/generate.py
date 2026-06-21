"""Grounded answer generation with inline citations and an honest refusal path."""
from __future__ import annotations

import re

from rag.config import Settings, get_settings
from rag.providers.llm import LLM, REFUSAL, get_llm
from rag.types import Answer, Citation, Scored

SYSTEM_PROMPT = (
    "You are InsightRAG, a careful assistant that answers ONLY from the provided context.\n"
    "Rules:\n"
    "1. Use only facts found in the numbered CONTEXT passages.\n"
    "2. Cite every claim with its passage marker, e.g. [1] or [2][3].\n"
    "3. If the context does not contain the answer, reply exactly: "
    f'"{REFUSAL}"\n'
    "4. Be concise and do not invent sources, numbers, or names."
)

_MARKER = re.compile(r"\[(\d+)\]")


def build_prompt(query: str, contexts: list[Scored], max_chars: int) -> str:
    lines, used = ["CONTEXT:"], 0
    for i, sc in enumerate(contexts, start=1):
        text = sc.chunk.text.strip().replace("\n", " ")
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        lines.append(f"[{i}] (source: {sc.chunk.source}) {text}")
        used += len(text)
        if used >= max_chars:
            break
    lines.append("")
    lines.append(f"QUESTION: {query}")
    lines.append("ANSWER (with citations):")
    return "\n".join(lines)


def _citations_for(text: str, contexts: list[Scored]) -> list[Citation]:
    cites: list[Citation] = []
    seen: set[int] = set()
    for m in _MARKER.finditer(text):
        n = int(m.group(1))
        if n in seen or not (1 <= n <= len(contexts)):
            continue
        seen.add(n)
        chunk = contexts[n - 1].chunk
        snippet = chunk.text.strip().replace("\n", " ")
        cites.append(
            Citation(
                marker=f"[{n}]",
                source=chunk.source,
                chunk_id=chunk.id,
                snippet=snippet[:240] + ("…" if len(snippet) > 240 else ""),
            )
        )
    return cites


def generate_answer(
    query: str,
    contexts: list[Scored],
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> Answer:
    settings = settings or get_settings()

    # Refuse before spending a token when retrieval is empty or too weak.
    top_score = contexts[0].score if contexts else 0.0
    if not contexts or top_score < settings.min_rerank_score:
        return Answer(query=query, text=REFUSAL, contexts=contexts, refused=True)

    llm = llm or get_llm(settings)
    prompt = build_prompt(query, contexts, settings.max_context_chars)
    text = llm.generate(SYSTEM_PROMPT, prompt).strip()

    refused = text.strip().lower().startswith(REFUSAL[:20].lower())
    citations = [] if refused else _citations_for(text, contexts)
    return Answer(
        query=query,
        text=text,
        citations=citations,
        contexts=contexts,
        refused=refused,
    )
