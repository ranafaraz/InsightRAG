"""LLM generation backends.

`stub`   -- deterministic extractive generator (CI/offline). Parses the numbered
            CONTEXT block from the prompt and returns the most query-relevant
            sentence with its citation marker. No network, no keys.
`ollama` -- local models via the Ollama HTTP API (free).
`openai` -- hosted OpenAI chat models (optional, needs OPENAI_API_KEY).

All backends share one interface: ``generate(system, user) -> str``.
"""
from __future__ import annotations

import re
from typing import Protocol

from rag.config import Settings, get_settings
from rag.text_utils import content_token_set as _tokens

_CTX_LINE = re.compile(r"^\s*\[(\d+)\]\s*(?:\(source:[^)]*\))?\s*(.*)$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_QUESTION = re.compile(r"^QUESTION:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

REFUSAL = "I don't have enough information in the provided context to answer that."


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class StubLLM:
    """Extractive, deterministic generator used for offline tests and CI.

    It never invents facts: it selects the highest-overlap sentence from the
    supplied context and cites the passage it came from, mirroring how a faithful
    RAG answer should behave.
    """

    def generate(self, system: str, user: str) -> str:
        qmatch = _QUESTION.search(user)
        query = qmatch.group(1) if qmatch else user
        qtok = _tokens(query)

        passages: list[tuple[int, str]] = []
        for line in user.splitlines():
            m = _CTX_LINE.match(line)
            if m and m.group(2).strip():
                passages.append((int(m.group(1)), m.group(2).strip()))

        if not passages or not qtok:
            return REFUSAL

        best_sentence, best_marker, best_score = "", 0, -1.0
        for marker, text in passages:
            for sent in _SENT_SPLIT.split(text):
                score = len(qtok & _tokens(sent))
                if score > best_score and sent.strip():
                    best_sentence, best_marker, best_score = sent.strip(), marker, score

        if best_score <= 0:
            return REFUSAL
        return f"{best_sentence} [{best_marker}]"


class OllamaLLM:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system: str, user: str) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


class OpenAILLM:
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()


def get_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    if settings.llm_backend == "ollama":
        return OllamaLLM(settings.llm_model, settings.ollama_base_url)
    if settings.llm_backend == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_BACKEND=openai but OPENAI_API_KEY is not set")
        return OpenAILLM(settings.openai_model, settings.openai_api_key)
    return StubLLM()
