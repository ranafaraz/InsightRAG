"""PII redaction stub.

Regex-based redaction for the highest-signal identifiers (email, phone, SSN, credit
card, IP). It is deliberately a *stub*: the full GuardrAIl library swaps this for
Microsoft Presidio (NER + context). The interface is identical so callers don't change.
"""
from __future__ import annotations

import re

_RULES: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, list_of_entity_types_found)."""
    found: list[str] = []
    redacted = text
    for label, pat in _RULES:
        if pat.search(redacted):
            found.append(label)
            redacted = pat.sub(f"[{label}_REDACTED]", redacted)
    return redacted, found
