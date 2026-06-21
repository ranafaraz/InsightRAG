"""Prompt-injection / jailbreak screening for user queries.

A pragmatic pattern-based detector. It is not a silver bullet, but it catches the
common 'ignore previous instructions / reveal your system prompt / exfiltrate'
families and is cheap enough to run on every request. Precision/recall on the
bundled labelled set is reported in the eval table.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore_instructions", re.compile(r"\bignore\s+(?:all\s+|the\s+|your\s+)?(?:previous|prior|above)\b.*\binstruction", re.I)),
    ("disregard", re.compile(r"\bdisregard\s+(?:all\s+|the\s+|your\s+)?(?:previous|prior|above|system)", re.I)),
    ("override_system", re.compile(r"\b(?:override|forget|reset|bypass)\b.*\b(?:system\s+prompt|instructions|rules|guardrails)\b", re.I)),
    ("reveal_prompt", re.compile(r"\b(?:reveal|show|print|repeat|leak)\b.*\b(?:system\s+prompt|your\s+instructions|prompt)\b", re.I)),
    ("role_override", re.compile(r"\byou\s+are\s+now\b|\bact\s+as\b.*\b(?:DAN|jailbreak|unfiltered|developer\s+mode)\b", re.I)),
    ("do_anything", re.compile(r"\bdo\s+anything\s+now\b|\bdeveloper\s+mode\b", re.I)),
    ("exfiltrate", re.compile(r"\b(?:send|email|post|exfiltrate|upload)\b.*\b(?:api\s*key|secret|password|credentials)\b", re.I)),
    ("delimiter_attack", re.compile(r"(?:^|\n)\s*(?:system|assistant)\s*:", re.I)),
]


class InjectionVerdict(BaseModel):
    is_injection: bool
    score: float           # 0..1 share of patterns triggered (capped at 1)
    matched: list[str]


def check_injection(text: str) -> InjectionVerdict:
    matched = [name for name, pat in _PATTERNS if pat.search(text)]
    score = min(1.0, len(matched) / 2.0)  # 2+ signals -> full confidence
    return InjectionVerdict(is_injection=bool(matched), score=score, matched=matched)
