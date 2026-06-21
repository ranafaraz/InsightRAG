"""Tiny shared text helpers for the deterministic offline backends."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

# A compact English stop-word list. Removing these stops trivial matches on
# function words ("the", "is", "of") from inflating lexical overlap scores —
# which is what lets the offline stub correctly refuse off-topic questions.
STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be by for from has have how in is it its of on or our that the
    their this to was were what when where which who why will with you your i we they he she
    do does did can could would should may might must""".split()
)


def content_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def content_token_set(text: str) -> set[str]:
    return set(content_tokens(text))
