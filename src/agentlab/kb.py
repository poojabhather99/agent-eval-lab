"""A tiny synthetic company handbook used as the agent's knowledge base."""
from __future__ import annotations

import re

HANDBOOK = [
    "Annual leave: full-time employees receive 24 days of paid annual leave per year.",
    "Sick leave: employees receive 12 paid sick days per year.",
    "Remote work: employees may work remotely up to 3 days per week with manager approval.",
    "Expenses: managers may approve expenses up to 5000 USD; directors up to 25000 USD.",
    "Notice period: the standard notice period for resignation is 30 days.",
    "Probation: new hires serve a probation period of 90 days.",
    "Working hours: the standard working week is 40 hours.",
]

_STOP = {"the", "a", "an", "of", "is", "for", "to", "do", "does", "how", "many", "what", "are", "in", "per", "and"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOP and len(t) > 2}


def search(query: str, top_k: int = 2) -> list[str]:
    """Rank handbook entries by keyword overlap; return the top entries with a non-zero score."""
    q = _tokens(query)
    scored = sorted(((len(q & _tokens(doc)), doc) for doc in HANDBOOK), key=lambda x: -x[0])
    return [doc for score, doc in scored[:top_k] if score > 0]
