"""Turn corpus pages into the passage units we index.

A long article squeezed into one vector averages many topics and hides the one
paragraph a query cares about. So we walk a fixed-size word window across each
page and emit one passage per step, repeating the title at the front of every
window so the subject travels with the text. Short pages become a single passage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from utils import entry_text

WIN = 180            # words per passage window
HOP_BACK = 40        # words shared with the previous window (overlap)

_SPACES = re.compile(r"\s+")


@dataclass
class Passage:
    """One indexed unit plus a back-pointer to its page."""
    page_id: int
    ordinal: int
    text: str


def _windows(words: List[str]) -> Iterable[str]:
    """Yield WIN-word slices advancing by (WIN - HOP_BACK) words each step."""
    step = max(1, WIN - HOP_BACK)
    total = len(words)
    at = 0
    while at < total:
        yield " ".join(words[at:at + WIN])
        if at + WIN >= total:
            return
        at += step


def split_page(record: Dict[str, Any]) -> List[Passage]:
    """Break one page into title-led passages."""
    page_id = int(record["page_id"])
    joined = entry_text(record)
    words = _SPACES.sub(" ", joined).split()

    if len(words) <= WIN:
        texts = [joined] if joined else [str(record.get("title", "")).strip()]
    else:
        title = str(record.get("title", "")).strip()
        prefix = f"{title}\n\n" if title else ""
        texts = [f"{prefix}{w}".strip() for w in _windows(words)]

    return [Passage(page_id, i, t) for i, t in enumerate(texts)]


def passages_of(records: Iterable[Dict[str, Any]]) -> List[Passage]:
    """Flatten page records into one passage list."""
    flat: List[Passage] = []
    for record in records:
        flat += split_page(record)
    return flat
