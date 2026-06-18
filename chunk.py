"""Turn corpus pages into the passage units we actually index.

A single embedding for a long article averages many unrelated topics together,
which hides the one paragraph a query cares about. We therefore slide a fixed-size
word window across each page and emit one passage per window, repeating the page
title at the start of every window so the entity name travels with the text. Short
pages stay as a single passage. The window/stride pair below was chosen by the
offline evaluation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from utils import entry_text

WINDOW_WORDS = 180       # words per passage window
STRIDE_BACK = 40         # words shared with the previous window (overlap)

_WS = re.compile(r"\s+")


@dataclass
class Passage:
    """A single indexed unit and a back-pointer to the page it came from."""
    page_id: int
    ordinal: int
    text: str


def _sliding(words: List[str]) -> Iterable[str]:
    """Yield successive WINDOW_WORDS slices advancing by (WINDOW_WORDS - STRIDE_BACK)."""
    advance = max(1, WINDOW_WORDS - STRIDE_BACK)
    cursor = 0
    n = len(words)
    while cursor < n:
        yield " ".join(words[cursor:cursor + WINDOW_WORDS])
        if cursor + WINDOW_WORDS >= n:
            return
        cursor += advance


def split_page(record: Dict[str, Any]) -> List[Passage]:
    """Break one page record into one or more title-led passages."""
    page_id = int(record["page_id"])
    full = entry_text(record)
    words = _WS.sub(" ", full).split()

    if len(words) <= WINDOW_WORDS:
        bodies = [full] if full else [str(record.get("title", "")).strip()]
    else:
        head = str(record.get("title", "")).strip()
        lead = f"{head}\n\n" if head else ""
        bodies = [f"{lead}{window}".strip() for window in _sliding(words)]

    return [Passage(page_id=page_id, ordinal=i, text=body) for i, body in enumerate(bodies)]


def passages_of(records: Iterable[Dict[str, Any]]) -> List[Passage]:
    """Flatten an iterable of page records into one passage list."""
    out: List[Passage] = []
    for record in records:
        out.extend(split_page(record))
    return out
