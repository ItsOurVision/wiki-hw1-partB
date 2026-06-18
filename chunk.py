"""Corpus preprocessing and chunking.

A whole Wikipedia page is often too long for a 256-token encoder, so a single
page embedding blurs many topics together. We instead split each page into
overlapping word windows and embed those; at query time the page inherits the
score of its best-matching window (max-pool). The page title is prepended to
every window so each unit still names the entity it describes.

`TARGET_WORDS` / `OVERLAP_WORDS` are the knobs we sweep offline; see the README
for the chosen values and their NDCG@10 effect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

TARGET_WORDS = 180       # words per window (before the title prefix)
OVERLAP_WORDS = 40       # words shared between consecutive windows

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """One retrieval unit and the page it belongs to."""
    page_id: int
    chunk_id: int
    text: str


def _windows(words: List[str], size: int, overlap: int) -> List[str]:
    """Slide a `size`-word window over `words`, stepping by `size - overlap`."""
    if not words:
        return []
    step = max(1, size - overlap)
    out: List[str] = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one page into title-prefixed, overlapping word windows."""
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    body = entry_text(record)
    words = body.split()

    if len(words) <= TARGET_WORDS:
        texts = [body] if body else [title]
    else:
        prefix = f"{title}\n\n" if title else ""
        texts = [f"{prefix}{w}".strip() for w in _windows(words, TARGET_WORDS, OVERLAP_WORDS)]

    return [Chunk(page_id=page_id, chunk_id=i, text=t) for i, t in enumerate(texts)]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Flatten every record into a single list of chunks."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
