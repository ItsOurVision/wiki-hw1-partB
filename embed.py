"""Dense text encoder (the only embedding model in the system).

Queries, pages, and passages all pass through MiniLM into one 384-dimensional
space, returned unit-length so a dot product is already a cosine. The encoder is
built lazily and reused; the build step can ask for a progress line, the timed
query path stays silent.
"""
from __future__ import annotations

import time
from typing import List, Sequence

import runtime  # noqa: F401  - OpenMP guard, must precede torch
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

VECTOR_WIDTH = 384       # MiniLM-L6 hidden size
TOKEN_CAP = 256          # the encoder truncates past this many tokens

_MODEL: SentenceTransformer | None = None


def _model() -> SentenceTransformer:
    """Build the encoder on first call and cache it (uses GPU/MPS if present)."""
    global _MODEL
    if _MODEL is None:
        m = SentenceTransformer(EMBEDDING_MODEL_NAME)
        m.max_seq_length = TOKEN_CAP
        _MODEL = m
    return _MODEL


def _run(rows: List[str], size: int) -> np.ndarray:
    return _model().encode(rows, batch_size=size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)


def encode_texts(texts: Sequence[str], *, batch: int = 128, report_every: int = 0) -> np.ndarray:
    """Encode `texts` to unit-norm float32 rows of shape (len(texts), 384).

    With `report_every > 0` a throughput/ETA line is printed every N rows (used by
    the offline build); the query path leaves it at 0.
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, VECTOR_WIDTH), dtype=np.float32)
    if report_every <= 0:
        return np.ascontiguousarray(_run(list(texts), batch), dtype=np.float32)

    buf = np.empty((n, VECTOR_WIDTH), dtype=np.float32)
    t0 = time.time()
    mark = report_every
    done = 0
    while done < n:
        chunk = list(texts[done:done + batch])
        buf[done:done + len(chunk)] = _run(chunk, batch)
        done += len(chunk)
        if done >= mark or done == n:
            rate = done / max(1e-9, time.time() - t0)
            print(f"      encoded {done:,}/{n:,} ({100 * done / n:.0f}%)  "
                  f"{rate:.0f}/s  eta {(n - done) / max(1e-9, rate) / 60:.1f}m", flush=True)
            mark += report_every
    return buf


def encode_queries(queries: List[str], *, batch: int = 128) -> np.ndarray:
    """Encode a batch of queries into the shared embedding space."""
    return encode_texts(queries, batch=batch)
