"""Dense text encoder built on the mandated MiniLM sentence model.

Every vector in the system — corpus pages, passages, and incoming queries — is
produced here so they all live in the same 384-dimensional space. Vectors are
returned unit-normalized; with unit vectors a plain dot product already equals
cosine similarity, which keeps the FAISS index and the page-matrix scoring on one
common scale.
"""
from __future__ import annotations

import time
from typing import List, Sequence

import runtime  # noqa: F401  - installs the OpenMP guard before torch is imported
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

VECTOR_WIDTH = 384       # MiniLM-L6 hidden size
TOKEN_BUDGET = 256       # the encoder truncates anything longer

_encoder: SentenceTransformer | None = None


def encoder() -> SentenceTransformer:
    """Return the shared encoder, constructing it on first use (GPU/MPS aware)."""
    global _encoder
    if _encoder is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.max_seq_length = TOKEN_BUDGET
        _encoder = model
    return _encoder


def _encode_block(block: List[str], batch: int) -> np.ndarray:
    return encoder().encode(
        block,
        batch_size=batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def encode_texts(texts: Sequence[str], *, batch: int = 128, report_every: int = 0) -> np.ndarray:
    """Encode `texts` to unit-norm float32 rows of shape (len(texts), 384).

    `report_every > 0` prints a throughput/ETA line every N rows; the offline
    build turns this on, while the timed query path leaves it silent.
    """
    count = len(texts)
    if count == 0:
        return np.zeros((0, VECTOR_WIDTH), dtype=np.float32)

    if report_every <= 0:
        return np.ascontiguousarray(_encode_block(list(texts), batch), dtype=np.float32)

    matrix = np.empty((count, VECTOR_WIDTH), dtype=np.float32)
    started = time.time()
    milestone = report_every
    filled = 0
    while filled < count:
        block = list(texts[filled:filled + batch])
        matrix[filled:filled + len(block)] = _encode_block(block, batch)
        filled += len(block)
        if filled >= milestone or filled == count:
            speed = filled / max(1e-9, time.time() - started)
            remaining = (count - filled) / max(1e-9, speed)
            print(f"      embedded {filled:,}/{count:,} "
                  f"({100 * filled / count:.0f}%)  {speed:.0f}/s  eta {remaining / 60:.1f}m",
                  flush=True)
            milestone += report_every
    return matrix


def encode_queries(queries: List[str], *, batch: int = 128) -> np.ndarray:
    """Encode a batch of query strings into the shared embedding space."""
    return encode_texts(queries, batch=batch)
