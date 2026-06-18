"""Dense text encoder.

All retrieval embeddings come from `sentence-transformers/all-MiniLM-L6-v2`
(384-d). Vectors are L2-normalized so that an inner product equals cosine
similarity, which lets the FAISS `IndexFlatIP` and the page-vector matmul share
one similarity definition.
"""
from __future__ import annotations

import time
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

EMBED_DIM = 384          # all-MiniLM-L6-v2 output width
MAX_SEQ_LENGTH = 256     # tokens; the encoder truncates past this

_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the encoder once and reuse it (picks GPU/MPS automatically)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _MODEL.max_seq_length = MAX_SEQ_LENGTH
    return _MODEL


def embed_texts(texts: Sequence[str], *, batch_size: int = 128,
                progress_every: int = 0) -> np.ndarray:
    """Encode `texts` into L2-normalized float32 rows, shape (len(texts), 384).

    Set `progress_every` > 0 to print an offline build rate/ETA every N rows
    (used by the index build; the query path leaves it at 0 for silence).
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    model = get_model()
    if progress_every <= 0:
        vectors = model.encode(
            list(texts), batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    out = np.empty((n, EMBED_DIM), dtype=np.float32)
    start = time.time()
    done = 0
    next_mark = progress_every
    for i in range(0, n, batch_size):
        block = list(texts[i:i + batch_size])
        out[i:i + len(block)] = model.encode(
            block, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        done += len(block)
        if done >= next_mark or done == n:
            rate = done / max(1e-9, time.time() - start)
            eta = (n - done) / max(1e-9, rate)
            print(f"      [embed] {done:,}/{n:,} ({100*done/n:.0f}%) | "
                  f"{rate:.0f} ch/s | ETA ~{eta/60:.1f}m", flush=True)
            next_mark += progress_every
    return out


def embed_queries(queries: List[str], *, batch_size: int = 128) -> np.ndarray:
    """Encode a batch of query strings (same space as the corpus vectors)."""
    return embed_texts(queries, batch_size=batch_size)
