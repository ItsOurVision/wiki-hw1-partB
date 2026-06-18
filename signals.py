"""The three retrieval signals, each as a self-contained scorer.

Rather than one large retrieval routine, every relevance view the system uses is
its own small object that scores the whole corpus for a single query:

* :class:`DenseSignal`   - whole-page cosine,
* :class:`LexicalSignal` - page-level BM25,
* :class:`PassageSignal` - best-matching passage cosine.

Splitting them this way means the offline build and the timed query path call the
exact same code, and any one signal can be tested or replaced on its own. The
fusion that merges their outputs lives here too, so `retrieve.py` only has to wire
the pieces together.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens (shared by the BM25 fit and query time)."""
    return _TOKEN.findall(text.lower())


def rescale(values: np.ndarray) -> np.ndarray:
    """Min-max a score vector into [0, 1]; a flat vector collapses to zeros."""
    if values.size == 0:
        return values
    low = float(values.min())
    width = float(values.max()) - low
    return (values - low) / width if width > 1e-12 else np.zeros_like(values)


def rank_fusion(views: List[np.ndarray], constant: int = 60) -> np.ndarray:
    """Reciprocal rank fusion: each view contributes 1/(constant + its rank)."""
    pooled = np.zeros(views[0].shape[0], dtype=np.float64)
    for view in views:
        place = np.empty(view.shape[0], dtype=np.int64)
        place[np.argsort(-view)] = np.arange(view.shape[0])
        pooled += 1.0 / (constant + place)
    return pooled


class DenseSignal:
    """Cosine between the query and every whole-page vector (one matmul)."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def score(self, qvec: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):            # silence macOS Accelerate FP noise
            return self.matrix @ qvec


class LexicalSignal:
    """Page-level BM25 with posting weights precomputed at build time."""

    K1, B = 1.5, 0.75

    def __init__(self, vocab: Dict[str, int], ptr: np.ndarray,
                 pages: np.ndarray, weights: np.ndarray, n_pages: int):
        self.vocab, self.ptr, self.pages, self.weights, self.n = vocab, ptr, pages, weights, n_pages

    def score(self, tokens: List[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for token in tokens:
            tid = self.vocab.get(token)
            if tid is None:
                continue
            a, b = int(self.ptr[tid]), int(self.ptr[tid + 1])
            if b > a:                              # each term lists a page at most once
                out[self.pages[a:b]] += self.weights[a:b]
        return out

    @classmethod
    def fit(cls, token_docs: List[List[str]]):
        """Return (vocab, ptr, pages, weights): term-major CSR of BM25 weights."""
        n_docs = len(token_docs)
        length = np.fromiter((len(d) for d in token_docs), dtype=np.float64, count=n_docs)
        avg = float(length.mean()) if n_docs else 0.0

        vocab: Dict[str, int] = {}
        df: List[int] = []
        per_doc: List[Dict[int, int]] = []
        for tokens in token_docs:
            local: Dict[int, int] = {}
            for tok in tokens:
                tid = vocab.get(tok)
                if tid is None:
                    tid = vocab[tok] = len(vocab)
                    df.append(0)
                local[tid] = local.get(tid, 0) + 1
            per_doc.append(local)
            for tid in local:
                df[tid] += 1

        idf = np.log1p((n_docs - np.asarray(df, dtype=np.float64) + 0.5) /
                       (np.asarray(df, dtype=np.float64) + 0.5))
        buckets: List[List[Tuple[int, float]]] = [[] for _ in range(len(vocab))]
        for doc, local in enumerate(per_doc):
            norm = cls.K1 * (1.0 - cls.B + cls.B * (length[doc] / avg if avg else 0.0))
            for tid, tf in local.items():
                buckets[tid].append((doc, idf[tid] * tf * (cls.K1 + 1.0) / (tf + norm)))

        ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        pages: List[int] = []
        weights: List[float] = []
        for tid, bucket in enumerate(buckets):
            for doc, w in bucket:
                pages.append(doc)
                weights.append(w)
            ptr[tid + 1] = len(pages)
        return vocab, ptr, np.asarray(pages, dtype=np.int32), np.asarray(weights, dtype=np.float32)


class PassageSignal:
    """Best passage-cosine per page, read from a PQ FAISS index."""

    def __init__(self, faiss_index, owner: np.ndarray):
        self.index = faiss_index
        self.owner = owner

    def best_per_page(self, qvec: np.ndarray, probe: int) -> Dict[int, float]:
        depth = min(probe, self.index.ntotal)
        if depth == 0:
            return {}
        sims, rows = self.index.search(qvec[None, :], depth)
        peak: Dict[int, float] = {}
        for sim, row in zip(sims[0], rows[0]):
            if row < 0:
                continue
            pid = int(self.owner[row])
            if sim > peak.get(pid, -1e9):
                peak[pid] = float(sim)
        return peak
