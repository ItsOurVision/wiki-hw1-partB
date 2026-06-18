"""Query-time ranking — the only timed code path.

A :class:`HybridRanker` keeps the loaded artifacts in memory and ranks each query
by combining three relevance views (whole-page cosine, BM25, best-passage cosine)
with reciprocal rank fusion, then refining the very top with a cross-encoder. The
ranker and its models are created once and reused for the whole batch. If the
cross-encoder cannot be instantiated, ranking degrades to the fusion order rather
than raising.
"""
from __future__ import annotations

import runtime  # noqa: F401  - OpenMP guard, must precede faiss/torch

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from embed import encode_queries
from index import CorpusIndex, load_index, term_tokens

# --- knobs (fixed by offline experiments) ------------------------------------
BM25_SHORTLIST = 100         # how many BM25 leaders enter the candidate set
PASSAGE_PROBE = 256          # passages pulled from FAISS before pooling to pages
OUTPUT_LIMIT = 50            # ranked pages returned per query (top 10 are scored)
RRF_C = 60                   # reciprocal-rank-fusion constant
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP = 12              # how deep the cross-encoder re-scores
RERANK_MIX = 0.85            # cross-encoder share when mixed with fusion


def _norm01(values: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1]; a constant vector becomes all zeros."""
    if values.size == 0:
        return values
    floor = float(values.min())
    spread = float(values.max()) - floor
    return (values - floor) / spread if spread > 1e-12 else np.zeros_like(values)


def _fuse_by_rank(views: List[np.ndarray], constant: int = RRF_C) -> np.ndarray:
    """Reciprocal rank fusion: each view votes 1/(c + rank); votes are summed."""
    tally = np.zeros(views[0].shape[0], dtype=np.float64)
    for view in views:
        position = np.empty(view.shape[0], dtype=np.int64)
        position[np.argsort(-view)] = np.arange(view.shape[0])
        tally += 1.0 / (constant + position)
    return tally


class HybridRanker:
    """Ranks queries against a loaded :class:`CorpusIndex`."""

    def __init__(self, store: CorpusIndex):
        self.store = store
        self._row_for = {pid: i for i, pid in enumerate(store.page_ids)}
        self._ce = None
        self._ce_dead = False

    # ---- the three views ----------------------------------------------------
    def _keyword_view(self, terms: List[str]) -> np.ndarray:
        store = self.store
        acc = np.zeros(len(store.page_ids), dtype=np.float32)
        ptr, docs, wts = store.lex_ptr, store.lex_pages, store.lex_weights
        for term in terms:
            tid = store.vocab.get(term)
            if tid is None:
                continue
            a, b = int(ptr[tid]), int(ptr[tid + 1])
            if b > a:                              # a term lists each page at most once
                acc[docs[a:b]] += wts[a:b]
        return acc

    def _passage_view(self, qvec: np.ndarray) -> Dict[int, float]:
        store = self.store
        depth = min(PASSAGE_PROBE, store.passages.ntotal)
        if depth == 0:
            return {}
        sims, rows = store.passages.search(qvec[None, :], depth)
        peak: Dict[int, float] = {}
        for sim, row in zip(sims[0], rows[0]):
            if row < 0:
                continue
            pid = int(store.passage_owner[row])
            if sim > peak.get(pid, -1e9):
                peak[pid] = float(sim)
        return peak

    def _reranker(self):
        if self._ce is None and not self._ce_dead:
            try:
                from sentence_transformers import CrossEncoder
                self._ce = CrossEncoder(RERANKER)
            except Exception as err:  # pragma: no cover - environment dependent
                print(f"[retrieve] reranker unavailable ({err}); fusion-only ranking")
                self._ce_dead = True
        return self._ce

    # ---- candidate set + ranking -------------------------------------------
    def _shortlist(self, page_cos: np.ndarray, keyword: np.ndarray,
                   passage: Dict[int, float]) -> np.ndarray:
        chosen = np.argsort(-keyword)[:BM25_SHORTLIST]
        if not bool((keyword[chosen] > 0).any()):     # no lexical overlap at all
            chosen = np.argsort(-page_cos)[:BM25_SHORTLIST]
        owning = [self._row_for[p] for p in passage if p in self._row_for]
        if owning:
            chosen = np.union1d(chosen, np.asarray(owning, dtype=chosen.dtype))
        return chosen

    def _rank_one(self, text: str, qvec: np.ndarray) -> List[int]:
        store = self.store
        with np.errstate(all="ignore"):               # quiet macOS Accelerate FP noise
            page_cos = store.page_matrix @ qvec
        keyword = self._keyword_view(term_tokens(text))
        passage = self._passage_view(qvec)

        cand = self._shortlist(page_cos, keyword, passage)
        passage_col = np.fromiter(
            (passage.get(store.page_ids[int(r)], 0.0) for r in cand),
            dtype=np.float32, count=cand.size)
        votes = _fuse_by_rank([page_cos[cand], keyword[cand], passage_col])
        order = np.argsort(-votes)
        sequence = cand[order]
        votes_sorted = votes[order]

        ce = self._reranker()
        if ce is not None and sequence.size:
            depth = min(RERANK_TOP, sequence.size)
            front = sequence[:depth]
            pairs = [(text, store.page_text[int(r)]) for r in front]
            try:
                ce_raw = np.asarray(ce.predict(pairs, show_progress_bar=False))
                merged = RERANK_MIX * _norm01(ce_raw) + (1.0 - RERANK_MIX) * _norm01(votes_sorted[:depth])
                sequence = np.concatenate([front[np.argsort(-merged)], sequence[depth:]])
            except Exception as err:  # pragma: no cover
                print(f"[retrieve] rerank skipped ({err}); keeping fusion order")

        ranked_ids = (store.page_ids[int(r)] for r in sequence)
        return list(dict.fromkeys(ranked_ids))[:OUTPUT_LIMIT]   # de-dup, keep order

    def run_batch(self, queries: List[str]) -> List[List[int]]:
        qvecs = encode_queries(queries)
        if qvecs.size == 0:
            return [[] for _ in queries]
        qvecs = np.ascontiguousarray(qvecs, dtype=np.float32)
        return [self._rank_one(q, qvecs[i]) for i, q in enumerate(queries)]


_ENGINE: Optional[HybridRanker] = None


def rank_queries(queries: List[str], *, artifacts_dir: Optional[Path] = None) -> List[List[int]]:
    """Rank every query; returns one best-first list of page_id per query."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = HybridRanker(load_index(artifacts_dir))
    return _ENGINE.run_batch(queries)
