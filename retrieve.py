"""Query-time ranking — the only timed code path.

A :class:`HybridRanker` holds the loaded artifacts and ranks each query in four
steps:

1. score every page by lexical BM25 and by dense cosine, and pull the best
   passage cosine per page from the FAISS index;
2. assemble a candidate pool (BM25 leaders, with a dense fallback, widened by the
   pages that own strong passages);
3. fuse the three signals over that pool with reciprocal rank fusion, which
   blends them by rank position and so needs no per-signal weight;
4. rerank the very top of the pool with a cross-encoder, mixed with the fusion
   score so a confident-but-wrong reranker cannot sink an already-good page.

If the cross-encoder cannot be instantiated (e.g. no model download is possible),
ranking silently degrades to the fusion order rather than erroring. The ranker and
its artifacts are built once and reused for the whole batch.
"""
from __future__ import annotations

import runtime  # noqa: F401  - installs the OpenMP guard before faiss/torch import

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np

from embed import encode_queries
from index import CorpusIndex, load_index, term_tokens

# ---- ranking configuration --------------------------------------------------
LEX_POOL = 100               # BM25 leaders taken as candidates
PASSAGE_POOL = 256           # passages fetched per query before pooling to pages
RESULT_DEPTH = 50            # pages emitted per query (grader scores the first 10)
RRF_K = 60                   # reciprocal-rank-fusion damping constant (standard)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_DEPTH = 12            # how many pool leaders the cross-encoder rescoring sees
RERANK_WEIGHT = 0.85         # cross-encoder share when mixed with the fusion score


def _unit_scale(values: np.ndarray) -> np.ndarray:
    """Min-max a score vector into [0, 1]; a constant vector collapses to zeros."""
    if values.size == 0:
        return values
    low = float(values.min())
    span = float(values.max()) - low
    return (values - low) / span if span > 1e-12 else np.zeros_like(values)


def _reciprocal_rank_fusion(columns: List[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Combine per-signal score columns by reciprocal rank fusion.

    Each signal votes for a candidate by 1 / (k + its_rank_in_that_signal); votes
    sum across signals. Using ranks rather than raw magnitudes makes the blend
    scale-free, so no per-signal weight needs tuning (and none can overfit).
    """
    fused = np.zeros(columns[0].shape[0], dtype=np.float64)
    for column in columns:
        order = np.argsort(-column)
        rank = np.empty(order.shape[0], dtype=np.int64)
        rank[order] = np.arange(order.shape[0])
        fused += 1.0 / (k + rank)
    return fused


class HybridRanker:
    """Ranks queries against a loaded :class:`CorpusIndex`."""

    def __init__(self, index: CorpusIndex):
        self.ix = index
        self._reranker = None
        self._reranker_off = False

    # -- individual signals ---------------------------------------------------
    def _lexical_scores(self, tokens: List[str]) -> np.ndarray:
        scores = np.zeros(len(self.ix.page_ids), dtype=np.float32)
        ptr, pages, weights = self.ix.lex_ptr, self.ix.lex_pages, self.ix.lex_weights
        for token in tokens:
            tid = self.ix.vocab.get(token)
            if tid is None:
                continue
            start, stop = int(ptr[tid]), int(ptr[tid + 1])
            if stop > start:                       # each term lists every page once
                scores[pages[start:stop]] += weights[start:stop]
        return scores

    def _best_passage(self, qvec: np.ndarray) -> Dict[int, float]:
        depth = min(PASSAGE_POOL, self.ix.passages.ntotal)
        if depth == 0:
            return {}
        sims, rows = self.ix.passages.search(qvec[None, :], depth)
        owner, best = self.ix.passage_owner, {}
        for sim, row in zip(sims[0], rows[0]):
            if row < 0:
                continue
            pid = int(owner[row])
            if sim > best.get(pid, -1e9):
                best[pid] = float(sim)
        return best

    def _reranker_model(self):
        if self._reranker is None and not self._reranker_off:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(RERANK_MODEL)
            except Exception as exc:  # pragma: no cover - depends on environment
                print(f"[retrieve] cross-encoder unavailable ({exc}); fusion-only ranking")
                self._reranker_off = True
        return self._reranker

    # -- per-query ranking ----------------------------------------------------
    def _candidate_pool(self, dense: np.ndarray, lex: np.ndarray,
                        passage: Dict[int, float]) -> np.ndarray:
        pool = np.argsort(-lex)[:LEX_POOL]
        if not bool((lex[pool] > 0).any()):        # query had no lexical overlap
            pool = np.argsort(-dense)[:LEX_POOL]
        owners = [self.ix.row_of[p] for p in passage if p in self.ix.row_of]
        if owners:
            pool = np.union1d(pool, np.asarray(owners, dtype=pool.dtype))
        return pool

    def rank_query(self, text: str, qvec: np.ndarray) -> List[int]:
        ix = self.ix
        with np.errstate(all="ignore"):            # silence macOS Accelerate FP noise
            dense = ix.page_matrix @ qvec
        lex = self._lexical_scores(term_tokens(text))
        passage = self._best_passage(qvec)

        pool = self._candidate_pool(dense, lex, passage)
        passage_col = np.fromiter(
            (passage.get(ix.page_ids[int(r)], 0.0) for r in pool),
            dtype=np.float32, count=pool.size)
        fused = _reciprocal_rank_fusion([dense[pool], lex[pool], passage_col])
        ordering = np.argsort(-fused)
        ranked = pool[ordering]
        fused_desc = fused[ordering]

        model = self._reranker_model()
        if model is not None and ranked.size:
            depth = min(RERANK_DEPTH, ranked.size)
            head = ranked[:depth]
            pairs = [(text, ix.page_text[int(r)]) for r in head]
            try:
                ce = np.asarray(model.predict(pairs, show_progress_bar=False))
                mixed = RERANK_WEIGHT * _unit_scale(ce) + (1.0 - RERANK_WEIGHT) * _unit_scale(fused_desc[:depth])
                ranked = np.concatenate([head[np.argsort(-mixed)], ranked[depth:]])
            except Exception as exc:  # pragma: no cover
                print(f"[retrieve] rerank skipped ({exc}); fusion order kept")

        out: List[int] = []
        emitted = set()
        for row in ranked:
            pid = ix.page_ids[int(row)]
            if pid not in emitted:
                emitted.add(pid)
                out.append(pid)
            if len(out) >= RESULT_DEPTH:
                break
        return out

    def rank_batch(self, queries: List[str]) -> List[List[int]]:
        qvecs = encode_queries(queries)
        if qvecs.size == 0:
            return [[] for _ in queries]
        qvecs = np.ascontiguousarray(qvecs, dtype=np.float32)
        return [self.rank_query(queries[i], qvecs[i]) for i in range(len(queries))]


_ENGINE: Optional[HybridRanker] = None


def rank_queries(queries: List[str], *, artifacts_dir: Optional[Path] = None) -> List[List[int]]:
    """Rank every query; returns one best-first list of page_id per query."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = HybridRanker(load_index(artifacts_dir))
    return _ENGINE.rank_batch(queries)
