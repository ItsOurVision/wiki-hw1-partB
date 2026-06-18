"""Offline construction and loading of the on-disk search structures.

Run once on a full machine, the build writes four complementary signals into
``artifacts/`` and never runs again at query time:

* a product-quantized FAISS index over passage vectors (the passage signal),
* one dense vector per page plus the page texts (the page + rerank signals),
* a page-level BM25 model with its posting weights pre-multiplied (the lexical
  signal).

Loading hydrates a single :class:`CorpusIndex` that the retriever queries. The
passage index is product-quantized so it stays a few tens of MB on disk instead
of hundreds, which lets the whole ``artifacts/`` folder ship in a normal git
repository.
"""
from __future__ import annotations

import runtime  # noqa: F401  - installs the OpenMP guard before faiss is imported

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from chunk import Passage, passages_of
from embed import VECTOR_WIDTH, encode_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, entry_text, iter_entries

faiss.omp_set_num_threads(1)  # keep faiss off the threads torch/OpenMP also use

# ---- artifact file names (our schema) ---------------------------------------
F_PASSAGE_INDEX = "chunk.faiss"
F_PASSAGE_OWNER = "chunk_pages.npy"
F_PAGE_MATRIX = "page_vecs.npy"
F_PAGE_IDS = "page_ids.npy"
F_PAGE_TEXT = "page_texts.json"
F_LEX_POSTINGS = "bm25.npz"
F_LEX_VOCAB = "bm25_vocab.json"
F_META = "meta.json"

PQ_CODE_BYTES = 96           # PQ subquantizers -> ~PQ_CODE_BYTES bytes per vector
PAGE_TEXT_WORDS = 400        # words retained per page for the page/rerank signals
BM25_K1, BM25_B = 1.5, 0.75

_WORD_RE = re.compile(r"[a-z0-9]+")


def term_tokens(text: str) -> List[str]:
    """Lowercase alphanumeric tokenization shared by the BM25 build and queries."""
    return _WORD_RE.findall(text.lower())


def capped_page_text(record: Dict) -> str:
    """Title+body of a page, trimmed to PAGE_TEXT_WORDS words."""
    return " ".join(entry_text(record).split()[:PAGE_TEXT_WORDS])


# ---- in-memory view of the artifacts ----------------------------------------
@dataclass
class CorpusIndex:
    page_ids: List[int]
    page_matrix: np.ndarray          # (num_pages, dim), unit-norm float32
    page_text: List[str]             # rerank input, aligned to page_ids
    vocab: Dict[str, int]
    lex_ptr: np.ndarray              # CSR row pointer over terms
    lex_pages: np.ndarray            # posting page rows
    lex_weights: np.ndarray          # posting BM25 weights
    passages: faiss.Index            # PQ index over passage vectors
    passage_owner: np.ndarray        # passage-row -> page_id
    row_of: Dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.row_of = {pid: i for i, pid in enumerate(self.page_ids)}


def _build_lexical(token_docs: List[List[str]]):
    """Return (vocab, ptr, pages, weights): a per-term CSR of BM25 posting weights.

    Each posting's weight already folds in IDF and length normalization, so query
    scoring is just a sum of the relevant postings.
    """
    n_docs = len(token_docs)
    lengths = np.fromiter((len(d) for d in token_docs), dtype=np.float64, count=n_docs)
    mean_len = float(lengths.mean()) if n_docs else 0.0

    vocab: Dict[str, int] = {}
    doc_terms: List[Dict[int, int]] = []
    seen_in: List[int] = []                       # document frequency per term id
    for tokens in token_docs:
        counts: Dict[int, int] = {}
        for tok in tokens:
            tid = vocab.get(tok)
            if tid is None:
                tid = vocab[tok] = len(vocab)
                seen_in.append(0)
            counts[tid] = counts.get(tid, 0) + 1
        doc_terms.append(counts)
        for tid in counts:
            seen_in[tid] += 1

    df = np.asarray(seen_in, dtype=np.float64)
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))

    # collect postings grouped by term so the CSR layout is term-major
    grouped: List[List[tuple]] = [[] for _ in range(len(vocab))]
    for doc, counts in enumerate(doc_terms):
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * (lengths[doc] / mean_len if mean_len else 0.0))
        for tid, tf in counts.items():
            weight = idf[tid] * tf * (BM25_K1 + 1.0) / (tf + norm)
            grouped[tid].append((doc, weight))

    ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
    pages: List[int] = []
    weights: List[float] = []
    for tid, postings in enumerate(grouped):
        for doc, weight in postings:
            pages.append(doc)
            weights.append(weight)
        ptr[tid + 1] = len(pages)

    return (vocab, ptr,
            np.asarray(pages, dtype=np.int32),
            np.asarray(weights, dtype=np.float32))


def build_index(*, entries_dir: Optional[Path] = None,
                artifacts_dir: Optional[Path] = None) -> None:
    """Build every artifact from the corpus and persist it under ``artifacts/``."""
    out = artifacts_dir or ensure_artifacts_dir()
    pages = list(iter_entries(entries_dir))
    print(f"[build] {len(pages)} pages", flush=True)

    # passage signal: PQ-compressed FAISS over sliding windows
    passages: List[Passage] = passages_of(pages)
    print(f"[build] {len(passages)} passages; encoding", flush=True)
    pvecs = encode_texts([p.text for p in passages], report_every=20000)
    dim = int(pvecs.shape[1]) if pvecs.size else VECTOR_WIDTH
    pq = faiss.IndexPQ(dim, PQ_CODE_BYTES, 8, faiss.METRIC_INNER_PRODUCT)
    if pvecs.size:
        pq.train(pvecs)
        pq.add(pvecs)
    faiss.write_index(pq, str(out / F_PASSAGE_INDEX))
    np.save(out / F_PASSAGE_OWNER, np.fromiter((p.page_id for p in passages),
                                               dtype=np.int32, count=len(passages)))

    # page signal: one vector + capped text per page
    page_ids = [int(r["page_id"]) for r in pages]
    page_text = [capped_page_text(r) for r in pages]
    print("[build] encoding pages", flush=True)
    page_matrix = encode_texts(page_text)
    if page_matrix.size == 0:
        page_matrix = np.zeros((0, dim), dtype=np.float32)
    np.save(out / F_PAGE_MATRIX, page_matrix.astype(np.float32))
    np.save(out / F_PAGE_IDS, np.asarray(page_ids, dtype=np.int64))
    (out / F_PAGE_TEXT).write_text(json.dumps(page_text), encoding="utf-8")

    # lexical signal: page-level BM25
    print("[build] building BM25", flush=True)
    vocab, ptr, post_pages, post_weights = _build_lexical([term_tokens(t) for t in page_text])
    np.savez(out / F_LEX_POSTINGS, term_ptr=ptr, post_pages=post_pages, post_weights=post_weights)
    (out / F_LEX_VOCAB).write_text(json.dumps(vocab), encoding="utf-8")

    (out / F_META).write_text(json.dumps({
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": dim,
        "pages": len(page_ids),
        "passages": len(passages),
        "vocab": len(vocab),
        "pq_bytes": PQ_CODE_BYTES,
    }, indent=2), encoding="utf-8")
    print(f"[build] artifacts written to {out}", flush=True)


def load_index(artifacts_dir: Optional[Path] = None) -> CorpusIndex:
    """Load every artifact into a :class:`CorpusIndex`."""
    root = artifacts_dir or ARTIFACTS_DIR
    postings = np.load(root / F_LEX_POSTINGS)
    return CorpusIndex(
        page_ids=[int(x) for x in np.load(root / F_PAGE_IDS)],
        page_matrix=np.ascontiguousarray(np.load(root / F_PAGE_MATRIX), dtype=np.float32),
        page_text=json.loads((root / F_PAGE_TEXT).read_text(encoding="utf-8")),
        vocab=json.loads((root / F_LEX_VOCAB).read_text(encoding="utf-8")),
        lex_ptr=postings["term_ptr"],
        lex_pages=postings["post_pages"],
        lex_weights=postings["post_weights"],
        passages=faiss.read_index(str(root / F_PASSAGE_INDEX)),
        passage_owner=np.load(root / F_PASSAGE_OWNER),
    )
