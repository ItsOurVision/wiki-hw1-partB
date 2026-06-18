"""Offline construction and loading of the on-disk search structures.

The build runs once on a full machine and writes everything `run()` needs into
``artifacts/``: a product-quantized FAISS index over passage vectors, one dense
vector per page, a page-level BM25 model with its posting weights folded in, and
the page texts the cross-encoder reads. Loading rehydrates a single
:class:`CorpusIndex`. The passage index is product-quantized so the folder stays a
few tens of MB and ships in a plain git repository (no LFS).
"""
from __future__ import annotations

import runtime  # noqa: F401  - OpenMP guard, must precede faiss/torch

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from chunk import Passage, passages_of
from embed import VECTOR_WIDTH, encode_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, entry_text, iter_entries

faiss.omp_set_num_threads(1)  # keep faiss off the threads torch/OpenMP also use

# ---- artifact file names ----------------------------------------------------
F_PASSAGE_INDEX = "chunk.faiss"
F_PASSAGE_OWNER = "chunk_pages.npy"
F_PAGE_MATRIX = "page_vecs.npy"
F_PAGE_IDS = "page_ids.npy"
F_PAGE_TEXT = "page_texts.json"
F_LEX_POSTINGS = "bm25.npz"
F_LEX_VOCAB = "bm25_vocab.json"
F_META = "meta.json"

PQ_CODE_BYTES = 96           # PQ sub-quantizers -> ~bytes per passage vector
PAGE_TEXT_WORDS = 400        # words kept per page for the page/rerank signals
BM25_SATURATION = 1.5        # BM25 k1
BM25_LENGTH_NORM = 0.75      # BM25 b

_WORD_RE = re.compile(r"[a-z0-9]+")


def term_tokens(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; shared by the BM25 build and query time."""
    return _WORD_RE.findall(text.lower())


def capped_page_text(record: Dict) -> str:
    """Title+body for a page, trimmed to PAGE_TEXT_WORDS words."""
    return " ".join(entry_text(record).split()[:PAGE_TEXT_WORDS])


@dataclass
class CorpusIndex:
    page_ids: List[int]
    page_matrix: np.ndarray
    page_text: List[str]
    vocab: Dict[str, int]
    lex_ptr: np.ndarray
    lex_pages: np.ndarray
    lex_weights: np.ndarray
    passages: faiss.Index
    passage_owner: np.ndarray
    row_of: Dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.row_of = {pid: i for i, pid in enumerate(self.page_ids)}


def _fit_bm25(token_docs: List[List[str]]) -> Tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """Return (vocab, term_ptr, post_pages, post_weights) — a term-major CSR whose
    posting values already include IDF and length normalization, so query scoring
    reduces to summing the postings of the query's terms."""
    doc_count = len(token_docs)
    length = np.fromiter((len(d) for d in token_docs), dtype=np.float64, count=doc_count)
    avg_len = float(length.mean()) if doc_count else 0.0

    vocab: Dict[str, int] = {}
    doc_freq: List[int] = []
    counts_per_doc: List[Dict[int, int]] = []
    for tokens in token_docs:
        local: Dict[int, int] = {}
        for tok in tokens:
            tid = vocab.get(tok)
            if tid is None:
                tid = vocab[tok] = len(vocab)
                doc_freq.append(0)
            local[tid] = local.get(tid, 0) + 1
        counts_per_doc.append(local)
        for tid in local:
            doc_freq[tid] += 1

    df = np.asarray(doc_freq, dtype=np.float64)
    idf = np.log1p((doc_count - df + 0.5) / (df + 0.5))

    # bucket postings per term so the CSR comes out term-major
    buckets: List[List[Tuple[int, float]]] = [[] for _ in range(len(vocab))]
    for doc, counts in enumerate(counts_per_doc):
        length_factor = BM25_SATURATION * (
            1.0 - BM25_LENGTH_NORM + BM25_LENGTH_NORM * (length[doc] / avg_len if avg_len else 0.0))
        for tid, tf in counts.items():
            buckets[tid].append((doc, idf[tid] * tf * (BM25_SATURATION + 1.0) / (tf + length_factor)))

    ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
    pages: List[int] = []
    weights: List[float] = []
    for tid, bucket in enumerate(buckets):
        for doc, weight in bucket:
            pages.append(doc)
            weights.append(weight)
        ptr[tid + 1] = len(pages)

    return (vocab, ptr,
            np.asarray(pages, dtype=np.int32),
            np.asarray(weights, dtype=np.float32))


def build_index(*, entries_dir: Optional[Path] = None,
                artifacts_dir: Optional[Path] = None) -> None:
    """Build and persist every artifact from the corpus."""
    out = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    print(f"[build] {len(records)} pages", flush=True)

    # passage channel: PQ-compressed FAISS over sliding windows
    units: List[Passage] = passages_of(records)
    print(f"[build] {len(units)} passages; encoding", flush=True)
    unit_vecs = encode_texts([u.text for u in units], report_every=20000)
    width = int(unit_vecs.shape[1]) if unit_vecs.size else VECTOR_WIDTH
    pq = faiss.IndexPQ(width, PQ_CODE_BYTES, 8, faiss.METRIC_INNER_PRODUCT)
    if unit_vecs.size:
        pq.train(unit_vecs)
        pq.add(unit_vecs)
    faiss.write_index(pq, str(out / F_PASSAGE_INDEX))
    np.save(out / F_PASSAGE_OWNER,
            np.fromiter((u.page_id for u in units), dtype=np.int32, count=len(units)))

    # page channel: one vector + capped text per page
    page_ids = [int(r["page_id"]) for r in records]
    page_text = [capped_page_text(r) for r in records]
    print("[build] encoding pages", flush=True)
    page_matrix = encode_texts(page_text)
    if page_matrix.size == 0:
        page_matrix = np.zeros((0, width), dtype=np.float32)
    np.save(out / F_PAGE_MATRIX, page_matrix.astype(np.float32))
    np.save(out / F_PAGE_IDS, np.asarray(page_ids, dtype=np.int64))
    (out / F_PAGE_TEXT).write_text(json.dumps(page_text), encoding="utf-8")

    # lexical channel: page-level BM25
    print("[build] fitting BM25", flush=True)
    vocab, ptr, post_pages, post_weights = _fit_bm25([term_tokens(t) for t in page_text])
    np.savez(out / F_LEX_POSTINGS, term_ptr=ptr, post_pages=post_pages, post_weights=post_weights)
    (out / F_LEX_VOCAB).write_text(json.dumps(vocab), encoding="utf-8")

    (out / F_META).write_text(json.dumps({
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": width, "num_pages": len(page_ids),
        "num_chunks": len(units), "vocab_size": len(vocab), "pq_bytes": PQ_CODE_BYTES,
    }, indent=2), encoding="utf-8")
    print(f"[build] artifacts written to {out}", flush=True)


def load_index(artifacts_dir: Optional[Path] = None) -> CorpusIndex:
    """Load every artifact into a :class:`CorpusIndex`."""
    root = artifacts_dir or ARTIFACTS_DIR
    lex = np.load(root / F_LEX_POSTINGS)
    return CorpusIndex(
        page_ids=[int(x) for x in np.load(root / F_PAGE_IDS)],
        page_matrix=np.ascontiguousarray(np.load(root / F_PAGE_MATRIX), dtype=np.float32),
        page_text=json.loads((root / F_PAGE_TEXT).read_text(encoding="utf-8")),
        vocab=json.loads((root / F_LEX_VOCAB).read_text(encoding="utf-8")),
        lex_ptr=lex["term_ptr"], lex_pages=lex["post_pages"], lex_weights=lex["post_weights"],
        passages=faiss.read_index(str(root / F_PASSAGE_INDEX)),
        passage_owner=np.load(root / F_PASSAGE_OWNER),
    )
