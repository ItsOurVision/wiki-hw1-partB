# Section B — Wikipedia Retrieval Pipeline

**Authors:** Daniel Kats (207730854) · Noga Nagel (322586082)

End-to-end retrieval over ~27,000 Wikipedia-style pages. For each query,
`run(queries)` returns a ranked list of `page_id`s, scored by mean **NDCG@10**.

**Method in one line:** page-level BM25 proposes candidates, reciprocal rank
fusion of three signals (page-dense cosine, BM25, chunk-dense max-pooled to page)
ranks them, and a cross-encoder reranks the top candidates — *blended* with the
fusion score rather than replacing it.

## Pipeline

| stage | file | what it does |
|-------|------|--------------|
| chunk | `chunk.py` | Split each page into title-prefixed overlapping word windows (`TARGET_WORDS=180`, `OVERLAP_WORDS=40`) so a long page is not blurred into one vector. |
| embed | `embed.py` | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized 384-d vectors (inner product = cosine). |
| index | `index.py` | Offline build of all retrieval artifacts (PQ chunk index, page vectors, page-level BM25, page texts). |
| retrieve | `retrieve.py` | Query-time hybrid retrieval + cross-encoder rerank (the timed path). |

### Retrieval detail (`retrieve.py`)

1. **Candidates** — top `LEX_POOL=100` pages by page-level BM25 (falls back to
   page-dense if a query has no lexical hit), widened with the best pages from
   the chunk index.
2. **Fusion** — reciprocal rank fusion (`RRF_K=60`) of the three signals
   (page-dense cosine, BM25, best-passage cosine). RRF blends by rank position,
   so it needs no per-signal weights — nothing to overfit on the public set.
3. **Cross-encoder rerank** — the top `RERANK_DEPTH=12` are scored by
   `cross-encoder/ms-marco-MiniLM-L-6-v2` and **blended** with the fusion score
   (`RERANK_WEIGHT=0.85·CE + 0.15·fusion`). Blending (vs. letting the CE fully
   replace the fusion order) was more robust in our experiments. If the
   cross-encoder cannot be loaded at runtime, retrieval falls back to the fusion
   ranking instead of failing.

## Artifacts (`artifacts/`, loaded by `run()` — never rebuilt at grading)

| file | content | format |
|------|---------|--------|
| `chunk.faiss` | chunk-level dense vectors, product-quantized | FAISS `IndexPQ` (m=96), ~42 MB |
| `chunk_pages.npy` | chunk-row → page_id map | int32 `(num_chunks,)` |
| `page_vecs.npy` | one dense vector per page | float32 `(num_pages, 384)` |
| `page_ids.npy` | page_ids aligned to `page_vecs.npy` / BM25 rows | int64 `(num_pages,)` |
| `page_texts.json` | per-page title+content (≤400 words), cross-encoder input | JSON list |
| `bm25.npz` | page-level BM25 postings (precomputed weights, CSR-by-term) | npz |
| `bm25_vocab.json` | term → term_id map | JSON |
| `meta.json` | build counts and parameters | JSON |

**No Git LFS required** — every artifact is under 100 MB (the chunk index is
product-quantized for exactly this reason), so a plain `git clone` yields a
ready-to-run repo.

## Setup

```bash
pip install -r requirements.txt
```

Pretrained MiniLM / cross-encoder weights are downloaded from the Hugging Face
hub on first use; they are not shipped in the repo.

## Run the public self-test (no rebuild needed)

A fresh clone already contains `artifacts/`, so the evaluation runs directly:

```bash
python scripts/eval_public.py        # prints mean NDCG@10 on the public queries
```

## Rebuild the index (offline, only on your own machine — not timed, not run by staff)

The corpus (`data/Wikipedia Entries/`) is **not** committed (it is part of the
course handout and is not needed at query time). To rebuild artifacts from
scratch, place the corpus there and run:

```bash
python scripts/build_index.py        # embeds the corpus and writes artifacts/
```

## Empirical results (public queries)

Mean NDCG@10 on the public set, measured with `scripts/ablation.py` /
`scripts/sweep.py`:

| configuration | NDCG@10 |
|---|---|
| dense only | 0.369 |
| − cross-encoder | 0.383 |
| − chunk channel | 0.381 |
| full pipeline, weighted min-max fusion | 0.445 |
| full pipeline, reciprocal rank fusion | **0.448** |

Each channel contributes: removing the cross-encoder or the chunk channel each
costs ~0.05 NDCG. Reciprocal rank fusion edged out a tuned weighted blend while
needing no weights to tune. The product-quantized chunk index matches the flat
index's quality at ~1/15th the size. The full query batch runs in a few seconds
(well under the 60 s limit).

## Video

Presentation: _link to be added_.
