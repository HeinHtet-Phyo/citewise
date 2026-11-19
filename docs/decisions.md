# Design decisions

## ADR-001 — Section-merge chunking strategy

**Date:** 18 November 2026  
**Status:** Implemented

### Context

GOV.UK pages are structured as short, focused sections (avg ~120 tokens each). A naïve sentence-splitting chunker treats every section independently, so both the 400- and 800-token size settings produce nearly identical distributions (avg 126 tokens in both cases). This makes the chunk-size ablation experiment meaningless.

### Decision

Greedily merge adjacent sections from the same guide part before sentence-splitting. Each merge group is bounded by:

1. **Token budget:** stop merging when adding the next section would exceed `max_tokens − title_prefix`.
2. **URL boundary:** sections from different guide parts (e.g. `/statutory-sick-pay/overview` vs `/statutory-sick-pay/what-you-ll-get`) are never merged, so every chunk cites the correct anchor URL.

Section headings are kept inline as `## Heading` markers so the LLM sees the structural context.

If a single section alone exceeds the token budget (a long table or prose block), the fallback sentence-splitter handles it with overlap.

### Results (full corpus, 3,302 pages)

| Chunk size | Chunks | Avg tokens |
|------------|--------|-----------|
| 400 | 21,740 | 282 |
| 800 | 12,311 | 486 |

Ratio: 1.72× — the two sizes now differ enough for the ablation to be informative.

### Alternatives considered

- **Fixed-overlap sentence splitting only:** produces avg 126 tokens at both size settings; the sizes converge because most GOV.UK sections are under 200 tokens.
- **Sliding window across all sections ignoring URLs:** increases avg tokens further but would merge content from different guide parts, breaking citation accuracy.
- **Recursive character splitter (LangChain-style):** does not respect GOV.UK section boundaries, producing chunks that mix headings from different sections.

---

## ADR-002 — Hybrid search (dense + BM25) in Qdrant

**Date:** [TBC — set when Phase 2 retrieval is benchmarked]  
**Status:** Planned

### Context

UK government guidance uses precise legal/policy terminology (e.g. "Statutory Maternity Pay", "Council Tax Reduction", "right to work"). Dense-only retrieval can miss exact phrase matches; sparse (BM25) is strong on exact terms but misses paraphrases.

### Decision

Use Qdrant's hybrid search: dense vectors from BAAI/bge-small-en-v1.5 (384 dim) fused with BM25 sparse vectors from fastembed, combined via Reciprocal Rank Fusion (RRF). A cross-encoder reranker (ms-marco-MiniLM-L-6-v2) rescores the top-k before passing to the LLM.

### Results

[TBC — retrieval sanity check results to be added after Phase 2]

---

## ADR-003 — GOV.UK taxonomy UUID filters

**Date:** 18 November 2026  
**Status:** Implemented

### Context

The GOV.UK Search API `filter_part_of_taxonomy_tree` parameter requires content IDs (UUIDs), not human-readable slugs. Using slugs returns 0 results with no error.

### Decision

Look up the content ID for each topic via the Content API (`/api/content/<path>`) and hard-code the four UUIDs in `discover.py`. The IDs are stable; GOV.UK does not reassign content IDs.

| Topic | UUID |
|-------|------|
| welfare/benefits | `dded88e2-f92e-424f-b73e-6ad24a839c51` |
| work/employment | `d0f1e5a3-c8f4-4780-8678-994f19104b21` |
| housing | `4794066e-e3cc-425e-8cc4-e7ff3edb4c39` |
| transport/driving | `a4038b29-b332-4f13-98b1-1c9709e216bc` |
