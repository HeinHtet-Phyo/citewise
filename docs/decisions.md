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

---

## ADR-004 — Gemini 2.0 Flash for generation

**Date:** 19 November 2026  
**Status:** Implemented

### Context

The project budget is Gemini free-tier only. We need a model that:
1. Follows precise citation-format instructions (uses `[n]` markers reliably)
2. Has a long enough context window for 6 × 400-token chunks + system prompt (~3,000 tokens input)
3. Is fast enough for interactive use

### Decision

Use `gemini/gemini-2.0-flash` via LiteLLM. Temperature = 0 for reproducible, factual answers.
Nebius models (Llama-3 variants) are reserved for a future cost/quality comparison on the eval set.

### Trade-offs

- Free tier rate limits may cause retries; backed off with tenacity.
- The file-based LLM cache (keyed on model + full prompt SHA-256) means the same question costs £0 on subsequent runs during development.

---

## ADR-005 — LLM-as-judge evaluation methodology

**Date:** 19 November 2026  
**Status:** Implemented

### Context

For a RAG system over factual government guidance, three dimensions matter:
- **Faithfulness:** does the answer stay within the retrieved sources?
- **Relevance:** does the answer address the question?
- **Citation quality:** are the `[n]` markers correctly placed?

We have no gold-standard reference answers for 150 questions, so we cannot use exact-match or ROUGE metrics.

### Decision

Use Gemini 2.0 Flash as an LLM judge (`eval/judge.py`) with a rubric that scores each dimension 0–3. The judge prompt instructs the model to output JSON only, which is parsed and stored.

The judge re-uses the same LLM cache as the pipeline — repeated scoring of the same question costs nothing.

**Dev/test split is sacred:** threshold and config selection are based only on dev-split scores; test-split scores are run once at the end for the final reported numbers.

### Alternatives considered

- **Human annotation:** More reliable but impractical for 150 × 5 configs = 750 evaluations.
- **RAGAS framework:** Adds complexity and ties us to its scoring logic; a custom judge is more transparent and interview-explainable.
- **Retrieval MRR/NDCG:** Would require labelled relevance judgements per question, which we don't have.

---

## ADR-006 — Refusal via evidence threshold

**Date:** 19 November 2026  
**Status:** Implemented (threshold = 0.0 pending dev-split tuning)

### Context

The corpus covers only four topic areas. A question about, say, immigration policy or tax returns will retrieve low-scoring chunks and produce a hallucinated or irrelevant answer.

### Decision

After retrieval, check the top candidate's score against `RagConfig.refusal_threshold`. If the score falls below the threshold, return a refused `Answer` with `refusal_reason = "low_evidence"` before calling the LLM.

The threshold starts at 0.0 (no refusals) and will be tuned on the dev split in Phase 5 to maximise the F1 of correct answers vs unnecessary refusals.

As a secondary safeguard, if the LLM itself returns the phrase "I cannot find sufficient information…" (from the system prompt instruction), the answer is marked as refused post-hoc.
