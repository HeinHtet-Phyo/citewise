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

## ADR-004 — Qwen3-235B-A22B for generation (Nebius)

**Date:** 19 November 2026  
**Status:** Implemented

### Context

The original plan used Gemini 2.0 Flash (free tier). By November 2026 the model was
deprecated and its replacement (`gemini-3.8-flash`) is limited to **20 requests per day**
on the free tier — insufficient for a 500-question eval run.

We need a model that:
1. Follows precise citation-format instructions (uses `[n]` markers reliably)
2. Has a long enough context window for 6 × 400-token chunks + system prompt (~3,000 tokens)
3. Is affordable within the $20 Nebius hackathon credit budget

### Decision

Use `Qwen/Qwen3-235B-A22B-Instruct-2507` via the Nebius Token Factory API (OpenAI-compatible),
routed through LiteLLM with `base_url=https://api.tokenfactory.nebius.com/v1`.

Qwen3-235B-A22B is a Mixture-of-Experts model: 235B total parameters, 22B active per forward
pass. This gives near-frontier quality at a cost comparable to a dense 22B model.

Temperature = 0 for reproducible, factual answers. `enable_thinking=False` is passed in
`extra_body` to suppress Qwen3's extended-reasoning mode; `<think>` blocks are also stripped
from the output as a defensive fallback.

**Estimated cost per eval run:**
- 500 dev generation calls × avg 3,500 tokens = ~$0.54
- 50 test generation calls × avg 3,500 tokens = ~$0.054
- Judge calls (DeepSeek-V4.1-Flash) = ~$0.21
- **Total: ~$0.80 per full run**

Pricing source: Nebius Token Factory, November 2026 — $0.20/M input, $0.60/M output.
LiteLLM does not have Nebius rates built in; a manual fallback in `generate.py` computes
cost from `response.usage` token counts.

### Trade-offs

- Self-hosted/third-party model: no Google data-use concerns, but requires a valid API key.
- Nebius free-tier hackathon credit ($120) comfortably covers the full eval suite.
- The file-based LLM cache (SHA-256 of model + prompt) means re-runs cost nothing once cached.

### Alternatives considered

- **Gemini free tier:** deprecated models, 20 RPD hard limit — cannot run a 500-question eval.
- **Qwen3-30B-A3B (3B active):** cheaper but weaker instruction-following on citation format.
- **Llama-3.1-70B:** not available on Nebius Token Factory as of November 2026.

---

## ADR-005 — LLM-as-judge evaluation methodology

**Date:** 19 November 2026  
**Status:** Implemented

### Context

For a RAG system over factual government guidance, three dimensions matter:
- **Faithfulness:** does the answer stay within the retrieved sources?
- **Relevance:** does the answer address the question?
- **Citation quality:** are the `[n]` markers correctly placed?

We have no gold-standard reference answers for 150 questions, so we cannot use exact-match
or ROUGE metrics.

### Decision

Use `deepseek-ai/DeepSeek-V4.1-Flash-1017` (Nebius) as the LLM judge, scoring each dimension
0–3. DeepSeek is a **different model family** from the Qwen3 generator, reducing self-preference
bias. The judge prompt instructs the model to output JSON only.

The judge re-uses the same LLM cache as the pipeline — repeated scoring of the same question
costs nothing.

**Dev/test split is sacred:** threshold and config selection are based only on dev-split
scores; test-split scores are run once at the end for the final reported numbers.

### Limitation

Using an LLM judge from the same provider (Nebius) as the generator introduces a potential
shared-infrastructure bias, but the different model family (DeepSeek vs Qwen) is the
meaningful independence property for avoiding self-preference. The eval measures *relative*
differences between retrieval configs with the same generator, so within-provider effects
cancel out across configs.

### Alternatives considered

- **Gemini as judge:** deprecated/quota-exhausted; cannot support 750 judgements per run.
- **Human annotation:** More reliable but impractical for 150 × 5 configs = 750 evaluations.
- **RAGAS framework:** Adds complexity and ties us to its scoring logic; a custom judge is
  more transparent and interview-explainable.
- **Retrieval MRR/NDCG:** Would require labelled relevance judgements per question.

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
