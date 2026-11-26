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

**Date:** 19 November 2026  
**Status:** Implemented

### Context

UK government guidance uses precise legal/policy terminology (e.g. "Statutory Maternity Pay", "Council Tax Reduction", "right to work"). Dense-only retrieval can miss exact phrase matches; sparse (BM25) is strong on exact terms but misses paraphrases.

### Decision

Use Qdrant's hybrid search: dense vectors from BAAI/bge-small-en-v1.5 (384 dim) fused with BM25 sparse vectors from fastembed, combined via Reciprocal Rank Fusion (RRF). A cross-encoder reranker (ms-marco-MiniLM-L-6-v2) rescores the top-k before passing to the LLM.

### Results

LLM-as-judge eval on 100 dev questions (19 November 2026):

| Config | Faithfulness | Relevance | Citation | Composite |
|--------|-------------|-----------|----------|-----------|
| dense_400 | 1.26 | 2.70 | 1.36 | 1.77 |
| hybrid_400 | 1.11 | 2.70 | 1.23 | 1.68 |
| **hybrid_400_rerank** | **1.25** | **2.67** | **1.42** | **1.78** |
| hybrid_800 | 1.05 | 2.80 | 1.15 | 1.67 |
| hybrid_800_rerank | 1.02 | 2.76 | 1.10 | 1.63 |

`hybrid_400_rerank` scored highest on composite. Scores are on a 0–3 scale.
Full results in `eval/results/summary_dev.json` and the README evaluation table.

Notably, 800-token chunks underperformed 400-token chunks despite including more context — the
larger chunks contained more irrelevant text per snippet (citations truncated to 300 chars), which
hurt faithfulness and citation quality scores.

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
called directly through the openai SDK with `base_url=https://api.tokenfactory.nebius.com/v1`.

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
The openai SDK does not carry pricing tables; cost is computed from `response.usage` token counts
against the manual pricing dict in `generate.py`.

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

> **Superseded (20 November 2026):** the 0-3 scoring scale and the 150-question set described here were replaced by a percentage-based judge and a rebuilt 190-question set with ground truth. See `docs/eval_method.md`. The text below is kept as history.

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

> **Status:** the 0.30 threshold for `hybrid_400_rerank` was re-checked on the rebuilt dev set (`eval/results/threshold_hybrid_400_rerank.json`) and still holds. Older figures below use the previous question set.

**Date:** 19 November 2026  
**Status:** Implemented (threshold = 0.0 pending dev-split tuning)

### Context

The corpus covers only four topic areas. A question about, say, immigration policy or tax returns will retrieve low-scoring chunks and produce a hallucinated or irrelevant answer.

### Decision

After retrieval, check the top candidate's score against `RagConfig.refusal_threshold`. If the score falls below the threshold, return a refused `Answer` with `refusal_reason = "low_evidence"` before calling the LLM.

The threshold was tuned on the dev split (100 questions, hybrid_400_rerank config) by sweeping
values from 0.00 to 1.00 in steps of 0.05. Results written to
`eval/results/threshold_hybrid_400_rerank.json`.

**Tuned value: 0.30** — raises the composite from 1.633 (threshold=0.0) to 1.635 by refusing
4 questions instead of 2. The marginal improvement confirms that all dev questions are within
the corpus scope; the threshold's main value is protecting against out-of-corpus production queries.

As a secondary safeguard, if the LLM itself returns the phrase "I cannot find sufficient information…" (from the system prompt instruction), the answer is marked as refused post-hoc.


## ADR-007 — Parametric leakage: prompt, claim verifier, small-to-big context

**Date:** 20 November 2026  
**Status:** Implemented; benefit not established

### Context

Early answers sometimes stated figures from the model's own memory and cited a chunk that did not contain them ("parametric leakage"). The judge's made-up rate measures this.

### Decision

Add three fixes and measure each on dev as a ladder (`eval/generate_candidates.py`):

- B: `answer_v2` prompt (figures only if written in the sources).
- C: B plus a claim verifier (`guardrails.verify_claims`): a small model checks each sentence, the answer is regenerated once with feedback, still-unsupported sentences are dropped, and if nothing is left the answer is refused.
- D: C plus small-to-big context (neighbouring chunks from the same page, within a token budget).
- E: D with 10 context chunks instead of 6.

### Results (dev, 47 answerable questions; see `eval/results/summary.md`)

Made-up rate: A 6.5%, B 11.1%, C 4.5%, D 7.0%, E 4.8%. Correct (of answerable): A 80.9%, B 76.1%, C 70.2%, D 63.8%, E 66.0%. False refusals rose from 2.1% (A) to 6.4% (C), 8.5% (D) and 10.6% (E).

### Consequences

- The fixes bought a small made-up reduction (one or two answers) at a clear cost in correctness and false refusals, so small-to-big and k=10 were not adopted.
- On the test split the baseline's made-up rate was 20.0% against 6.5% on dev, so the dev estimate was noisy. For the dense config, test shows 8.3% (plain) vs 6.5% (with prompt and verifier), with no gain in faithfulness. The fixes are not shown to help.

## ADR-008 — Deployed config: `L_dense`

**Date:** 20 November 2026  
**Status:** Implemented

### Context

Render's free tier has 512 MB. Measured in Docker (`eval/results/memory.json`, `scripts/measure_memory.py`): `hybrid_400_rerank` peaks at 1,945 MiB over 10 requests; the dense configs now peak at **221 MiB** (idle ~247 MiB) after replacing LiteLLM with the openai SDK (see ADR-010).

### Decision

Deploy `L_dense` (dense search plus `answer_v2` and the claim verifier), chosen on dev as the best dense variant. Models are baked into the image so the first question does not download them. A 10 requests per minute per-IP limit and a daily £0.50 spend cap protect the public demo.

### Consequences

The deployed config has its own test numbers (README). Retrieval is weaker than hybrid with reranking (hit rate 89.8% vs 92.9% on test) and the dense config has lower refusal recall (93.3% vs 96.7%).

## ADR-009 — Refusal fix: `L2_dense`

Live spot checks showed common questions being refused. Diagnosis (20 questions, step by step): retrieval scores were high and the threshold is 0; refusals came from the LLM (two corpus gaps and one over-strict prompt) and from the claim verifier flagging "the guidance does not cover" sentences. Decision: answer prompt v3 (partial answers allowed) and a verifier filter for limitation sentences, deployed as `L2_dense`. Dev: refusal recall stayed 100.0% (n=15); correctness equal at 72.3%. Test (run once): recall 96.7%, correct 65.3% (down from 69.4%), made-up 7.4%. Consequence: fewer visible refusals on real questions at the price of slightly lower measured correctness on test; both numbers are in the README.

## ADR-011 — Payload field preference: `updated_at` over `public_updated_at`

**Date:** 26 November 2026  
**Status:** Implemented

GOV.UK's `public_updated_at` field is frequently frozen at an old date (e.g. 2015-03-09 for Statutory Sick Pay) even when page content is updated; `updated_at` reflects the actual CMS edit time. `clean.py` now prefers `updated_at` with a fallback to `public_updated_at` for pages that carry only the latter field.

**Incident note:** a backfill script (`fix_updated_at.py`) initially used `qdrant_client.overwrite_payload()`, which replaces the *entire* payload with the supplied dict — destroying all other fields (text, url, tokens, etc.) and breaking the live pipeline. The bug was caught by testing the live site (KeyError on retrieval). Fixed by switching to `set_payload()` (partial update), then re-indexing from regenerated JSONL.

---

## ADR-010 — Replace LiteLLM with openai SDK to fix OOM on Render

**Date:** 26 November 2026  
**Status:** Implemented

### Context

After launch the live service crashed in a loop with "Ran out of memory (used over 512 MB)". Local measurements had shown ~378 MiB for `L2_dense`, leaving ~134 MiB headroom, but those measurements were taken on a Mac where ONNX Runtime saw 10 CPUs and allocated proportionally. Reproducing with `docker run --memory=512m` confirmed: the first `/ask` call reached 492 MiB and the container was OOM-killed.

Investigation:
- Thread limits (`OMP_NUM_THREADS=1`, `ONNXRUNTIME_MAX_INTEROP_THREADS=1`, `fastembed threads=1`) reduced peak to 459 MiB.
- ONNX arena strategy `kSameAsRequested` saved a further ~2 MiB.
- Neither was enough. Profiling found **LiteLLM** (175 MB on disk, ~240 MiB resident) as the dominant consumer.

### Decision

Replace LiteLLM with the openai SDK (28 MB). Nebius exposes an OpenAI-compatible endpoint, so the change was mechanical: `litellm.completion(model=..., base_url=..., ...)` → `OpenAI(api_key=..., base_url=...).chat.completions.create(model=..., ...)`. Cost is computed from the manual pricing table already present in `generate.py`. LiteLLM moved to `[dev]` optional extras so `eval/judge.py` (which still uses it) and its tests continue to work.

### Consequences

Peak RSS under `--memory=512m` across 10 questions: **492 MiB (OOM) → 221 MiB** (57% reduction). The 512 MB Render limit now has a 291 MiB safety margin. No change to answers, eval numbers, or the external API.
