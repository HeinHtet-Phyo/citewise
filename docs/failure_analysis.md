# Failure analysis

_Analysis of failure modes observed in the dev-split eval run. Updated after eval results are available._

## Methodology

Failures are drawn from `eval/results/hybrid_400_rerank_dev.jsonl` — the best-performing config on the dev split. A "failure" is defined as any answer that scores < 2 on any judge dimension, plus all refused answers where the question was answerable.

Each case is annotated with:
- **Root cause** — what went wrong in the pipeline
- **Stage** — where the failure originates (retrieval / chunking / generation / guardrails)
- **Fix direction** — what change would address it

---

## Cases

_[TBC — will be populated after eval run completes. The structure below shows the intended format.]_

### Case 1 — Retrieval miss on government terminology

**Question:** [TBC]  
**Expected:** [TBC]  
**Actual answer:** [TBC]  
**Judge scores:** Faithfulness [TBC] | Relevance [TBC] | Citation [TBC]

**Root cause:** The query uses everyday language ("sick pay") while the GOV.UK page uses the full legal term ("Statutory Sick Pay"). BM25 misses the match; dense retrieval finds a semantically close but incorrect page.

**Stage:** Retrieval  
**Fix direction:** Query expansion or synonym injection before retrieval.

---

### Case 2 — Cross-part merge confusion

**Question:** [TBC]  
**Expected:** [TBC]  
**Actual answer:** [TBC]  
**Judge scores:** Faithfulness [TBC] | Relevance [TBC] | Citation [TBC]

**Root cause:** A multi-part guide has the eligibility criteria in one part and the rates in another. The reranker selects the rates chunk; the answer misses the eligibility question entirely.

**Stage:** Retrieval (top-k too small)  
**Fix direction:** Increase `top_k_context` from 6 to 8 for multi-answer questions.

---

### Case 3 — Hallucinated figure

**Question:** [TBC]  
**Expected:** [TBC]  
**Actual answer:** [TBC]  
**Judge scores:** Faithfulness [TBC] | Relevance [TBC] | Citation [TBC]

**Root cause:** The retrieved chunk contains a table with multiple weekly rates. The LLM selects the wrong row and states an incorrect figure with a citation that does exist — but to the wrong part of the table.

**Stage:** Generation  
**Fix direction:** Table-aware chunking that includes row headers with every cell value.

---

### Case 4 — Stale guidance cited

**Question:** [TBC]  
**Expected:** [TBC]  
**Actual answer:** [TBC]  
**Judge scores:** Faithfulness [TBC] | Relevance [TBC] | Citation [TBC]

**Root cause:** The index was built on a corpus snapshot; the GOV.UK page has since been updated. The answer cites the old figure.

**Stage:** Data freshness  
**Fix direction:** The incremental refresh pipeline (`refresh.py`) addresses this; run `make ingest` monthly.

---

### Case 5 — Low evidence refusal on answerable question

**Question:** [TBC]  
**Expected:** An answer exists in the corpus  
**Actual answer:** Refused with `low_evidence`  
**Judge scores:** N/A (refused)

**Root cause:** The question uses a less common phrasing that scores below the refusal threshold, even though relevant chunks exist.

**Stage:** Guardrails (refusal threshold)  
**Fix direction:** Tune threshold lower, or add query rewriting before retrieval.

---

### Case 6 — Citation placed on wrong sentence

**Question:** [TBC]  
**Expected:** Citation [1] on eligibility claim  
**Actual answer:** Citation [1] placed on a general introductory sentence  
**Judge scores:** Faithfulness [TBC] | Relevance [TBC] | Citation [TBC]

**Root cause:** The LLM places the citation on the sentence immediately before the specific claim rather than on the claim itself.

**Stage:** Generation  
**Fix direction:** Add an explicit instruction in the system prompt: "Place the citation [n] immediately after the specific claim it supports, not at the end of the paragraph."

---

### Case 7 — Out-of-scope question answered (should refuse)

**Question:** [TBC]  
**Expected:** Refused (topic not in corpus)  
**Actual answer:** A plausible but unsupported answer  
**Judge scores:** Faithfulness 0 | Relevance [TBC] | Citation [TBC]

**Root cause:** The question is loosely related to a corpus topic. BM25 returns chunks containing partial keyword matches; the LLM synthesises an answer from weakly related guidance.

**Stage:** Retrieval + guardrails  
**Fix direction:** Tune refusal threshold upward; consider a topic-classification pre-filter.

---

### Case 8 — Verbatim extraction instead of synthesis

**Question:** [TBC]  
**Expected:** A synthesised explanation across multiple sources  
**Actual answer:** A direct copy of a single source chunk with minimal rewording  
**Judge scores:** Faithfulness [TBC] | Relevance 1 | Citation [TBC]

**Root cause:** The question requires combining information from two separate guide parts. Only one part scores above the reranker threshold, so the LLM only sees one chunk.

**Stage:** Retrieval  
**Fix direction:** Diversity-aware retrieval: ensure at least one chunk per distinct guide part is included.

---

## Summary table

_[TBC — after eval run]_

| Case | Stage | Failure type | Fix direction |
|------|-------|-------------|--------------|
| 1 | Retrieval | Term mismatch | Query expansion |
| 2 | Retrieval | Multi-part miss | Larger top_k |
| 3 | Generation | Table hallucination | Table-aware chunking |
| 4 | Data | Stale guidance | Incremental refresh |
| 5 | Guardrails | Over-refusal | Threshold tuning |
| 6 | Generation | Citation placement | Prompt instruction |
| 7 | Guardrails | Under-refusal | Threshold tuning |
| 8 | Retrieval | Diversity failure | MMR retrieval |
