# Failure analysis

Analysis of failure modes observed in `hybrid_400_rerank` dev-split results (100 questions).
A "failure" is any answer scoring < 2 on any judge dimension, plus LLM-initiated refusals
on questions the corpus can answer.

## Methodology

- **Config:** `hybrid_400_rerank` — hybrid BM25 + dense search, cross-encoder reranker,
  400-token chunks, 6 chunks in context
- **Judge:** DeepSeek-V4.1-Flash, scoring faithfulness / relevance / citation quality 0–3
- **Failure threshold:** composite score < 2.0 OR refused with reason `low_evidence`
- **Data source:** `eval/results/hybrid_400_rerank_dev.jsonl`

### Failure distribution (100 dev questions)

| Failure type | Count | % |
|---|---|---|
| Parametric leakage (faith=0, rel≥2) | 64 | 64% |
| LLM over-refusal on answerable question | 10 | 10% |
| Retrieval mismatch (faith≤1, rel≤1) | 2 | 2% |
| Passing answers (composite ≥ 2.0) | 24 | 24% |

---

## Pattern A — Parametric leakage (64 cases)

The dominant failure. Qwen3 retrieves the correct GOV.UK page (relevance=3) but the
specific numeric or policy detail is in a different section or chunk. The model answers
from training-data knowledge, cites a retrieved chunk, and receives faithfulness=0 because
the cited snippet does not contain the stated figure.

---

### Case 1 — UC standard allowance rate not in retrieved chunk

**Question:** What is the Universal Credit standard allowance for a single person over 25?  
**Actual answer:** "The Universal Credit standard allowance for a single person over 25 is £424.90 per month [1]."  
**Judge scores:** Faithfulness 0 | Relevance 3 | Citation 0  
**Judge rationale:** "The answer gives a specific monetary amount not supported by any provided source, and the sole citation [1] does not contain that figure."

**Retrieved chunk [1] snippet:**
> "Universal Credit — What you'll get: Universal Credit is paid monthly. How much you get depends on: your standard allowance; an…"

**Root cause:** The chunk retrieved is the *introduction* of the "What you'll get" page. The actual allowance table (listing £344.00 / £424.90 by age and household) is in a later section of the same page that was not included in the top-6 chunks.

**Stage:** Retrieval — the relevant fact is on the correct page but in an un-retrieved section.  
**Fix direction:** Parent-child chunking: retrieve the fact-level chunk but include the surrounding table as context, or increase `top_k_context` from 6 to 10.

---

### Case 2 — UC taper rate figure not in any retrieved chunk

**Question:** How does the Universal Credit taper rate work?  
**Actual answer:** "…for every £1 a claimant earns, their Universal Credit payment is reduced by 55p [1]."  
**Judge scores:** Faithfulness 0 | Relevance 3 | Citation 0  
**Judge rationale:** "The answer correctly explains the taper rate concept but all factual claims are unsupported by the provided sources."

**Retrieved chunk [1] snippet:**
> "Universal Credit and employers: Universal Credit is a single monthly payment for people…"

**Root cause:** The hybrid query matched employer-focused pages rather than the claimant-facing "if you're working" section that contains the taper rate. The 55p figure comes from Qwen3's training knowledge, not any retrieved chunk.

**Stage:** Retrieval — query semantics matched employer pages; the taper rate is explained on a different section of the UC guidance.  
**Fix direction:** Query rewriting or expansion to target the correct GOV.UK section more precisely.

---

### Case 3 — Statutory redundancy weekly pay cap not in retrieved chunk

**Question:** What is the maximum amount of a week's pay for statutory redundancy calculations?  
**Actual answer:** "The maximum amount of a week's pay for statutory redundancy calculations is £751 [2]. This applies to employees made redundant on or after 6 April 2026 [3]."  
**Judge scores:** Faithfulness 0 | Relevance 3 | Citation 0  
**Judge rationale:** "The answer states a specific maximum of £751 and an effective date of 6 April 2026, but neither claim is supported by the provided sources."

**Retrieved chunk [2] snippet:**
> "Making staff redundant — Redundancy pay: Employees you make redundant might be entitled to redund…"

**Root cause:** The cap figures for statutory redundancy pay are published in a dedicated table on GOV.UK. The retrieved chunks contain the section introductions ("you might be entitled to…") but not the numerical table rows (the cap was £643 in 2024, £719 in 2025, £751 in 2026). The model cites a chunk that links to but does not contain the figure.

**Stage:** Retrieval (table content not surfaced by chunk boundaries).  
**Fix direction:** Table-aware chunking that includes row headers with every cell; or add a post-processing step to include numeric rows explicitly.

---

### Case 4 — JSA entitlement duration not in retrieved chunk

**Question:** How long can I claim contribution-based Jobseeker's Allowance?  
**Actual answer:** "You can claim New Style Jobseeker's Allowance…for up to 182 days [1]. After this period, your work coach will discuss your options with you [1]."  
**Judge scores:** Faithfulness 0 | Relevance 3 | Citation 0  
**Judge rationale:** "The 182-day limit and work coach discussion are not supported by the cited source [1] or any provided passage."

**Retrieved chunk [1] snippet:**
> "New Style Jobseeker's Allowance — Overview: If you're unemployed or work less than 16…"

**Root cause:** The overview chunk introduces JSA without stating the duration. The "How long you can claim" detail (182 days / 26 weeks) is a separate sub-section not included in the top-6 results. Qwen3 knows this figure from training data and reports it with a citation to the overview page.

**Stage:** Retrieval — sub-section with the specific duration was not ranked in the top 6.  
**Fix direction:** Increase `top_k_context`; or index section headings as metadata for filtering.

---

## Pattern B — LLM over-refusal (10 cases)

Qwen3 responds with "I cannot find sufficient information in the guidance…" even though
6 relevant chunks were retrieved. The evidence guardrail (`refusal_threshold = 0.0`) did
not trigger; the LLM itself decided the evidence was insufficient.

---

### Case 5 — JSA part-time work: LLM refuses an answerable question

**Question:** Can I get Jobseeker's Allowance while doing part-time work?  
**Actual answer:** "I cannot find sufficient information in the guidance to answer this question."  
**Judge scores:** Faithfulness 3 | Relevance 1 | Citation 3  
(Faithfulness = 3 because the refusal made no false claims; relevance = 1 because the question was not addressed.)

**Root cause:** The retrieved chunks cover JSA eligibility and the application process. The specific "less than 16 hours" rule is mentioned in the overview ("work less than 16 hours a week on average") but Qwen3's system prompt instructs it to only answer from "the sources provided", and the overview chunk stops before giving the full rule. The model treats the truncated snippet as insufficient and refuses.

**Stage:** Generation — the model is overly cautious when chunks are truncated to 300-character snippets.  
**Fix direction:** Increase citation snippet length from 300 to 500 characters; or modify the system prompt to allow synthesis across multiple partial snippets.

---

### Case 6 — Shared Parental Leave: LLM refuses despite relevant chunks

**Question:** What is Statutory Shared Parental Leave?  
**Actual answer (refused):** `low_evidence`  
**Chunks retrieved:** 6

**Root cause:** Shared Parental Leave is covered in the GOV.UK corpus, but it is a complex topic spread across multiple pages. The retrieved chunks each cover a different aspect (eligibility, curtailment, pay). No single chunk gives a concise definition, so the model — instructed to give evidence-grounded answers — declines to synthesise.

**Stage:** Generation / prompting — the system prompt does not encourage multi-chunk synthesis strongly enough.  
**Fix direction:** Add a prompt instruction: "If the sources collectively answer the question (even if no single source is complete), synthesise an answer across all sources."

---

## Pattern C — Retrieval mismatch (2 cases)

Both faithfulness and relevance are low, indicating the retrieved chunks are not about
the question topic.

---

### Case 7 — State Pension qualifying age: key fact on un-retrieved page

**Question:** What is the qualifying age for State Pension?  
**Actual answer:** "The State Pension age varies depending on the individual and when they were born. The sources do not provide a single fixed age…"  
**Judge scores:** Faithfulness 1 | Relevance 1 | Citation 1  
**Judge rationale:** "The answer partially addresses the question but includes unsupported claims about birth-date eligibility and gradual increases, and one citation is mismatched while the exact age is not provided."

**Retrieved chunk [1] snippet:**
> "Home Responsibilities Protection — What you'll get: The amount of State Pension you may…"
> 
**Root cause:** The query matched pages about historical NI contributions (Home Responsibilities Protection, contracted-out pensions) rather than the main "State Pension age" checker page. The standard answer (currently 66, rising to 67 by 2028) is on a page that scored below the top-30 retrieved candidates.

**Stage:** Retrieval — the most relevant page ("Check your State Pension age") was not ranked in the top 30. The historical/NI pages share vocabulary but not the core fact.  
**Fix direction:** Add topic-based filtering or a classifier step to prefer "current guidance" pages over historical/supplementary pages.

---

### Case 8 — Carer's Allowance earnings limit: hallucinated figure with partial retraction

**Question:** What is the earnings limit for Carer's Allowance?  
**Actual answer:** "…if you earn more than £160 a week (after deductions), you will not be eligible for Carer's Allowance [not explicitly stated in provided sources]. I cannot find sufficient information in the guidance to answer this question."  
**Judge scores:** Faithfulness 1 | Relevance 1 | Citation 1  
**Judge rationale:** "The answer correctly cites source [1] for how earnings are calculated, but it introduces an unsupported £160 earnings limit and then refuses."

**Retrieved chunk [2] snippet:**
> "Carer's Allowance — How it works: You could get £86.45 a week if you care for someo…"

**Root cause:** The earnings limit (£196/week as of April 2026) is on the Carer's Allowance eligibility page but was not in the retrieved chunks. The model hallucinates "£160/week" (an older limit from its training data), immediately flags it as unsupported, and then partially refuses. The result is a self-contradictory answer that neither commits to the figure nor guides the user to the correct source.

**Stage:** Retrieval (earnings limit section not retrieved) + Generation (hallucination without full retraction).  
**Fix direction:** Stronger system prompt instruction: "If you are unsure, say so clearly and direct the user to the GOV.UK page. Do not state a figure and immediately retract it."

---

## Summary table

| Case | Question ID | Stage | Failure type | Faith | Rel | Cit | Fix direction |
|------|-------------|-------|-------------|-------|-----|-----|--------------|
| 1 | q004 | Retrieval | Fact in un-retrieved section | 0 | 3 | 0 | Increase `top_k_context` to 10 |
| 2 | q035 | Retrieval | Query matched wrong topic page | 0 | 3 | 0 | Query rewriting / expansion |
| 3 | q044 | Retrieval | Table row not in chunk boundaries | 0 | 3 | 0 | Table-aware chunking |
| 4 | q009 | Retrieval | Duration sub-section not ranked | 0 | 3 | 0 | Increase `top_k_context` |
| 5 | q008 | Generation | Over-cautious refusal on truncated snippet | 3 | 1 | 3 | Longer citation snippets |
| 6 | q039 | Generation | No single chunk gives full definition | — | — | — | Multi-chunk synthesis prompt |
| 7 | q023 | Retrieval | Main page below top-30 cutoff | 1 | 1 | 1 | Topic-based re-ranking |
| 8 | q028 | Retrieval + Generation | Stale figure + partial retraction | 1 | 1 | 1 | Retrieval depth + prompt clarity |

---

## Overall assessment

The pipeline's main weakness is **parametric leakage**: Qwen3-235B-A22B has strong
prior knowledge of UK government figures and uses it when retrieved chunks are topically
correct but lack the specific fact. This is an artefact of citation snippet truncation
(300 chars) and chunk boundaries that often capture section introductions rather than
the numeric tables they precede.

The reranker improves citation quality and relevance slightly (composite 1.78 vs 1.67 for
hybrid_800 without reranking) but cannot compensate for missing facts within chunks.

The highest-impact fix for the next iteration would be **increasing chunk snippet length
from 300 to 600 characters** and **expanding `top_k_context` from 6 to 10** — both
are low-risk, one-line configuration changes that would expose more source text to the
LLM before it falls back on parametric knowledge.
