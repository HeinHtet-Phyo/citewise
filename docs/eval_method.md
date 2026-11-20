# How Citewise is evaluated

All numbers in the README come from `eval/results/` (produced by `eval/experiments.py`). Nothing here
was typed in by hand.

## Questions and ground truth

- 190 questions: 62 dev, 128 test, stratified by type and topic (`eval/questions/README.md`).
- Types: single fact, multi page, how-to, unanswerable (must refuse), adversarial (must not obey).
- Each answerable question has a reference answer and one or more gold URLs.

### Limitation: no human validation

**No human has checked this ground truth.** It was verified by machine only:

1. The reference answers were drafted by a model (Qwen3-235B) and kept only if every evidence quote
   appeared verbatim in the gold page and every figure in the answer appeared in that page's text.
2. A second, independent check (`eval/ai_check.py`) re-tested all 150 answerable questions with a
   programmatic figure check plus a third model family (Gemma 3 27B, not Qwen and not DeepSeek).
   145 passed; the 5 that failed were dropped, not repaired.
3. The 45 unanswerable and adversarial questions were written by me and only spot-checked against
   the corpus.

This catches wrong figures and unsupported answers, but not answers that are incomplete, badly
phrased, or answerable from a page that is not in the gold URL list. A human review is the obvious next step.

## Judge

- DeepSeek-V4.1-Flash judges each non-refused answer once with strict JSON output
  (`src/citewise/prompts/judge_combined.txt`): correctness vs the reference, whether every claim is
  supported by the sources it was given, whether any figure was made up, and safety.
- The judge is a different family from the generator (Qwen) to limit self-preference bias.
- It is still an LLM: it can be wrong, and results move by a few points between runs.

### Judge-to-judge agreement (not judge-to-human)

I did not have humans label answers. Instead, 30 answered test questions from the
`hybrid_400_rerank` run were re-judged by Gemma 3 27B with the same prompt
(`eval/judge_agreement.py`, results in `eval/results/judge_agreement.json`):

| Field | Agreement | Cohen's kappa |
|---|---|---|
| Correctness (3 classes) | 83.3% | 0.50 |
| Fully faithful | 73.3% | 0.25 |
| Made up | 83.3% | 0.00 |

Reading it honestly: the two models mostly agree on correctness, agree less on faithfulness, and the
made-up kappa of 0.00 means Gemma almost never flagged an answer as made up, so Gemma is a weak
cross-check for that metric. This shows how consistent two models are, not that either is right.

## Metrics

Retrieval: hit rate at 6 (a gold URL is in the context) and MRR. Answers: correct, partly correct,
fully faithful, made-up rate, citation correctness. Refusals: refusal recall (unanswerable questions
refused), false refusal rate (answerable questions refused), adversarial pass rate. Plus p50 and p95
latency and cost per answer. Denominators are printed next to every number in `eval/results/summary.md`.

## Dev and test discipline

- Configs, prompts and thresholds were chosen on the dev split only.
- The test split was run once per config: `hybrid_400_rerank`, `dense_400` and `L_dense`, then once more for `L2_dense` (the refusal fix). `L2_dense` was designed on dev and a 20-question spot check, and its test run was not used to change it.
- Selection rule fixed before the test run: highest correct% of answerable questions with made-up
  rate at most 10% and refusal recall at least 90% on dev. That rule picks `hybrid_400_rerank`, whose
  dev made-up rate (6.5%) turned out to be 20.0% on test. With about 46 answered dev questions the dev
  estimate was too noisy. I did not re-pick on test.
- The live demo runs `L2_dense` (`L_dense` with a partial-answer prompt) because `hybrid_400_rerank` needs about 1.9 GB of memory (see below).

## Small samples

Dev has 47 answerable and 15 refusal-required questions, so one answer moves a metric by 2 to 7 points.
Treat differences under about 5 points as noise.

## Cost

Generation and verification cost for every run in `eval/results/` sums to about £0.41 (from the
`cost_gbp` field of each answer). Question building cost $0.10 (`verification_report.json`).
Judge and AI-check costs were not separately metered: [TBC].
