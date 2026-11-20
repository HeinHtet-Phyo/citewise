# Evaluation questions

190 questions with ground truth, built by `eval/build_questions.py`
(seed 20261119). Schema: `eval/question_schema.py`.

## How they were made

1. The original 150 questions (archived in `original_questions.jsonl`)
   were each matched to candidate GOV.UK passages by BM25 over `data/processed/chunks_400.jsonl`.
2. `openai/Qwen/Qwen3-235B-A22B-Instruct-2507` (Nebius) drafted a 1-3 sentence reference answer, a question type and
   verbatim evidence quotes with URLs. 64 further answerable questions were
   drafted the same way from seed passages sampled with the fixed seed.
3. Every draft was machine-checked: each evidence quote must appear verbatim (after whitespace
   and punctuation normalisation) in the text of its gold page, the gold URL must exist in the
   corpus, and every figure in the reference answer must appear in the gold page text.
   Drafts that failed were dropped, not repaired. 86 of the original
   questions survived.
4. 45 unanswerable and adversarial questions were written by hand
   (`build_questions_refusals.py`), each with a note giving the reason for refusal.
5. Questions were split into dev and test, stratified by type and topic, with the fixed seed.

### AI check (no human validation)

`eval/ai_check.py` re-checks every answerable reference answer against the gold page text: a programmatic
check that every number in the answer appears in the page, plus a third-family model
(`google/gemma-3-27b-it`, neither the Qwen generator nor the DeepSeek judge) that reads the page and
says whether the answer is supported and answers the question. 145 of 150 passed. The 5 that
failed (q077, q134, q137, q152, q154) were dropped, not repaired; some may have been false alarms,
but dropping is the cautious choice. Results and reasons are in `ai_check.json`.

Dropped by verification (before the AI check):

- `existing:model_says_not_answerable`: 22
- `existing:number_not_in_page`: 3
- `existing:quote_not_in_page`: 37
- `existing:quote_too_short`: 2
- `new:bad_topic`: 1
- `new:model_says_not_answerable`: 9
- `new:number_not_in_page`: 1
- `new:quote_not_in_page`: 39
- `new:quote_too_short`: 2

## Counts (computed by code)

### By split
| Group | Questions |
|---|---|
| dev | 62 |
| test | 128 |

### By type
| Group | Questions |
|---|---|
| adversarial | 15 |
| how_to | 15 |
| multi_page | 29 |
| single_fact | 101 |
| unanswerable | 30 |

### By topic
| Group | Questions |
|---|---|
| benefits | 65 |
| employment | 46 |
| housing | 36 |
| transport | 43 |

### Split by type
| Group | Questions |
|---|---|
| dev/adversarial | 5 |
| dev/how_to | 5 |
| dev/multi_page | 10 |
| dev/single_fact | 32 |
| dev/unanswerable | 10 |
| test/adversarial | 10 |
| test/how_to | 10 |
| test/multi_page | 19 |
| test/single_fact | 69 |
| test/unanswerable | 20 |

### Split by topic
| Group | Questions |
|---|---|
| dev/benefits | 22 |
| dev/employment | 16 |
| dev/housing | 12 |
| dev/transport | 15 |
| test/benefits | 44 |
| test/employment | 31 |
| test/housing | 27 |
| test/transport | 28 |

## Dev/test rule

Dev and test do not overlap. Tune thresholds and configurations on dev only; report final
numbers from a single test run. Never choose anything based on test results.

## Limitations

- Reference answers are LLM-drafted. The machine check confirms the quoted evidence and the
  figures exist in the gold page, not that the answer is complete or perfectly phrased.
- Gold URLs list the page(s) the evidence was taken from; other pages may also answer the question.
- The "unanswerable" questions were chosen by hand and only
  spot-checked against the corpus.
- **No human has validated this ground truth.** It was machine-verified against the source pages and
  cross-checked by a second model (the AI check above). `hand_check_questions.csv` holds 30 sampled
  answerable questions with the `ai_check` result and a reason for each. A human review would be
  the natural next step.
- Nebius spend for this build: $0.1028 over 266 calls (first, uncached run; reruns use the cache).
