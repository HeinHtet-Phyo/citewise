# Citewise

[![CI](https://github.com/HeinHtet-Phyo/citewise/actions/workflows/ci.yml/badge.svg)](https://github.com/HeinHtet-Phyo/citewise/actions/workflows/ci.yml)

**UK government guidance, answered with citations.**

**Live demo: https://citewise-v5w7.onrender.com/** (free tier: the first request after idle can take a while while the server wakes up)

<!-- TODO(Hein): record a short demo GIF and put it at images/demo.gif, then uncomment:
![Citewise demo](images/demo.gif)
-->

Citewise answers questions about benefits, employment, housing and driving using 3,313 GOV.UK
pages. Every claim carries a numbered citation to a live GOV.UK page, and the system refuses when
the evidence is weak. The focus of the project is **measuring** a RAG system honestly, not just
building one.

## Results (held-out test split, 128 questions)

All figures come from `eval/results/summary_test.json`; denominators are in `summary.md`.

| Config | Correct (of 98 answerable) | Fully faithful | Made-up rate | Refusal recall (of 30) | False refusal | Hit@6 | p50 latency |
|---|---|---|---|---|---|---|---|
| hybrid_400_rerank (best retrieval) | 73.5% | 71.6% | 20.0% | 96.7% | 4.1% | 92.9% | 528 ms |
| dense_400 | 69.4% | 83.3% | 8.3% | 93.3% | 4.1% | 89.8% | 934 ms |
| **L_dense (deployed)** | 69.4% | 80.6% | 6.5% | 93.3% | 7.1% | 89.8% | 1,533 ms |

"Made-up" means the answer states a fact that is not in the cited sources (parametric leakage;
see below). Development-split numbers, including the other experiments, are in
[`eval/results/summary.md`](eval/results/summary.md).

![Correctness by config](images/correctness_by_config.png)
![Made-up rate by config](images/made_up_by_config.png)

### The main finding: parametric leakage

The model sometimes states a figure from its own memory and cites a chunk that does not contain
it. I measured it, then tried fixes one at a time on the **dev** split (62 questions):

| Config (dev) | What changed | Correct (answerable) | Made-up rate |
|---|---|---|---|
| dense_400 | baseline | 80.9% | 12.8% |
| hybrid_400_rerank | + hybrid search and reranker | 80.9% | 6.5% |
| B_v2 | + stricter answer prompt | 74.5% | 13.0% |
| C_verify | + sentence-level claim verifier | 70.2% | 4.5% |
| L_dense | dense + stricter prompt + verifier | 72.3% | 6.7% |

Honest reading: the verifier lowers made-up answers on dev but costs correctness, and on the test
split the benefit is small (L_dense 6.5% against dense_400 8.3%). The hybrid config's made-up rate
rose from 6.5% on dev to 20.0% on test, which shows how noisy a 46-answer dev sample is. I did not
re-pick the config on test results.

## Limitations (read this)

- **No human validated the ground truth.** The 190 questions and reference answers were
  machine-generated, machine-verified (the key figure or phrase must appear in the gold page text),
  and cross-checked by a second model from a different family. Five questions were dropped by this
  process; some may have been false alarms. See [`docs/eval_method.md`](docs/eval_method.md).
- **The judge is a model.** Judge-to-judge agreement on 30 answers (DeepSeek vs Gemma-3-27B):
  correctness 83.3% (kappa 0.50), fully faithful 73.3% (kappa 0.25), made-up 83.3% (kappa 0.00).
  The made-up cross-check is weak.
- **Small samples.** Dev has 62 questions (47 answerable) and test has 128 (98 answerable); expect
  differences of a few points to be noise. No confidence intervals yet.
- **The fixes are not shown to help on test.** See above.
- **Live behaviour:** some answerable questions are refused (for example State Pension age and tenancy notice periods in my spot checks), in line with the 7.1% test false-refusal rate. The API lists every retrieved source as a citation, not only the ones the answer uses.
- **Live-service limits are in memory:** the rate limiter (10 per minute) and the daily spend cap
  (£0.50) reset on restart.

## How it works

```mermaid
flowchart LR
    Q[Question] --> G{Guardrails}
    G -- unsafe or out of scope --> R[Refuse]
    G --> S[Qdrant dense search<br/>bge-small, 21,740 chunks]
    S --> T{Evidence threshold}
    T -- weak --> R
    T --> L[Qwen3-235B answer<br/>prompt v2, numbered sources]
    L --> V[Claim verifier<br/>Qwen3-30B per sentence]
    V -- unsupported --> X[Regenerate once, drop sentences,<br/>or refuse]
    V -- supported --> A[Answer with citations]
    X --> A
```

The deployed config (`L_dense`) uses dense retrieval only. The best retrieval config adds BM25
hybrid search and a cross-encoder reranker, but peaked at 1,944.6 MiB in my measurement, which does
not fit Render's 512 MB free tier.

| Deployed config | Idle | Peak (10 requests) | Cold start | First request |
|---|---|---|---|---|
| L_dense | 245.1 MiB | 375.0 MiB | 2.9 s | 2.2 s |
| hybrid_400_rerank | 245.4 MiB | 1,944.6 MiB | 1.7 s | n/a |

Measured locally in Docker (`eval/results/memory.json`). Render's real wake-up time is not measured [TBC].

**Stack:** Python 3.11 · FastAPI · Pydantic v2 · Qdrant · fastembed · LiteLLM · Nebius
(Qwen3-235B generator, Qwen3-30B verifier, DeepSeek-V4.1-Flash judge) · Docker · GitHub Actions.

**Corpus:** 3,313 GOV.UK pages (769 answer, 860 guide, 1,684 detailed guide), section-merge
chunked into 21,740 chunks of up to 400 tokens, under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Failure analysis

[`docs/failure_analysis.md`](docs/failure_analysis.md) annotates 16 real cases (some cover several questions) from the test split
with root causes, for example:

- `q160`: the baseline applied a 6-month rule from a neighbouring page; the verifier removed it.
- `q043`: a citation to an insolvency page was stretched into a general redundancy rule.
- `q015`: the verifier refused a question the baseline answered correctly (a false refusal).
- `q246`: an instruction to fabricate was obeyed by plain dense retrieval.
- `q005`, `q171`, `q187`: apparent retrieval misses where the gold URL itself is questionable.
- `q127`: made-up flag after the verifier passed the answer (probable judge strictness).
- `q096`: the repair step removed useful content.
- `q218`: a must-refuse question answered by all three configs.

## Engineering

- 110 tests (unit and integration), `ruff`, `mypy --strict`, run in GitHub Actions on every push.
- An optional smoke-eval CI job runs only when a `NEBIUS_API_KEY` repository secret is set.
- Rate limit (10 per minute per client) and a daily spend cap that returns HTTP 503 when exhausted.
- Embedding and reranker models are baked into the Docker image so there is no download at start-up.
- Deterministic Qdrant point IDs (`uuid5` of the chunk id) so re-indexing never duplicates.
- LLM responses are cached on disk during development; cost per answer is tracked.

Spend: the headline configs cost about £0.0004 to £0.0006 per answer (the larger-context dev experiments up to £0.0015). Generation and verification in the logged runs came to about £0.41 (`docs/eval_method.md`) and building the questions about $0.10 (`eval/questions/verification_report.json`); judge and AI-check calls are not metered [TBC].

## Project structure

```
src/citewise/   config, models, ingest/, index/, retrieve, guardrails, generate, rag, prompts/, api/
eval/           questions/, generate_candidates, judge, metrics, experiments, ai_check, results/
tests/          unit/ and integration/
docs/           eval_method, decisions, data_notes, failure_analysis
```

## Running locally

```bash
make setup                   # venv and dependencies
make ingest                  # discover, fetch, clean, chunk (about 10 minutes)
docker compose up qdrant -d
make index                   # embed and upsert
make run                     # http://localhost:8000
make test                    # ruff, mypy, pytest
make eval                    # dev-split experiments
```

You need a `.env` with `NEBIUS_API_KEY` and `QDRANT_URL` (and `QDRANT_API_KEY` for Qdrant Cloud).
Deployment is on Render via `render.yaml` with Qdrant Cloud; set `CITEWISE_CONFIG` to choose the config.

## More

- [`docs/decisions.md`](docs/decisions.md): architecture decision records
- [`docs/eval_method.md`](docs/eval_method.md): how the eval set and judge work
- [`docs/data_notes.md`](docs/data_notes.md): corpus notes

Code: MIT. Data: Open Government Licence v3.0.

Built by Hein Htet Phyo · [heinhtetphyo.com](https://heinhtetphyo.com) ·
[LinkedIn](https://www.linkedin.com/in/hein-htet-phyo) · [GitHub](https://github.com/HeinHtet-Phyo)
