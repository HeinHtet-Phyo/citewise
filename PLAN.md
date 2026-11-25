# PLAN.md — Citewise build plan

**Project:** Citewise — UK government guidance, answered with citations.
**Owner:** Hein Htet Phyo
**Goal:** A live, measured, deployed RAG system that proves production AI engineering skills to UK hiring managers.

---

## Interface contract

See `CLAUDE.md` for the full model definitions. Summary:

- `Chunk` — one indexed piece of GOV.UK text with a stable ID, URL, section, tokens, format, and date.
- `Citation` — a source reference with title, section, URL, snippet, and date.
- `Answer` — the full pipeline output: question, answer text with `[n]` markers, citations, refusal flag, latency, cost, trace ID.
- `answer_question(question: str, config: RagConfig) -> Answer` — the single entry point in `src/citewise/rag.py`.

---

## Repository structure

```
citewise/
├── README.md
├── PLAN.md                     (this file)
├── CLAUDE.md                   project rules for all agents
├── LICENSE                     MIT (code); data is OGL v3.0
├── .env.example
├── .gitignore
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .github/workflows/
│   ├── ci.yml
│   └── deploy.yml
├── src/citewise/
│   ├── config.py
│   ├── models.py
│   ├── ingest/                 discover.py, fetch.py, clean.py, chunk.py, refresh.py
│   ├── index/                  embed.py, build.py
│   ├── retrieve.py
│   ├── generate.py
│   ├── guardrails.py
│   ├── rag.py
│   ├── observability.py
│   ├── prompts/                answer_v1.txt
│   └── api/                    main.py, static/
├── eval/
│   ├── questions/              dev.jsonl, test.jsonl, README.md
│   ├── generate_candidates.py
│   ├── run_eval.py
│   ├── metrics.py
│   ├── judge.py
│   ├── experiments.py
│   └── results/
├── scripts/
├── tests/
│   ├── fixtures/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── architecture.md
│   ├── data_notes.md
│   ├── eval_method.md
│   ├── decisions.md
│   └── failure_analysis.md
├── images/
└── private/                    (not committed)
```

---

## Phases

### Phase 0 — Setup (done when PLAN.md exists and tests pass on empty repo)
- [x] Python 3.11 venv created at `.venv/`
- [x] `.gitignore` with `.env`, `data/raw/`, `data/cache/`, `.llm_cache/`, `.venv/`, `private/`
- [x] `.env.example` with all required variable names
- [x] `CLAUDE.md` written
- [x] `PLAN.md` written (this file)
- [ ] `.env` checked for required variable names; missing ones listed
- [ ] `pyproject.toml` with pinned deps and tool config
- [ ] `Makefile` with setup/ingest/index/eval/test/run/docker targets
- [ ] Directory skeleton created
- [ ] `src/citewise/models.py` and `config.py` written
- [ ] Initial commit on `main`

### Phase 1 — Ingestion (sequential; everything depends on it)
Branch: `feat/phase-1-ingest`

**Files:** `src/citewise/ingest/discover.py`, `fetch.py`, `clean.py`, `chunk.py`, `refresh.py`
**Output:** `data/manifest.csv`, `data/raw/<path>.json` (cached), `data/processed/chunks_400.jsonl`, `data/processed/chunks_800.jsonl`
**Docs:** `docs/data_notes.md`
**Tests:** `tests/unit/test_clean.py`, `tests/unit/test_chunk.py`

Steps:
1. `discover.py` — Search API, formats `answer`+`guide`+`detailed_guide` (filtered topics), save manifest.
2. `fetch.py` — Content API, polite (5 req/s, retries, backoff, User-Agent), cache to `data/raw/`.
3. `clean.py` — HTML→text via BeautifulSoup, keep headings as section markers, strip nav/boilerplate.
4. `chunk.py` — Heading-first split, then size (400 tokens / 60 overlap default), stable IDs, save JSONL.
5. `refresh.py` — Diff manifest timestamps vs stored; re-fetch+re-chunk changed pages.

Target: 3,000–6,000 pages, real count in `docs/data_notes.md`.

Commit: `feat(ingest): discover, fetch, clean, chunk GOV.UK pages`

---

### Phase 2 — Index and retrieval (sequential)
Branch: `feat/phase-2-index`

**Files:** `src/citewise/index/embed.py`, `index/build.py`, `retrieve.py`
**Output:** Qdrant collection(s) locally, `docs/retrieval_sanity.md`
**Tests:** `tests/integration/test_retrieve.py` (fixture index)

Steps:
1. Start Qdrant with `docker compose up -d qdrant`.
2. `build.py` — create collection with named dense vector (BAAI/bge-small-en-v1.5, 384-dim) + named sparse (BM25), payload indexes, batch upsert, idempotent.
3. `retrieve.py` — `search()` (dense or hybrid+RRF), `rerank()` (cross-encoder via fastembed), deduplication.
4. Sanity check: 10 hand-written questions, top 5 results each, saved to `docs/retrieval_sanity.md`.

Commit: `feat(index): build hybrid Qdrant index and retrieval pipeline`
**Push to GitHub after this phase.**

---

### Phase 3 — Generation and guardrails (parallel with 4 and 5-skeleton; Subagent A)
Branch: `feat/phase-3-rag`

**Files:** `src/citewise/generate.py`, `guardrails.py`, `rag.py`, `prompts/answer_v1.txt`, `observability.py`
**Tests:** `tests/unit/test_guardrails.py`, `tests/unit/test_generate.py`

Key behaviours:
- Prompt instructs: answer only from numbered sources, plain English, `[n]` markers, structured JSON output.
- Input guardrails: max length 500 chars, PII detection (NI number, card number), injection resistance.
- Low-evidence refusal: skip LLM if best reranker score < threshold (saves cost).
- Out-of-scope refusal: clearly non-GOV.UK questions refused with `out_of_scope`.
- Citation validation: every `[n]` must refer to a provided source; strip invalid markers.
- LLM cache: keyed on `hash(model + prompt)`, stored in `.llm_cache/`.
- LiteLLM for all LLM calls; cost tracked per call.

---

### Phase 4 — Evaluation set (parallel; Subagent B)
Branch: `feat/phase-4-eval-set`

**Files:** `eval/questions/dev.jsonl`, `eval/questions/test.jsonl`, `eval/questions/README.md`, `eval/generate_candidates.py`

Target: 150 questions (50 dev / 100 test), stratified by type:
- ~90 single-page factual (answerable)
- ~15 multi-page (answerable)
- ~30 unanswerable (must refuse)
- ~15 adversarial

Fields per question: `id`, `question`, `type`, `topic`, `reference_answer`, `gold_urls`, `must_refuse`, `notes`.

**Human check:** `eval/questions/hand_check_sample.csv` (30 questions) — tell Hein when ready.

---

### Phase 5 — Metrics, experiments, tuning (sequential; needs Phase 3+4)
Branch: `feat/phase-5-experiments`

**Files:** `eval/run_eval.py`, `eval/metrics.py`, `eval/judge.py`, `eval/experiments.py`, `eval/results/`
**Docs:** `docs/decisions.md`, `docs/failure_analysis.md`

Configs:

| Config | Chunk | Search | Reranker | Model |
|--------|-------|--------|----------|-------|
| A | 400 | dense | no | Gemini fast |
| B | 400 | hybrid | no | Gemini fast |
| C | 400 | hybrid | yes | Gemini fast |
| D | 800 | hybrid | yes | Gemini fast |
| E | best(C/D) | hybrid | yes | Nebius strong |
| F (opt.) | best(C/D) | hybrid+Nebius embed | yes | Gemini fast |

Sequence:
1. Run A–E on **dev split** only. Estimate cost first; ask if >$5.
2. Tune refusal threshold on dev (plot false vs missed refusal rate).
3. Choose final config on dev.
4. Run all configs on **test split once**. These are the headline numbers.
5. Write `eval/results/summary.csv`, `summary.md`, charts.

Metrics: hit@k, MRR, judge-scored correctness, faithfulness/made-up rate, citation correctness, refusal recall, false refusal rate, adversarial pass rate, p50/p95 latency, cost per question in GBP.

**Judge check CSV:** 30 answers for Hein to hand-label — tell Hein when ready.

---

### Phase 6 — API, UI, observability (parallel with Phase 5 tail; Subagent C)
Branch: `feat/phase-6-api`

**Files:** `src/citewise/api/main.py`, `api/static/index.html`, `style.css`, `app.js`

Endpoints:
- `POST /api/ask` → `Answer` JSON
- `GET /api/health` → status, index size, model, git SHA
- `GET /` → web page

Protection: rate limit (10/min, 100/day per IP), daily budget cap, request timeouts, CORS locked to own origin, no stack traces in errors.

UI: question box, 4 example questions, answer with clickable `[n]` markers, sources list, refusal message, latency+cost footer, "How it works" panel with eval numbers from JSON, unofficial-project notice, mobile-friendly.

---

### Phase 7 — Docker, CI/CD, deployment (sequential; needs Phase 6)
Branch: `feat/phase-7-deploy`

- Dockerfile: multi-stage, slim, non-root, models baked in at build time, health check.
- `docker-compose.yml`: app + Qdrant.
- `ci.yml`: ruff → mypy → pytest (unit+integration) → smoke eval (20 dev questions, accuracy gate).
- `deploy.yml`: on push to main after CI, deploy to Hugging Face Spaces.
- Cloud index: `make index-cloud` upserts to Qdrant Cloud.
- Live check: health endpoint, 5 real questions, 1 refusal, rate limit test.

---

### Phase 8 — README (sequential; needs results)
Full spec in `CLAUDE.md`. Key sections: live link, GIF, results table, Mermaid diagram, failures, run-locally steps. All numbers from `eval/results/summary.csv`.

---

### Phase 9 — Private interview notes (not committed)
File: `private/interview_notes.md`
Contents: 60-second pitch, 3 CV bullet options, 25 interview Q&A, what-I'd-do-with-more-time list, glossary.

---

### Phase 10 — Final review and push
- Fresh-clone smoke test.
- Independent reviewer subagent (no prior context): checks numbers, live link, tests, secrets, README quality.
- Secret scan.
- GitHub repo description and topics set.
- Final summary to Hein: live link, repo link, headline numbers, total LLM spend.

---

## Multi-agent plan

```
Phase 0+1+2  →  you (main agent), sequential
                       ↓
        ┌──────────────┼──────────────────┐
        A              B                  C
   RAG pipeline    Eval set          App shell +
   (Phase 3)      (Phase 4)         CI skeleton
                                     (Phase 6+7 start)
        └──────────────┼──────────────────┘
                       ↓
                   Phase 5 (you: experiments, needs A+B)
                       ↓
                   Phase 7 (wire real pipeline into app)
                       ↓
        ┌──────────────┼
        D              you
   Docs draft      README + GIF
        └──────────────┼
                   Phase 9+10
```

Each subagent receives: the interface contract, the folder structure, CLAUDE.md, and the rule that every number comes from code that ran.

---

## Technical stack (pinned at setup)

| Component | Library | Version |
|-----------|---------|---------|
| Python | — | 3.11.x |
| Web framework | fastapi | ≥0.115 |
| Data models | pydantic | ≥2.9 |
| HTTP client | httpx | ≥0.27 |
| HTML parsing | beautifulsoup4 | ≥4.12 |
| Vector DB client | qdrant-client | ≥1.12 |
| Embeddings + sparse | fastembed | ≥0.4 |
| LLM routing | litellm | ≥1.52 |
| Testing | pytest | ≥8.3 |
| Lint/format | ruff | ≥0.7 |
| Type checking | mypy | ≥1.13 |
| Config | pydantic-settings | ≥2.6 |
| ASGI server | uvicorn | ≥0.32 |
| Tracing | langfuse | ≥2.0 |

Exact pinned versions will be in `pyproject.toml` after `uv pip compile` or equivalent.

---

## Final numbers (test split, 128 questions)

| Config | Correct | Fully faithful | Made-up | Refusal recall | False refusal | Hit@6 | p50 ms |
|---|---|---|---|---|---|---|---|
| hybrid_400_rerank | 73.5% [64–82%] | 71.6% | 20.0% [12–28%] | 96.7% | 4.1% | 92.9% | 528 |
| dense_400 | 69.4% [60–78%] | 83.3% | 8.3% [3–14%] | 93.3% | 4.1% | 89.8% | 934 |
| **L2_dense (deployed)** | **65.3% [56–74%]** | **83.0%** | **7.4% [3–13%]** | **96.7%** | **5.1%** | **89.8%** | **1,330** |

95% bootstrap CIs (1,000 resamples, seed 42). Full CIs in `eval/results/bootstrap_cis_test.json`.

## Budget tracker

| Phase | Actual |
|-------|--------|
| Question generation (Qwen3-235B) | ~$0.10 |
| Dev + test eval runs (Qwen3-235B generator + Qwen3-30B verifier) | ~£0.41 |
| Judge calls (DeepSeek-V4.1-Flash) | not separately metered |
| **Total Nebius** | **well under $30** |

Gemini: $0 (free tier only).

---

## Status: COMPLETE

All phases done as of 2026-11-25. Remaining optional ideas:

- Human-validate a sample of the 190 ground-truth questions (the main honesty gap).
- Enlarge the dev set to 150+ questions so single-answer noise is under 2 points.
- Re-run eval after corpus updates to track metric drift.
- Add a nightly Langfuse monitor alert for unusual refusal rates.
- Persistent rate-limit counters (Redis) so a Render restart cannot reset them.
- Fill `human_ok` in both hand-check CSVs and record the demo GIF.
