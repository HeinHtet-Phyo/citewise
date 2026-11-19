# Citewise

**UK government guidance, answered with citations.**

Citewise is a retrieval-augmented generation (RAG) system built over the [GOV.UK](https://www.gov.uk) content APIs. Ask a question about benefits, employment rights, housing, or driving — the system retrieves relevant passages, generates a grounded answer, and cites every claim back to a live GOV.UK page.

Built as a portfolio project demonstrating production-quality RAG engineering for UK AI Engineer roles.

---

## Architecture

```
User question
     │
     ▼
 Guardrails ──── unsafe / out-of-scope ──→ refuse
     │
     ▼
 Qdrant hybrid search (BM25 + dense cosine)
     │  retrieve top-30 candidate chunks
     ▼
 Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
     │  rerank, deduplicate by URL, keep top-6
     ▼
 Qwen3-235B-A22B via Nebius (citation-grounded prompt)
     │
     ▼
 Answer with [n] citation markers + source cards
```

**Tech stack:** Python 3.11 · FastAPI · Pydantic v2 · Qdrant (hybrid BM25 + bge-small-en-v1.5) · fastembed · LiteLLM · Nebius (Qwen3-235B-A22B generator, DeepSeek-V4.1-Flash judge) · tiktoken

---

## Corpus

| Format | Count | Topics |
|--------|-------|--------|
| `answer` | 769 | All |
| `guide` | 860 | All |
| `detailed_guide` | 1,684 | Benefits, employment, housing, transport |
| **Total** | **3,313** | — |

Data from the [GOV.UK Search API](https://docs.publishing.service.gov.uk/repos/search-api/) and [Content API](https://content-api.publishing.service.gov.uk/), licenced under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

---

## Chunk statistics

| Chunk size | Count | Avg tokens |
|------------|-------|-----------|
| 400 | 21,740 | 282 |
| 800 | 12,311 | 486 |

Adjacent short sections are greedily merged before splitting, so the 400- and 800-token sizes produce chunks that differ meaningfully (ratio 1.72×).

---

## Evaluation

Experiments run on 100 development questions across four topics (benefits, employment, housing, transport).
Generator: Qwen3-235B-A22B (Nebius). Judge: DeepSeek-V4.1-Flash (Nebius, different family).
LLM-as-judge scores are 0–3 for faithfulness, relevance, and citation quality.

**Dev split (100 questions, 5 configs):**

| Config | Faithfulness | Relevance | Citation | Composite | Refusal |
|--------|-------------|-----------|----------|-----------|---------|
| dense_400 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| hybrid_400 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| hybrid_400_rerank | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| hybrid_800 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| hybrid_800_rerank | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |

**Test split (50 questions, best config):**

| Config | Faithfulness | Relevance | Citation | Composite | Refusal |
|--------|-------------|-----------|----------|-----------|---------|
| [TBC] | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |

---

## Project structure

```
citewise/
├── src/citewise/
│   ├── config.py          # Settings + RagConfig
│   ├── models.py          # Chunk, Citation, Answer (Pydantic v2)
│   ├── ingest/            # discover → fetch → clean → chunk
│   ├── index/             # embed (bge + BM25) → upsert to Qdrant
│   ├── retrieve.py        # hybrid search + rerank + dedup
│   ├── guardrails.py      # input safety + evidence threshold
│   ├── generate.py        # LiteLLM Nebius call + file-based cache
│   ├── rag.py             # answer_question() entry point
│   ├── prompts/           # system prompt + judge prompt
│   └── api/               # FastAPI app + static UI
├── eval/
│   ├── questions/         # 100 dev + 50 test questions (JSONL)
│   ├── generate_candidates.py
│   ├── judge.py           # DeepSeek-V4.1-Flash-as-judge scorer
│   └── experiments.py     # run all configs, aggregate, print table
├── tests/unit/            # 56 unit tests, all passing
├── docs/                  # data notes, architecture decisions
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Running locally

### Prerequisites

- Python 3.11
- Docker (for local Qdrant)
- A `.env` file with `NEBIUS_API_KEY` and `QDRANT_URL`

```bash
# 1. Set up the virtual environment
make setup

# 2. Download and chunk GOV.UK data (~10 min, one-off)
make ingest

# 3. Start Qdrant and build the index
docker compose up qdrant -d
make index

# 4. Run the app
make run
# → http://localhost:8000
```

### Tests

```bash
make test   # ruff + mypy + 56 unit tests
```

---

## Deploying to Render

The app is deployed to [Render.com](https://render.com) free tier using `render.yaml`.

**Prerequisites before deploying:**
1. Create a free cluster on [Qdrant Cloud](https://cloud.qdrant.io)
2. Upload the index: set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` to the cloud values, then run `make index`
3. Connect your GitHub repo in Render; it will detect `render.yaml` automatically
4. In the Render dashboard, set the following environment variables:
   - `NEBIUS_API_KEY` — your Nebius Token Factory key
   - `QDRANT_URL` — the Qdrant Cloud cluster URL
   - `QDRANT_API_KEY` — the Qdrant Cloud API key

**Memory:** 247 MB idle, 367 MB peak (measured locally) — fits within Render's 512 MB free tier.

**Cold start note:** Render free tier sleeps after 15 minutes of inactivity. The first request after wakeup downloads ~300 MB of fastembed model files and takes ~60 seconds. Subsequent requests are fast.

---

## Design decisions

See [`docs/decisions.md`](docs/decisions.md) for the rationale behind:
- Section-merge chunking strategy
- Hybrid BM25 + dense search
- Cross-encoder reranker choice
- LLM-as-judge evaluation methodology
- Refusal via evidence threshold

---

## Licence

Code: MIT. Data: [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

Built by [Hein Htet Phyo](https://github.com/HeinHtet-Phyo).
