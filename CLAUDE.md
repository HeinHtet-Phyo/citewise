# CLAUDE.md — Citewise project rules

Every agent (main or subagent) working in this repo must follow these rules without exception.

---

## Identity

**Project:** Citewise — UK government guidance, answered with citations.
**Owner:** Hein Htet Phyo (HeinHtet-Phyo on GitHub)
**Purpose:** Portfolio project demonstrating production-quality RAG engineering for UK AI Engineer roles.

---

## Hard rules (never break these)

1. **No secrets ever.** Never print, log, echo, cat, or commit `.env` or any file containing real keys. Run `grep -rE '(sk-|AIza|hf_|ey[A-Za-z0-9])[A-Za-z0-9_-]{10,}' --include="*.py" --include="*.md" --include="*.json" src/ eval/ docs/` before every commit.
2. **No invented numbers.** Every metric (accuracy, latency, cost, chunk count, page count) must come from code that ran and a results file in the repo. Use `[TBC]` until measured.
3. **No overwriting `.env`.** Check names only (`cut -d= -f1 .env`). Never read or print values.
4. **British spelling** in all prose, docstrings, comments, and documentation: "analyse", "behaviour", "organisation", "licence", "colour".
5. **Dev/test split is sacred.** Never tune thresholds or choose configs based on test-split results. Tune on dev, report final numbers from one test-split run.

---

## Code style

- Python 3.11, type hints everywhere, Pydantic v2 models for all data structures.
- Small functions: one clear purpose, a one-line docstring, explicit parameter and return types.
- **No file over ~300 lines.** Split by responsibility if approaching the limit.
- Config in one place: `src/citewise/config.py`, read from env vars with sensible defaults via `pydantic-settings`.
- No hard-coded paths or keys anywhere outside `config.py`.
- `ruff` for formatting and linting (line length 100). `mypy --strict` on `src/`.
- Imports: stdlib → third-party → local, separated by blank lines.

---

## Interface contract

All agents must code against these models exactly. Do not change field names or types without updating this file first.

```python
# src/citewise/models.py
class Chunk(BaseModel):
    chunk_id: str          # stable: f"{base_path}#{section_slug}-{n}"
    url: str               # https://www.gov.uk/<path>[#anchor or /part-slug]
    title: str
    section: str | None
    text: str
    format: str            # answer | guide | detailed_guide
    updated_at: str        # ISO date from public_updated_at
    tokens: int

class Citation(BaseModel):
    n: int
    title: str
    section: str | None
    url: str
    snippet: str           # under 300 chars
    updated_at: str

class Answer(BaseModel):
    question: str
    answer: str            # plain English with [n] markers
    citations: list[Citation]
    refused: bool
    refusal_reason: str | None   # "low_evidence" | "out_of_scope" | "unsafe" | None
    config_name: str
    latency_ms: int
    cost_gbp: float
    retrieved_chunk_ids: list[str]
    trace_id: str | None
```

`answer_question(question: str, config: RagConfig) -> Answer` in `src/citewise/rag.py` is the single entry point for the pipeline.

---

## Subagent rules

Each subagent works **only in its assigned files**. It must not modify files outside its scope.

| Subagent | Owns |
|----------|------|
| A — RAG pipeline | `src/citewise/generate.py`, `guardrails.py`, `rag.py`, `prompts/`, `retrieve.py` |
| B — Eval set | `eval/questions/`, `eval/generate_candidates.py` |
| C — App shell | `src/citewise/api/`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |
| D — Docs | `docs/architecture.md`, `docs/decisions.md`, `docs/failure_analysis.md` |

Before finishing, every subagent must:
- Run `ruff check` and `ruff format --check` on its files.
- Run `mypy src/` if it touched `src/`.
- Run `pytest tests/` (unit only if no index is available).
- Report exactly what it built and any `[TBC]` items.

---

## Running the project

```bash
# Setup
make setup          # creates .venv, installs deps

# Data pipeline
make ingest         # discover + fetch + clean + chunk
make index          # embed + upsert into Qdrant (local Docker)

# Evaluation
make eval           # run experiments on dev split

# Tests
make test           # ruff + mypy + pytest

# Local app
make run            # starts Docker Compose (app + Qdrant)

# Docker
make docker         # build and run the Docker image
```

---

## Commit style

Conventional commits, lowercase subject, under 72 characters:

```
feat(ingest): discover and fetch GOV.UK pages with caching
fix(retrieve): deduplicate chunks from same page before rerank
test(guardrails): add injection and PII unit tests
docs(eval): add failure analysis with 8 annotated cases
```

Commit at the end of each phase. Work on short-lived branches; merge to main when tests pass. Push to GitHub after Phase 2.

---

## Cost and budget

- Gemini: free tier only. On rate limit, back off and retry; never switch to a paid tier.
- Nebius: total project spend under $30. Estimate cost before any large run; ask Hein if a single run exceeds $5.
- LLM responses are cached in `.llm_cache/` (keyed on model + prompt hash) during development. Cache is on by default; set `LLM_CACHE=false` to disable.

---

## Security checklist before every push

```bash
grep -rE '(sk-|AIza|hf_|ey[A-Za-z0-9])[A-Za-z0-9_-]{10,}' \
  --include="*.py" --include="*.md" --include="*.json" --include="*.txt" \
  src/ eval/ docs/ tests/ scripts/ pyproject.toml README.md PLAN.md CLAUDE.md
```

If anything matches, stop and investigate before pushing.

---

## What Hein can explain in an interview

This constraint drives every design choice: if a component is here, it must be explainable. Prefer simple and readable over clever and compact. If you would not be comfortable being asked "why did you write it this way?" — rewrite it.
