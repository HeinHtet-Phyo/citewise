"""FastAPI application for Citewise.

Endpoints:
  GET  /          → serve index.html
  GET  /health    → {"status": "ok", "model": ..., "git_sha": ...}
  GET  /configs   → available RAG configs
  POST /ask       → run the RAG pipeline and return an Answer
"""

from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from time import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from citewise.api.budget import DailyBudget
from citewise.config import RagConfig, settings
from citewise.models import Answer
from citewise.rag import answer_question

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# In-memory rate limiter — per-IP requests per minute (RATE_LIMIT_PER_MINUTE, default 10).
_rate_buckets: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = settings.rate_limit_per_minute
_budget = DailyBudget(settings.daily_budget_gbp)
_RATE_WINDOW = 60.0


# Compute once at startup — avoids a subprocess on every health probe.
def _read_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


_GIT_SHA: str = _read_git_sha()


def _client_ip(request: Request) -> str:
    """Caller address: first X-Forwarded-For entry behind Render's proxy, else the socket peer."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _rate_limit(request: Request) -> None:
    """Raise 429 if the caller exceeds the per-minute request limit."""
    ip = _client_ip(request)
    now = time()
    _rate_buckets[ip] = [t for t in _rate_buckets[ip] if now - t < _RATE_WINDOW]
    if len(_rate_buckets[ip]) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429, detail=f"Rate limit: {_RATE_LIMIT} requests/minute. Please wait."
        )
    _rate_buckets[ip].append(now)
    # Prune keys whose buckets are now empty to prevent unbounded dict growth.
    if not _rate_buckets[ip]:
        del _rate_buckets[ip]


app = FastAPI(
    title="Citewise",
    description="UK government guidance, answered with citations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Named configs. dense_400 is the unfixed baseline; L_dense adds answer_v2 and the claim verifier; L2_dense uses answer_v3 (partial answers).
_CONFIGS: dict[str, RagConfig] = {
    "hybrid_400_rerank": RagConfig(
        config_name="hybrid_400_rerank",
        search_mode="hybrid",
        use_reranker=True,
        refusal_threshold=0.3,
    ),
    "dense_400": RagConfig(config_name="dense_400", search_mode="dense", use_reranker=False),
    "L_dense": RagConfig(
        config_name="L_dense",
        search_mode="dense",
        use_reranker=False,
        prompt_version="v2",
        verify_claims=True,
    ),
    "L2_dense": RagConfig(
        config_name="L2_dense",
        search_mode="dense",
        use_reranker=False,
        prompt_version="v3",
        verify_claims=True,
    ),
}
# The deployed config is set by CITEWISE_CONFIG (the 512 MB Render limit decides which fits).
_DEFAULT_CONFIG = settings.default_config


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(..., min_length=5, max_length=500)
    config_name: str = Field(default=_DEFAULT_CONFIG)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model": _CONFIGS[_DEFAULT_CONFIG].llm_model,
            "default_config": _DEFAULT_CONFIG,
            "git_sha": _GIT_SHA,
        }
    )


@app.get("/configs")
async def list_configs() -> JSONResponse:
    return JSONResponse({"configs": list(_CONFIGS.keys()), "default": _DEFAULT_CONFIG})


@app.post("/ask", response_model=Answer)
async def ask(
    body: AskRequest,
    _: None = Depends(_rate_limit),
) -> Answer:
    """Run the RAG pipeline for the given question and return a structured Answer."""
    if body.config_name not in _CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown config '{body.config_name}'. Available: {list(_CONFIGS)}",
        )
    if _budget.exhausted():
        raise HTTPException(
            status_code=503, detail="The demo's daily budget has been used up. Try again tomorrow."
        )
    config = _CONFIGS[body.config_name]
    log.info("POST /ask config=%s question=%.60s", config.config_name, body.question)
    answer = answer_question(body.question, config)
    _budget.add(answer.cost_gbp)
    return answer
