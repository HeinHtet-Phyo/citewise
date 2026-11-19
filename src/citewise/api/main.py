"""FastAPI application for Citewise.

Endpoints:
  GET  /          → serve index.html
  GET  /health    → {"status": "ok"}
  POST /ask       → run the RAG pipeline and return an Answer
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from citewise.config import RagConfig
from citewise.models import Answer
from citewise.rag import answer_question

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Citewise",
    description="UK government guidance, answered with citations.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# Named configs exposed to the UI — mirrors generate_candidates.CONFIGS.
_CONFIGS: dict[str, RagConfig] = {
    "hybrid_400_rerank": RagConfig(
        config_name="hybrid_400_rerank",
        chunk_size=400,
        collection_name="citewise_400",
        search_mode="hybrid",
        top_k_retrieve=30,
        use_reranker=True,
        top_k_context=6,
    ),
    "dense_400": RagConfig(
        config_name="dense_400",
        chunk_size=400,
        collection_name="citewise_400",
        search_mode="dense",
        top_k_retrieve=30,
        use_reranker=False,
        top_k_context=6,
    ),
}
_DEFAULT_CONFIG = "hybrid_400_rerank"


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(..., min_length=5, max_length=500)
    config_name: str = Field(default=_DEFAULT_CONFIG)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/configs")
async def list_configs() -> JSONResponse:
    return JSONResponse({"configs": list(_CONFIGS.keys()), "default": _DEFAULT_CONFIG})


@app.post("/ask", response_model=Answer)
async def ask(request: AskRequest) -> Answer:
    """Run the RAG pipeline for the given question and return a structured Answer."""
    if request.config_name not in _CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown config '{request.config_name}'. Available: {list(_CONFIGS)}",
        )
    config = _CONFIGS[request.config_name]
    log.info("POST /ask config=%s question=%.60s", config.config_name, request.question)
    return answer_question(request.question, config)
