"""Hugging Face Spaces entry point.

HF Spaces expects this file at the root. It simply re-exports the FastAPI app
so that uvicorn (started by the Space's Dockerfile) can find it.
"""

from citewise.api.main import app  # noqa: F401 — re-export for HF Spaces / uvicorn

__all__ = ["app"]
