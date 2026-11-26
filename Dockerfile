FROM python:3.11-slim

WORKDIR /app

# Install OS-level build deps for lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install third-party dependencies first (better layer caching).
# Copy only pyproject.toml — src/ is not needed to resolve deps.
COPY pyproject.toml ./
# Create a minimal stub so setuptools can read the package metadata.
RUN mkdir -p src/citewise && touch src/citewise/__init__.py
RUN pip install --no-cache-dir -e .

# Copy real source (overwrites stub)
COPY src/ src/
COPY app.py ./

# Limit ONNX Runtime and linear-algebra libraries to one thread each.
# Without this, they claim all host CPUs (Render shows 32+) and allocate
# a proportionally large memory arena, causing OOM under the 512 MB limit.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    ONNXRUNTIME_MAX_INTEROP_THREADS=1 \
    ONNXRUNTIME_MAX_INTRAOP_THREADS=1 \
    FASTEMBED_CACHE_PATH=/app/models

# Bake the embedding and reranker models into the image so a cold start does not download them.
# threads=1 matches the runtime limit set above.
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
p=[('CPUExecutionProvider',{'arena_extend_strategy':'kSameAsRequested'})]; \
TextEmbedding('BAAI/bge-small-en-v1.5', threads=1, providers=p); \
SparseTextEmbedding('Qdrant/bm25', threads=1, providers=p); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', threads=1, providers=p)"

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# PORT env var is injected by Render; falls back to 8000 for local use.
# --workers 1: a second worker would double resident memory on a 512 MB host.
CMD ["sh", "-c", "exec python -m uvicorn citewise.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
