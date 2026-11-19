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

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# PORT env var is injected by Render; falls back to 8000 for local use.
CMD ["sh", "-c", "exec python -m uvicorn citewise.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
